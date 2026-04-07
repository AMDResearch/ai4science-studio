#!/usr/bin/env bash
# MATEY distributed training on AMD MI300X nodes via SLURM.
#
# Modelled after the upstream Frontier demo script:
#   https://github.com/ORNL/MATEY/blob/main/examples/submit_JHTDB_demo.sh
#
# Prerequisites
# -------------
# 1. Run build_sif.sh once to create the SIF and writable overlay:
#      ./build_sif.sh
#    This pulls the ROCm image, converts it to a SIF, and installs MATEY
#    into a writable ext3 overlay.  Set MATEY_SIF and MATEY_OVERLAY to
#    non-default paths before running if needed.
#
# 2. Set MATEY_DATA_DIR to the location of your HDF5 training data.
#    Set MATEY_YAML to your config file.
#
# Submit:
#   export MATEY_SIF=/path/to/matey.sif
#   export MATEY_OVERLAY=/path/to/matey_overlay.img   # optional, created by build_sif.sh
#   sbatch sbatch_train_mi300x.sh
#
# Bare-metal fallback (no container):
#   MATEY_BARE_METAL=1 sbatch sbatch_train_mi300x.sh
#   (You must activate your ROCm environment before submitting.)

#SBATCH --job-name=matey-train
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=04:00:00
#SBATCH --output=matey-train-%j.out
#SBATCH --error=matey-train-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
MATEY_SIF="${MATEY_SIF:-}"
MATEY_OVERLAY="${MATEY_OVERLAY:-}"           # writable overlay created by build_sif.sh
MATEY_BARE_METAL="${MATEY_BARE_METAL:-0}"   # set to 1 to skip Apptainer entirely
MATEY_YAML="${MATEY_YAML:-/matey/config/Demo_JHUTDB_TT.yaml}"
MATEY_RUN_NAME="${MATEY_RUN_NAME:-matey_slurm_${SLURM_JOB_ID:-local}}"
MATEY_CONFIG="${MATEY_CONFIG:-basic_config}"
MATEY_DATA_DIR="${MATEY_DATA_DIR:-/data/JHTDB}"

# MIOpen cache (use NVMe scratch if available)
MIOPEN_CACHE="${MIOPEN_CACHE:-/tmp/miopen_cache_${SLURM_JOB_ID:-$$}}"
export OMP_NUM_THREADS=1
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE}"
mkdir -p "${MIOPEN_CACHE}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ "${MATEY_BARE_METAL}" != "1" && -z "${MATEY_SIF}" ]]; then
    echo "error: MATEY_SIF is not set." >&2
    echo "" >&2
    echo "  Build the SIF first:" >&2
    echo "    cd $(dirname "$0") && ./build_sif.sh" >&2
    echo "" >&2
    echo "  Then submit:" >&2
    echo "    export MATEY_SIF=/path/to/matey.sif" >&2
    echo "    export MATEY_OVERLAY=/path/to/matey_overlay.img  # if using overlay" >&2
    echo "    sbatch sbatch_train_mi300x.sh" >&2
    echo "" >&2
    echo "  To skip Apptainer and use a bare-metal env instead:" >&2
    echo "    MATEY_BARE_METAL=1 sbatch sbatch_train_mi300x.sh" >&2
    exit 1
fi

if [[ "${MATEY_BARE_METAL}" != "1" && ! -f "${MATEY_SIF}" ]]; then
    echo "error: SIF file not found: ${MATEY_SIF}" >&2
    echo "  Run ./build_sif.sh to create it." >&2
    exit 1
fi

echo "=== MATEY distributed training ==="
echo "  Run name : ${MATEY_RUN_NAME}"
echo "  YAML     : ${MATEY_YAML}"
echo "  Data dir : ${MATEY_DATA_DIR}"
echo "  Nodes    : ${SLURM_JOB_NUM_NODES:-2}"
echo "  Tasks    : ${SLURM_NTASKS:-16}"
echo "  Job      : ${SLURM_JOB_ID:-local}"

TRAIN_CMD="python /matey-src/basic_usage.py \
    --run_name ${MATEY_RUN_NAME} \
    --config   ${MATEY_CONFIG} \
    --yaml_config ${MATEY_YAML} \
    --use_ddp"

# Build optional overlay flag
OVERLAY_FLAG=""
if [[ -n "${MATEY_OVERLAY}" && -f "${MATEY_OVERLAY}" ]]; then
    OVERLAY_FLAG="--overlay ${MATEY_OVERLAY}"
fi

if [[ "${MATEY_BARE_METAL}" == "1" ]]; then
    echo "  Runtime: bare-metal"
    echo "  WARNING: ensure your ROCm + MATEY environment is activated before submitting."
    srun -c7 --gpu-bind=closest \
        bash -c "${TRAIN_CMD}"
else
    echo "  Runtime: Apptainer (${MATEY_SIF})"
    [[ -n "${OVERLAY_FLAG}" ]] && echo "  Overlay: ${MATEY_OVERLAY}"
    # shellcheck disable=SC2086
    srun -c7 --gpu-bind=closest \
        apptainer exec \
            --rocm \
            ${OVERLAY_FLAG} \
            --bind "${SCRIPT_DIR}:/workspace" \
            --bind "${MATEY_DATA_DIR}:/data" \
            "${MATEY_SIF}" \
            bash -c "${TRAIN_CMD}"
fi

echo "=== Done ==="
