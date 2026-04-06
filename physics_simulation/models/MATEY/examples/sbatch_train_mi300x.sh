#!/usr/bin/env bash
# MATEY distributed training on AMD MI300X nodes via SLURM.
#
# Modelled after the upstream Frontier demo script:
#   https://github.com/ORNL/MATEY/blob/main/examples/submit_JHTDB_demo.sh
#
# Prerequisites
# -------------
# 1. Build or pull a Singularity/Apptainer SIF from:
#      rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0
#    Set MATEY_SIF below or export it before submitting.
# 2. Install MATEY inside the container:
#      pip install -e /matey
# 3. Set MATEY_DATA_DIR to the location of your HDF5 training data.
#    Set MATEY_YAML to your config file.
#
# Submit:
#   sbatch sbatch_train_mi300x.sh
#   MATEY_SIF=/path/to/matey.sif MATEY_YAML=/path/to/config.yaml sbatch sbatch_train_mi300x.sh

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

echo "=== MATEY distributed training ==="
echo "  Run name : ${MATEY_RUN_NAME}"
echo "  YAML     : ${MATEY_YAML}"
echo "  Data dir : ${MATEY_DATA_DIR}"
echo "  Nodes    : ${SLURM_JOB_NUM_NODES:-2}"
echo "  Tasks    : ${SLURM_NTASKS:-16}"
echo "  Job      : ${SLURM_JOB_ID:-local}"

TRAIN_CMD="python /matey/basic_usage.py \
    --run_name ${MATEY_RUN_NAME} \
    --config   ${MATEY_CONFIG} \
    --yaml_config ${MATEY_YAML} \
    --use_ddp"

if [[ -n "${MATEY_SIF}" ]]; then
    echo "  Runtime: Apptainer (${MATEY_SIF})"
    srun -c7 --gpu-bind=closest \
        apptainer exec \
            --rocm \
            --bind "${SCRIPT_DIR}:/workspace" \
            --bind "${MATEY_DATA_DIR}:/data" \
            "${MATEY_SIF}" \
            bash -c "${TRAIN_CMD}"
else
    echo "  Runtime: bare-metal (activate your env before submitting)"
    # Uncomment as appropriate for your site:
    # source ~/miniconda3/etc/profile.d/conda.sh && conda activate matey
    # module load rocm
    srun -c7 --gpu-bind=closest \
        bash -c "${TRAIN_CMD}"
fi

echo "=== Done ==="
