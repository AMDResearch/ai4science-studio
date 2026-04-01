#!/usr/bin/env bash
# ArchesWeather inference on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Build the Apptainer/Singularity SIF from the silogen/ai-samples Dockerfile
#    (ai4sciences/geoarches-training) or use Docker on an interactive node.
# 2. Stage the ERA5 test dataset (~35 GB for one year) at $DATA_PATH.
# 3. Download or train a checkpoint; set AW_CHECKPOINT before submitting:
#       AW_CHECKPOINT=/path/to/checkpoint sbatch sbatch_inference_mi300x.sh
#
# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#
# See: ../recipes/inference/README.md

#SBATCH --job-name=archesweather-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=archesweather-infer-%j.out
#SBATCH --error=archesweather-infer-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
AW_MODEL_NAME="${AW_MODEL_NAME:-archesweather-m-seed0}"
AW_YEAR="${AW_YEAR:-2020}"
AW_DATA_PATH="${AW_DATA_PATH:-/data/era5_240/full}"
AW_OUTPUT_PATH="${AW_OUTPUT_PATH:-./results/predictions}"
AW_CHECKPOINT="${AW_CHECKPOINT:-}"   # leave empty to auto-download from HF

# --- Singularity / Apptainer SIF path (set AW_SIF or fall back to bare metal) ---
AW_SIF="${AW_SIF:-}"

echo "=== ArchesWeather inference ==="
echo "  Model      : ${AW_MODEL_NAME}"
echo "  Test year  : ${AW_YEAR}"
echo "  Data       : ${AW_DATA_PATH}"
echo "  Output     : ${AW_OUTPUT_PATH}"
echo "  Job        : ${SLURM_JOB_ID:-local}"

RUN_CMD=(
    python -m geoarches.inference.encode_dataset
    "++checkpoint_path=${AW_CHECKPOINT}"
    "++dataloader.dataset.path=${AW_DATA_PATH}"
    "++dataloader.dataset.start_year=${AW_YEAR}"
    "++dataloader.dataset.end_year=${AW_YEAR}"
    "++output_path=${AW_OUTPUT_PATH}"
)

if [[ -n "${AW_SIF}" ]]; then
    echo "  Runtime: Apptainer (${AW_SIF})"
    apptainer exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${AW_SIF}" \
        "${RUN_CMD[@]}"
else
    # Bare-metal / module path — activate your environment before submitting:
    # source ~/miniconda3/etc/profile.d/conda.sh && conda activate geoarches
    # module load rocm
    echo "  Runtime: bare-metal (activate your env before submitting)"
    "${RUN_CMD[@]}"
fi

echo "=== Done ==="
