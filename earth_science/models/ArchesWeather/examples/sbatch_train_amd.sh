#!/usr/bin/env bash
# ArchesWeather pretraining / fine-tuning on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Build the Apptainer/Singularity SIF from the silogen/ai-samples Dockerfile
#    (ai4sciences/geoarches-training) or use Docker on an interactive node.
# 2. Stage the full ERA5 dataset (~735 GB) at $AW_DATA_PATH.
# 3. Set AW_PHASE=pretrain (default) or AW_PHASE=finetune; for finetune also set
#    AW_LOAD_FROM to a pretrained checkpoint path.
#
# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#
# See: ../recipes/train/README.md

#SBATCH --job-name=archesweather-train
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=archesweather-train-%j.out
#SBATCH --error=archesweather-train-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
AW_MODEL="${AW_MODEL:-archesweather}"       # archesweather | archesweathergen
AW_PHASE="${AW_PHASE:-pretrain}"            # pretrain | finetune
AW_SEED="${AW_SEED:-0}"
AW_PRECISION="${AW_PRECISION:-16-mixed}"   # 16-mixed (batch 8) or 32-true (batch 5) on MI300X
AW_BATCH_SIZE="${AW_BATCH_SIZE:-8}"
AW_DATA_PATH="${AW_DATA_PATH:-/data/era5_240/full}"
AW_LOAD_FROM="${AW_LOAD_FROM:-}"           # required when AW_PHASE=finetune

# --- Singularity / Apptainer SIF path ---
AW_SIF="${AW_SIF:-}"

echo "=== ArchesWeather training ==="
echo "  Model      : ${AW_MODEL}"
echo "  Phase      : ${AW_PHASE}"
echo "  Seed       : ${AW_SEED}"
echo "  Precision  : ${AW_PRECISION}  (MI300X: batch 5 at 32-true, 8 at 16-mixed)"
echo "  Batch size : ${AW_BATCH_SIZE}"
echo "  Data       : ${AW_DATA_PATH}"
echo "  Job        : ${SLURM_JOB_ID:-local}"

# Default max steps per phase
if [[ "${AW_PHASE}" == "pretrain" ]]; then
    AW_MAX_STEPS="${AW_MAX_STEPS:-$([ "${AW_MODEL}" == "archesweathergen" ] && echo 200000 || echo 250000)}"
    AW_NAME="${AW_NAME:-${AW_MODEL}-seed${AW_SEED}}"
else
    AW_MAX_STEPS="${AW_MAX_STEPS:-$([ "${AW_MODEL}" == "archesweathergen" ] && echo 60000 || echo 50000)}"
    AW_NAME="${AW_NAME:-${AW_MODEL}-seed${AW_SEED}-finetuned}"
fi

RUN_CMD=(
    python -m geoarches.main_hydra "++log=True"
    dataloader=era5
    "module=${AW_MODEL}"
    "++name=${AW_NAME}"
    "++cluster.precision=${AW_PRECISION}"
    "++batch_size=${AW_BATCH_SIZE}"
    "++max_steps=${AW_MAX_STEPS}"
    "++save_step_frequency=50000"
    "++dataloader.dataset.path=${AW_DATA_PATH}"
)
[[ -n "${AW_LOAD_FROM}" ]] && RUN_CMD+=("++load_from=${AW_LOAD_FROM}")

if [[ -n "${AW_SIF}" ]]; then
    echo "  Runtime: Apptainer (${AW_SIF})"
    GPU_OK=$(apptainer exec --rocm "${AW_SIF}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$GPU_OK" != "True" ]]; then
        echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
        echo "  Try a newer ROCm SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
    fi
    apptainer exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${AW_SIF}" \
        "${RUN_CMD[@]}"
else
    echo "  Runtime: bare-metal (activate your env before submitting)"
    "${RUN_CMD[@]}"
fi

echo "=== Done ==="
