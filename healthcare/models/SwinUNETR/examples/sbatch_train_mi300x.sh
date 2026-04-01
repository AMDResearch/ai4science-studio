#!/usr/bin/env bash
# SwinUNETR training on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Build an Apptainer/Singularity SIF from the ROCm 6.4 PyTorch base image
#    (rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0) and install
#    MONAI[all] + nibabel inside it. Or run interactively via docker_run.sh.
# 2. NSCLC-Radiomics data is auto-downloaded by MONAI on first run.
# 3. MIOpen auto-tuning gives ~3× speedup; it is default on ROCm 6.4+ / PyTorch 2.6+
#    and set explicitly in run_train.sh via MIOPEN_FIND_MODE / MIOPEN_FIND_ENFORCE.
#
# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#
# See: ../recipes/train/README.md

#SBATCH --job-name=swinunetr-train
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=swinunetr-train-%j.out
#SBATCH --error=swinunetr-train-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
SU_DATA_DIR="${SU_DATA_DIR:-/data}"
SU_CKPT_DIR="${SU_CKPT_DIR:-./checkpoints}"
SU_MAX_EPOCHS="${SU_MAX_EPOCHS:-700}"
SU_ROI_X="${SU_ROI_X:-96}"
SU_ROI_Y="${SU_ROI_Y:-96}"
SU_ROI_Z="${SU_ROI_Z:-96}"         # MI300X 192 GB supports up to 480×480×96
SU_NUM_WORKERS="${SU_NUM_WORKERS:-64}"  # >32 eliminates DataLoader bottleneck

# --- Singularity / Apptainer SIF path ---
SU_SIF="${SU_SIF:-}"

echo "=== SwinUNETR training ==="
echo "  Data       : ${SU_DATA_DIR}  (MONAI auto-downloads NSCLC-Radiomics)"
echo "  Checkpoints: ${SU_CKPT_DIR}"
echo "  ROI size   : ${SU_ROI_X}×${SU_ROI_Y}×${SU_ROI_Z}"
echo "  Max epochs : ${SU_MAX_EPOCHS}"
echo "  Workers    : ${SU_NUM_WORKERS}"
echo "  Job        : ${SLURM_JOB_ID:-local}"

if [[ -n "${SU_SIF}" ]]; then
    echo "  Runtime: Apptainer (${SU_SIF})"
    apptainer exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${SU_SIF}" \
        bash /workspace/run_train.sh
else
    echo "  Runtime: bare-metal (activate your ROCm 6.4 env before submitting)"
    DATA_DIR="${SU_DATA_DIR}" CKPT_DIR="${SU_CKPT_DIR}" \
    MAX_EPOCHS="${SU_MAX_EPOCHS}" \
    ROI_X="${SU_ROI_X}" ROI_Y="${SU_ROI_Y}" ROI_Z="${SU_ROI_Z}" \
    NUM_WORKERS="${SU_NUM_WORKERS}" \
        bash "${SCRIPT_DIR}/run_train.sh"
fi

echo "=== Done ==="
