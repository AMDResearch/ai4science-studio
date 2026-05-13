#!/usr/bin/env bash
# SwinUNETR optimized inference on a single AMD MI300X via SLURM.
# AMP + torch.compile(max-autotune) delivers ~2.9× speedup vs baseline.
#
# Prerequisites
# -------------
# 1. Build an Apptainer/Singularity SIF from the ROCm 7.0 PyTorch base image
#    (rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0) and install
#    MONAI[all] + nibabel inside it. Or run interactively via docker_run.sh infer.
# 2. Set SU_CHECKPOINT to your trained .pth checkpoint before submitting.
#
# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#
# See: ../recipes/inference/README.md

#SBATCH --job-name=swinunetr-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=swinunetr-infer-%j.out
#SBATCH --error=swinunetr-infer-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
SU_CHECKPOINT="${SU_CHECKPOINT:-}"      # REQUIRED: path to trained .pth file
SU_INPUT_DIR="${SU_INPUT_DIR:-/data/test}"
SU_OUTPUT_DIR="${SU_OUTPUT_DIR:-./results}"
SU_ROI_X="${SU_ROI_X:-96}"
SU_ROI_Y="${SU_ROI_Y:-96}"
SU_ROI_Z="${SU_ROI_Z:-96}"
SU_USE_COMPILE="${SU_USE_COMPILE:-1}"  # 1 = torch.compile max-autotune (~2.9× total)

if [[ -z "${SU_CHECKPOINT}" ]]; then
    echo "ERROR: SU_CHECKPOINT is not set."
    echo "  Export it before submitting, e.g.:"
    echo "  SU_CHECKPOINT=/path/to/model.pth sbatch sbatch_inference_amd.sh"
    exit 1
fi

# --- Singularity / Apptainer SIF path ---
SU_SIF="${SU_SIF:-}"

echo "=== SwinUNETR inference ==="
echo "  Checkpoint   : ${SU_CHECKPOINT}"
echo "  Input        : ${SU_INPUT_DIR}"
echo "  Output       : ${SU_OUTPUT_DIR}"
echo "  ROI size     : ${SU_ROI_X}×${SU_ROI_Y}×${SU_ROI_Z}"
echo "  torch.compile: $([ "${SU_USE_COMPILE}" == "1" ] && echo "max-autotune (~2.9× speedup)" || echo "disabled")"
echo "  Job          : ${SLURM_JOB_ID:-local}"

if [[ -n "${SU_SIF}" ]]; then
    echo "  Runtime: Apptainer (${SU_SIF})"
    GPU_OK=$(apptainer exec --rocm "${SU_SIF}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$GPU_OK" != "True" ]]; then
        echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
        echo "  Try a newer ROCm SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
    fi
    apptainer exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${SU_SIF}" \
        bash /workspace/run_inference.sh
else
    echo "  Runtime: bare-metal (activate your ROCm 7.0 env before submitting)"
    CHECKPOINT="${SU_CHECKPOINT}" \
    INPUT_DIR="${SU_INPUT_DIR}" \
    OUTPUT_DIR="${SU_OUTPUT_DIR}" \
    ROI_X="${SU_ROI_X}" ROI_Y="${SU_ROI_Y}" ROI_Z="${SU_ROI_Z}" \
    USE_COMPILE="${SU_USE_COMPILE}" \
        bash "${SCRIPT_DIR}/run_inference.sh"
fi

echo "=== Done ==="
