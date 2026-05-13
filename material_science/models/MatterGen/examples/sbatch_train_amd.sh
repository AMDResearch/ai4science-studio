#!/usr/bin/env bash
# MatterGen training on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Build an Apptainer/Singularity SIF from the rocm/pytorch base image and
#    install MatterGen + ROCm forks of pytorch_scatter/pytorch_sparse (see docker_run.sh).
#    Or run interactively via Docker (see docker_run.sh).
# 2. Full mp_20 dataset is downloaded automatically by run_train.sh on first run
#    (requires Git LFS inside the container).
#
# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#
# See: ../recipes/train/README.md

#SBATCH --job-name=mattergen-train
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=mattergen-train-%j.out
#SBATCH --error=mattergen-train-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
MG_DATA_DIR="${MG_DATA_DIR:-./datasets}"
MG_OUTPUT_DIR="${MG_OUTPUT_DIR:-./checkpoints}"
MG_MAX_EPOCHS="${MG_MAX_EPOCHS:-900}"   # ~15 hours on single MI300X

# --- Singularity / Apptainer SIF path ---
MG_SIF="${MG_SIF:-}"

echo "=== MatterGen training ==="
echo "  Dataset    : ${MG_DATA_DIR}"
echo "  Checkpoints: ${MG_OUTPUT_DIR}"
echo "  Max epochs : ${MG_MAX_EPOCHS}  (~15 hours on single MI300X)"
echo "  Job        : ${SLURM_JOB_ID:-local}"

if [[ -n "${MG_SIF}" ]]; then
    echo "  Runtime: Apptainer (${MG_SIF})"
    GPU_OK=$(apptainer exec --rocm "${MG_SIF}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$GPU_OK" != "True" ]]; then
        echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
        echo "  Try a newer ROCm SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
    fi
    apptainer exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${MG_SIF}" \
        bash /workspace/run_train.sh
else
    echo "  Runtime: bare-metal (activate your env before submitting)"
    DATA_DIR="${MG_DATA_DIR}" OUTPUT_DIR="${MG_OUTPUT_DIR}" \
    MAX_EPOCHS="${MG_MAX_EPOCHS}" \
        bash "${SCRIPT_DIR}/run_train.sh"
fi

echo "=== Done ==="
