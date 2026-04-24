#!/usr/bin/env bash
# MatterGen crystal generation (inference) on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Build an Apptainer/Singularity SIF from the rocm/pytorch base image and
#    install MatterGen + ROCm forks of pytorch_scatter/pytorch_sparse (see docker_run.sh).
#    Or run interactively via Docker (see docker_run.sh).
# 2. Set MG_PRETRAINED_NAME to the checkpoint you want to use (default: mattergen_base).
#
# Adjust #SBATCH directives to match your site's partition, account, and runtime.
#
# See: ../recipes/inference/README.md

#SBATCH --job-name=mattergen-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=mattergen-infer-%j.out
#SBATCH --error=mattergen-infer-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
MG_PRETRAINED_NAME="${MG_PRETRAINED_NAME:-mattergen_base}"
MG_BATCH_SIZE="${MG_BATCH_SIZE:-16}"
MG_NUM_BATCHES="${MG_NUM_BATCHES:-1}"
MG_OUTPUT_DIR="${MG_OUTPUT_DIR:-./results}"
MG_PROPERTIES="${MG_PROPERTIES:-}"
MG_GUIDANCE_FACTOR="${MG_GUIDANCE_FACTOR:-2.0}"

# --- Singularity / Apptainer SIF path ---
MG_SIF="${MG_SIF:-}"

echo "=== MatterGen generation ==="
echo "  Checkpoint  : ${MG_PRETRAINED_NAME}"
echo "  Batch size  : ${MG_BATCH_SIZE}"
echo "  Num batches : ${MG_NUM_BATCHES}"
echo "  Output      : ${MG_OUTPUT_DIR}"
[[ -n "${MG_PROPERTIES}" ]] && echo "  Properties  : ${MG_PROPERTIES}"
echo "  Job         : ${SLURM_JOB_ID:-local}"

if [[ -n "${MG_SIF}" ]]; then
    echo "  Runtime: Apptainer (${MG_SIF})"
    apptainer exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${MG_SIF}" \
        bash /workspace/run_inference.sh
else
    echo "  Runtime: bare-metal (activate your env before submitting)"
    PRETRAINED_NAME="${MG_PRETRAINED_NAME}" \
    BATCH_SIZE="${MG_BATCH_SIZE}" \
    NUM_BATCHES="${MG_NUM_BATCHES}" \
    OUTPUT_DIR="${MG_OUTPUT_DIR}" \
    PROPERTIES="${MG_PROPERTIES}" \
    GUIDANCE_FACTOR="${MG_GUIDANCE_FACTOR}" \
        bash "${SCRIPT_DIR}/run_inference.sh"
fi

echo "=== Done ==="
