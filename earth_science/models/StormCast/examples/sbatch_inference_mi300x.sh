#!/usr/bin/env bash
# StormCast deterministic inference on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Pull the ROCm PyTorch container and build a Singularity/Apptainer SIF, or
#    use Docker directly on an interactive node (see recipes/inference/README.md).
# 2. pip install "earth2studio[stormcast]" cartopy   (inside the container)
# 3. Set SC_START and SC_STEPS below, or export them before submitting:
#       SC_START=2025-01-01T06 SC_STEPS=6 sbatch sbatch_inference_mi300x.sh
#
# Adjust #SBATCH directives to match your site's partition name, account, and
# container runtime (Singularity, Apptainer, or Docker).
#
# See: ../recipes/inference/README.md

#SBATCH --job-name=stormcast-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=stormcast-infer-%j.out
#SBATCH --error=stormcast-infer-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
SC_START="${SC_START:-2025-01-01T06}"
SC_STEPS="${SC_STEPS:-6}"
SC_OUTPUT="${SC_OUTPUT:-}"          # leave empty for default outputs/pred-<date>.zarr

# --- Singularity / Apptainer path (set SC_SIF or use Docker fallback) ---
SC_SIF="${SC_SIF:-}"
ROCM_IMAGE="rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0"

# Build argument list
EXTRA_ARGS=()
if [[ -n "${SC_OUTPUT}" ]]; then
    EXTRA_ARGS+=(--output "${SC_OUTPUT}")
fi

echo "=== StormCast deterministic inference ==="
echo "  Start : ${SC_START}"
echo "  Steps : ${SC_STEPS}"
echo "  Job   : ${SLURM_JOB_ID:-local}"

if [[ -n "${SC_SIF}" ]]; then
    # --- Singularity / Apptainer path ---
    echo "  Runtime: Singularity/Apptainer (${SC_SIF})"
    singularity exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${SC_SIF}" \
        python /workspace/run_inference.py \
            --start "${SC_START}" \
            --steps "${SC_STEPS}" \
            "${EXTRA_ARGS[@]}"
else
    # --- Bare-metal / module path ---
    # Uncomment the lines appropriate for your site:
    # source ~/miniconda3/etc/profile.d/conda.sh && conda activate stormcast
    # module load rocm
    echo "  Runtime: bare-metal (activate your env before submitting)"
    python "${SCRIPT_DIR}/run_inference.py" \
        --start "${SC_START}" \
        --steps "${SC_STEPS}" \
        "${EXTRA_ARGS[@]}"
fi

echo "=== Done ==="
