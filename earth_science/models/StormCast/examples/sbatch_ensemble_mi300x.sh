#!/usr/bin/env bash
# StormCast ensemble inference on a single AMD MI300X via SLURM.
#
# Prerequisites
# -------------
# 1. Pull the ROCm PyTorch container and build a Singularity/Apptainer SIF, or
#    use Docker directly on an interactive node (see recipes/ensemble/README.md).
# 2. pip install "earth2studio[stormcast]" cartopy   (inside the container)
# 3. Set SC_START, SC_STEPS, SC_MEMBERS below, or export them before submitting:
#       SC_START=2025-08-09T12 SC_STEPS=12 SC_MEMBERS=8 sbatch sbatch_ensemble_mi300x.sh
#
# Memory note: each additional ensemble member adds ~9.6 GiB VRAM.
# The MI300X has 192 GiB HBM — up to ~16 members fit comfortably.
#
# Adjust #SBATCH directives to match your site's partition name and account.
#
# See: ../recipes/ensemble/README.md

#SBATCH --job-name=stormcast-ens
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=stormcast-ens-%j.out
#SBATCH --error=stormcast-ens-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Configuration (override by exporting before sbatch) ---
SC_START="${SC_START:-2025-08-09T12}"
SC_STEPS="${SC_STEPS:-12}"
SC_MEMBERS="${SC_MEMBERS:-4}"
SC_OUTPUT="${SC_OUTPUT:-}"          # leave empty for default outputs/ens-<date>.zarr

# --- Singularity / Apptainer path (set SC_SIF or use bare-metal fallback) ---
SC_SIF="${SC_SIF:-}"

# Build argument list
EXTRA_ARGS=()
if [[ -n "${SC_OUTPUT}" ]]; then
    EXTRA_ARGS+=(--output "${SC_OUTPUT}")
fi

echo "=== StormCast ensemble inference ==="
echo "  Start  : ${SC_START}"
echo "  Steps  : ${SC_STEPS}"
echo "  Members: ${SC_MEMBERS}"
echo "  Job    : ${SLURM_JOB_ID:-local}"

if [[ -n "${SC_SIF}" ]]; then
    # --- Singularity / Apptainer path ---
    echo "  Runtime: Singularity/Apptainer (${SC_SIF})"
    singularity exec \
        --rocm \
        --bind "${SCRIPT_DIR}:/workspace" \
        "${SC_SIF}" \
        python /workspace/run_ensemble.py \
            --start "${SC_START}" \
            --steps "${SC_STEPS}" \
            --members "${SC_MEMBERS}" \
            "${EXTRA_ARGS[@]}"
else
    # --- Bare-metal / module path ---
    # Uncomment the lines appropriate for your site:
    # source ~/miniconda3/etc/profile.d/conda.sh && conda activate stormcast
    # module load rocm
    echo "  Runtime: bare-metal (activate your env before submitting)"
    python "${SCRIPT_DIR}/run_ensemble.py" \
        --start "${SC_START}" \
        --steps "${SC_STEPS}" \
        --members "${SC_MEMBERS}" \
        "${EXTRA_ARGS[@]}"
fi

echo "=== Done ==="
