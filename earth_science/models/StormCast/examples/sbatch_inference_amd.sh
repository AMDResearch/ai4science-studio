#!/usr/bin/env bash
# StormCast deterministic inference on AMD Instinct via SLURM + Apptainer.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export SC_SIF=/path/to/rocm_pytorch.sif
#   export SC_START=2025-01-01T06 SC_STEPS=6
#   sbatch sbatch_inference_amd.sh
#
#   # Faster subsequent runs (skip ~5-min pip install):
#   export SC_OVERLAY=/path/to/stormcast-overlay.img   # built once with build_overlay_amd.sh
#   sbatch sbatch_inference_amd.sh
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   1. Pull the ROCm PyTorch container and build an Apptainer SIF:
#        apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#      Recommended image: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
#   2. (Optional) Build a persistent overlay with pre-installed deps:
#        sbatch build_overlay_amd.sh   # runs once, creates stormcast-overlay.img
#      Without an overlay, deps are installed on each job start (~5 min overhead).
#   3. Adjust #SBATCH directives below for your site's partition and account.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   SC_SIF        Path to Apptainer SIF image (required for container mode)
#   SC_OVERLAY    Path to pre-built ext3 overlay (optional, skips pip install)
#   SC_START      Forecast start time ISO-8601, e.g. 2025-01-01T06 (default: 2025-01-01T06)
#   SC_STEPS      Number of 1-h inference steps (default: 6)
#   SC_OUTPUT     Output zarr path (default: outputs/pred-<start>.zarr next to script)
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   rocm7.2.2 image covers MI250X (gfx90a), MI300X (gfx942), and MI350X (gfx950).
#   For older hardware (MI100/gfx908): use a rocm6.x image.
#   For future hardware: update SC_SIF to the matching ROCm image.
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

# ---------------------------------------------------------------------------
# Configuration (override by exporting before sbatch)
# ---------------------------------------------------------------------------
SC_START="${SC_START:-2025-01-01T06}"
SC_STEPS="${SC_STEPS:-6}"
SC_OUTPUT="${SC_OUTPUT:-}"          # leave empty for default outputs/pred-<date>.zarr

# Apptainer / bare-metal selection
SC_SIF="${SC_SIF:-}"

# Optional pre-built overlay (skips pip install on each job)
SC_OVERLAY="${SC_OVERLAY:-}"

# ---------------------------------------------------------------------------
# Resolve output path and wipe any previous run's zarr
# (ZarrBackend refuses to overwrite an existing store)
# ---------------------------------------------------------------------------
if [[ -z "${SC_OUTPUT}" ]]; then
    SAFE_START="${SC_START//[: T]/-}"
    SC_OUTPUT="${SCRIPT_DIR}/outputs/pred-${SAFE_START}.zarr"
fi
echo "  Output zarr : ${SC_OUTPUT}"
if [[ -e "${SC_OUTPUT}" ]]; then
    echo "  Removing previous output: ${SC_OUTPUT}"
    rm -rf "${SC_OUTPUT}"
fi

# Build argument list
EXTRA_ARGS=(--output "${SC_OUTPUT}")

echo "=== StormCast deterministic inference ==="
echo "  Start : ${SC_START}"
echo "  Steps : ${SC_STEPS}"
echo "  Job   : ${SLURM_JOB_ID:-local}"

# ---------------------------------------------------------------------------
# Apptainer path
# ---------------------------------------------------------------------------
if [[ -n "${SC_SIF}" ]]; then
    echo "  Runtime : Apptainer (${SC_SIF})"

    OVERLAY_ARG=()
    if [[ -n "${SC_OVERLAY}" ]] && [[ -f "${SC_OVERLAY}" ]]; then
        OVERLAY_ARG=(--overlay "${SC_OVERLAY}:ro")
        echo "  Overlay : ${SC_OVERLAY}"
    else
        echo "  No overlay — deps will be installed at job start (~5 min)"
    fi

    apptainer exec \
        --rocm \
        "${OVERLAY_ARG[@]}" \
        --bind "${SCRIPT_DIR}:/workspace" \
        --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
        "${SC_SIF}" \
        bash -c '
source /opt/venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}"

# Install deps if not pre-baked in an overlay
if ! python3 -c "import earth2studio, cartopy" 2>/dev/null; then
    echo "--- Installing earth2studio[stormcast] and cartopy ---"
    pip install -q --no-cache-dir "earth2studio[stormcast]" cartopy 2>&1 | tail -5
fi

python /workspace/run_inference.py \
    --start "'"${SC_START}"'" \
    --steps "'"${SC_STEPS}"'" \
    '"${EXTRA_ARGS[*]}"'
'

# ---------------------------------------------------------------------------
# Bare-metal / module path
# ---------------------------------------------------------------------------
else
    echo "  Runtime: bare-metal (activate your env before submitting)"
    # Uncomment whichever applies to your site:
    # source ~/miniconda3/etc/profile.d/conda.sh && conda activate stormcast
    # module load rocm
    python "${SCRIPT_DIR}/run_inference.py" \
        --start "${SC_START}" \
        --steps "${SC_STEPS}" \
        "${EXTRA_ARGS[@]}"
fi

echo "=== Done ==="
echo "  Output: ${SC_OUTPUT}"
