#!/usr/bin/env bash
# HydraGNN Predictive GFM 2024 — inference on AMD Instinct via SLURM (Apptainer).
#
# Prerequisites:
#   1. Build the overlay once:   sbatch build_overlay_amd.sh
#   2. Download weights:
#        huggingface-cli download mlupopa/HydraGNN_Predictive_GFM_2024 \
#            --include "Ensemble_of_models/gfm_0.229/*" \
#            --local-dir /path/to/hydragnn-weights
#
# Quick-start:
#   export HG_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
#   export HG_OVERLAY=/path/to/hydragnn-overlay.img
#   export HG_CHECKPOINT=/path/to/gfm_0.229.pk
#   export HG_CONFIG=/path/to/config.json
#   sbatch sbatch_infer_amd.sh
#
# Key environment variables:
#   HG_SIF          Path to Apptainer SIF image
#                   (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif)
#   HG_OVERLAY      Path to pre-built ext3 overlay from build_overlay_amd.sh
#                   (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN/overlays/hydragnn-overlay.img)
#   HG_CHECKPOINT   Path to .pk checkpoint file
#                   (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN/weights/gfm_0.229.pk)
#   HG_CONFIG       Path to matching config.json
#                   (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN/weights/config.json)
#   HG_OUTPUT_DIR   Where to write prediction outputs
#                   (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN/outputs)

#SBATCH -J hydragnn-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o hydragnn-infer-%j.out
#SBATCH -e hydragnn-infer-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# SCRIPT_DIR — works both at submit time and inside the SLURM job
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# ---------------------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------------------
HG_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/HydraGNN"
HG_SIF="${HG_SIF:-${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
HG_OVERLAY="${HG_OVERLAY:-${HG_BASE}/overlays/hydragnn-overlay.img}"
HG_CHECKPOINT="${HG_CHECKPOINT:-${HG_BASE}/weights/gfm_0.229.pk}"
HG_CONFIG="${HG_CONFIG:-${HG_BASE}/weights/config.json}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-${HG_BASE}/outputs}"

for var in HG_SIF HG_OVERLAY HG_CHECKPOINT HG_CONFIG; do
  if [[ -z "${!var:-}" ]]; then
    echo "error: $var must be set" >&2
    exit 2
  fi
done

if [[ ! -f "$HG_OVERLAY" ]]; then
  echo "error: overlay not found: $HG_OVERLAY" >&2
  echo "  Build it first: sbatch build_overlay_amd.sh" >&2
  exit 2
fi
mkdir -p "$HG_OUTPUT_DIR"

# ---------------------------------------------------------------------------
# ROCm env
# ---------------------------------------------------------------------------
export HSA_NO_SCRATCH_RECLAIM=1
export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/hydragnn-miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ---------------------------------------------------------------------------
# Sanity-check GPU visibility
# ---------------------------------------------------------------------------
GPU_OK=$(apptainer exec --rocm --overlay "${HG_OVERLAY}:ro" "$HG_SIF" \
    python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$GPU_OK" != "True" ]]; then
  echo "WARNING: torch.cuda.is_available() = False — check SIF or try HSA_OVERRIDE_GFX_VERSION=9.5.0" >&2
fi

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "=== HydraGNN Predictive GFM 2024 — Inference ==="
echo "  SIF        : $HG_SIF"
echo "  Overlay    : $HG_OVERLAY"
echo "  Checkpoint : $HG_CHECKPOINT"
echo "  Config     : $HG_CONFIG"
echo "  Output     : $HG_OUTPUT_DIR"
echo ""

HG_INFER_REPO="${HG_INFER_REPO:-${HG_OUTPUT_DIR}/HydraGNN-infer}"

apptainer exec --rocm \
    --overlay "${HG_OVERLAY}:ro" \
    --bind "$(dirname "$HG_CHECKPOINT"):$(dirname "$HG_CHECKPOINT"):ro" \
    --bind "$(dirname "$HG_CONFIG"):$(dirname "$HG_CONFIG"):ro" \
    --bind "${HG_OUTPUT_DIR}:${HG_OUTPUT_DIR}" \
    --bind "$SCRIPT_DIR":/examples:ro \
    --env HG_CHECKPOINT="$HG_CHECKPOINT" \
    --env HG_CONFIG="$HG_CONFIG" \
    --env HG_OUTPUT_DIR="$HG_OUTPUT_DIR" \
    --env HG_INFER_REPO="$HG_INFER_REPO" \
    --env PYTHONPATH="/opt/hydragnn-pkgs" \
    "$HG_SIF" \
    bash -c "source /opt/venv/bin/activate && bash /examples/run_inference.sh"
