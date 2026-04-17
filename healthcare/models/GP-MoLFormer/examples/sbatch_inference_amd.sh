#!/usr/bin/env bash
# GP-MoLFormer molecule generation on AMD Instinct via SLURM + Apptainer.
#
# RESEARCH/ENGINEERING USE ONLY. This script generates novel molecules for
# drug-discovery research purposes only. Outputs are not validated for clinical
# or therapeutic use and must not be used for patient treatment decisions.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export GPMOL_SIF=/path/to/rocm_pytorch.sif
#   sbatch sbatch_inference_amd.sh                            # unconditional
#   SCAFFOLD="c1ccccc1" sbatch sbatch_inference_amd.sh        # scaffold mode
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   1. Build an Apptainer SIF from the recommended ROCm PyTorch image:
#        apptainer pull docker://rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
#      Set GPMOL_SIF to the resulting .sif path.
#   2. Adjust #SBATCH directives below for your site's partition and account.
#   3. Internet access from compute nodes is required on first run to clone
#      IBM/gp-molformer and download HuggingFace model weights.
#      On subsequent runs the clone and weights are reused from GPMOL_WORK_DIR.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   GPMOL_SIF        Path to Apptainer SIF image (required)
#   GPMOL_WORK_DIR   Host directory for repo clone, weights, and output
#                    (default: same directory as this script)
#   SCAFFOLD         SMILES fragment for constrained generation (unset = unconditional)
#   NUM_BATCHES      Number of batches of 1000 molecules (default: 1 = 1000 molecules)
#   OUTPUT_FILE      Output CSV path inside container (default: /workspace/generated.csv)
#                    Maps to GPMOL_WORK_DIR/generated.csv on the host.
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   Tested image: rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
#   Covers: MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950)
#   GP-MoLFormer is CPU-heavy (GP inference) — 1 GPU is sufficient.
#   For older hardware (MI100/gfx908): use a rocm6.x image.

#SBATCH --job-name=gpmolformer-infer
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=gpmolformer-infer-%j.out
#SBATCH --error=gpmolformer-infer-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
if [[ -z "${GPMOL_SIF:-}" ]]; then
    echo "error: set GPMOL_SIF to your Apptainer SIF path before submitting" >&2
    echo "  Recommended image: rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1" >&2
    exit 1
fi

GPMOL_WORK_DIR="${GPMOL_WORK_DIR:-$SCRIPT_DIR}"
SCAFFOLD="${SCAFFOLD:-}"
NUM_BATCHES="${NUM_BATCHES:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-/workspace/generated.csv}"

echo "=== GP-MoLFormer molecule generation ==="
echo "  SIF          : $GPMOL_SIF"
echo "  Work dir     : $GPMOL_WORK_DIR"
echo "  Scaffold     : ${SCAFFOLD:-<none — unconditional>}"
echo "  Num batches  : $NUM_BATCHES (${NUM_BATCHES}000 molecules)"
echo "  Output (ctr) : $OUTPUT_FILE"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo ""

# ---------------------------------------------------------------------------
# Run inside container
# ---------------------------------------------------------------------------
apptainer exec \
    --rocm \
    --bind "$GPMOL_WORK_DIR":/workspace \
    --env SCAFFOLD="$SCAFFOLD" \
    --env NUM_BATCHES="$NUM_BATCHES" \
    --env OUTPUT_FILE="$OUTPUT_FILE" \
    "$GPMOL_SIF" \
    bash -c '
set -euo pipefail
cd /workspace

# Clone repo on first run
if [[ ! -d gp-molformer ]]; then
    echo "--- Cloning IBM/gp-molformer ---"
    git clone https://github.com/IBM/gp-molformer.git
fi

cd gp-molformer

echo "--- Installing dependencies ---"
pip install -q --no-cache-dir -r requirements.txt 2>&1 | tail -5

echo "--- Running generation ---"
bash /workspace/run_generation.sh
'

echo ""
echo "=== Done ==="
echo "  Host output: ${GPMOL_WORK_DIR}/$(basename "$OUTPUT_FILE")"
