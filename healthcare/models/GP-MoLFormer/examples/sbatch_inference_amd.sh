#!/usr/bin/env bash
# GP-MoLFormer molecule generation on AMD Instinct via SLURM + Apptainer.
#
# RESEARCH/ENGINEERING USE ONLY. This script generates novel molecules for
# drug-discovery research purposes only. Outputs are not validated for clinical
# or therapeutic use and must not be used for patient treatment decisions.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export GPMOL_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
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
#   GPMOL_SIF        Path to Apptainer SIF image
#                    (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif)
#   GPMOL_WORK_DIR   Host directory for repo clone, weights, and output
#                    (default: ${AI4S_SHARED_DIR:-/your/shared/dir}/models/GP-MoLFormer)
#   SCAFFOLD         SMILES fragment for constrained generation (unset = unconditional)
#   NUM_BATCHES      Number of batches of 1000 molecules (default: 1 = 1000 molecules)
#   OUTPUT_FILE      Output CSV path inside container (default: /workspace/generated.csv)
#                    Maps to GPMOL_WORK_DIR/outputs/generated.csv on the host.
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   Tested images (pick one based on your host ROCm driver):
#     rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1   (py3.10)
#     rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0 (py3.12)
#   If torch.cuda.is_available() returns False inside the container, your host
#   amdgpu driver is likely too new for the SIF's ROCm userspace. Try the newer
#   SIF or set HSA_OVERRIDE_GFX_VERSION=9.4.2 as a workaround.
#   Covers: MI250X (gfx90a), MI300A/MI300X (gfx942), MI350X (gfx950)
#   GP-MoLFormer is CPU-heavy (GP inference) — 1 GPU is sufficient.
#   For older hardware (MI100/gfx908): use a rocm6.x image.

# Adjust #SBATCH directives to match your site's partition, account, and runtime.
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPMOL_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/GP-MoLFormer"
GPMOL_SIF="${GPMOL_SIF:-${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
GPMOL_WORK_DIR="${GPMOL_WORK_DIR:-$GPMOL_BASE}"
SCAFFOLD="${SCAFFOLD:-}"
NUM_BATCHES="${NUM_BATCHES:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-/workspace/generated.csv}"

# Per-job package dir for pip install --target (SIF is read-only)
GPMOL_PKGDIR="${TMPDIR:-/tmp}/gpmol-pkgs-${SLURM_JOB_ID:-$$}"
mkdir -p "$GPMOL_PKGDIR"

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
    --bind "$SCRIPT_DIR":/scripts \
    --bind "${GPMOL_PKGDIR}:/opt/gpmol-pkgs" \
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

echo "--- Installing dependencies (from environment.yml, via pip) ---"
pip install -q --no-cache-dir --target /opt/gpmol-pkgs \
    "accelerate==0.26.1" "datasets==2.20.0" "networkx>=3.1" \
    "numpy<2" "pandas>=2.2" "peft==0.10.0" "scikit-learn>=1.5" \
    "transformers>=4.36,<4.41" 2>&1 | tail -5
pip install -q --no-cache-dir --target /opt/gpmol-pkgs \
    rdkit-pypi 2>&1 | tail -3 \
    || echo "  rdkit unavailable for $(python3 --version) — validity check will be skipped"

echo "--- Stripping torch / nvidia / triton (keep ROCm torch from SIF) ---"
for pkg in torch torchvision torchaudio nvidia triton; do
    rm -rf "/opt/gpmol-pkgs/${pkg}" "/opt/gpmol-pkgs/${pkg}"-*.dist-info 2>/dev/null || true
done

export PYTHONPATH="/opt/gpmol-pkgs:${PYTHONPATH:-}"

GPU_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$GPU_OK" != "True" ]]; then
    echo "WARNING: torch.cuda.is_available() = False — falling back to CPU." >&2
    echo "  This makes generation ~10-20x slower. To fix:" >&2
    echo "  1. Use a newer ROCm SIF (e.g. rocm7.2.2), or" >&2
    echo "  2. Set HSA_OVERRIDE_GFX_VERSION=9.4.2" >&2
fi

echo "--- Running generation ---"
bash /scripts/run_generation.sh
'

echo ""
echo "=== Done ==="
echo "  Host output: ${GPMOL_WORK_DIR}/$(basename "$OUTPUT_FILE")"
