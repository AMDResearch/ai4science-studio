#!/usr/bin/env bash
# GP-MoLFormer molecule generation on AMD Instinct via SLURM + Docker.
#
# RESEARCH/ENGINEERING USE ONLY. This script generates novel molecules for
# drug-discovery research purposes only. Outputs are not validated for clinical
# or therapeutic use and must not be used for patient treatment decisions.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   sbatch sbatch_inference_docker.sh                            # unconditional
#   SCAFFOLD="c1ccccc1" sbatch sbatch_inference_docker.sh        # scaffold mode
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   1. Docker must be available on compute nodes.
#   2. Adjust #SBATCH directives below for your site's partition and account.
#   3. Internet access from compute nodes is required on first run to pull the
#      Docker image, clone IBM/gp-molformer, and download HuggingFace weights.
#      On subsequent runs the clone and weights are reused from GPMOL_WORK_DIR.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   GPMOL_IMAGE      Docker image (default: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0)
#   GPMOL_WORK_DIR   Host directory for repo clone, weights, and output
#                    (default: same directory as this script)
#   SCAFFOLD         SMILES fragment for constrained generation (unset = unconditional)
#   NUM_BATCHES      Number of batches of 1000 molecules (default: 1 = 1000 molecules)
#   OUTPUT_FILE      Output CSV path inside container (default: /workspace/generated.csv)
#                    Maps to GPMOL_WORK_DIR/generated.csv on the host.
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   Default image covers MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950).
#   GP-MoLFormer is CPU-heavy (GP inference) — 1 GPU is sufficient.
#   For older hardware (MI100/gfx908): override GPMOL_IMAGE with a rocm6.x image.

#SBATCH --job-name=gpmolformer-docker
#SBATCH --partition=amd-arad
##SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=gpmolformer-docker-%j.out
#SBATCH --error=gpmolformer-docker-%j.out

set -euo pipefail

# SLURM copies the script to its spool dir before execution, so BASH_SOURCE[0]
# would resolve to the spool path. Use scontrol to recover the original path.
SCRIPT_DIR=$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")" && pwd)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPMOL_IMAGE="${GPMOL_IMAGE:-rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
GPMOL_WORK_DIR="${GPMOL_WORK_DIR:-$SCRIPT_DIR}"
SCAFFOLD="${SCAFFOLD:-}"
NUM_BATCHES="${NUM_BATCHES:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-/workspace/generated.csv}"

CONTAINER_NAME="gpmolformer-${SLURM_JOB_ID:-$$}"

echo "=== GP-MoLFormer molecule generation (Docker) ==="
echo "  Image        : $GPMOL_IMAGE"
echo "  Work dir     : $GPMOL_WORK_DIR"
echo "  Scaffold     : ${SCAFFOLD:-<none — unconditional>}"
echo "  Num batches  : $NUM_BATCHES (${NUM_BATCHES}000 molecules)"
echo "  Output (ctr) : $OUTPUT_FILE"
echo "  Container    : $CONTAINER_NAME"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo ""

# ---------------------------------------------------------------------------
# Detect GPU access method
# ---------------------------------------------------------------------------
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi "amd"; then
    GPU_ARGS=(--runtime=amd -e AMD_VISIBLE_DEVICES=all)
    echo "  GPU method : AMD Container Toolkit"
else
    GPU_ARGS=(--device=/dev/kfd)
    for dev in /dev/dri/renderD*; do
        [[ -e "$dev" ]] && GPU_ARGS+=(--device="$dev")
    done
    GPU_ARGS+=(--group-add video)
    echo "  GPU method : device passthrough"
fi
echo ""

# ---------------------------------------------------------------------------
# Cleanup trap — remove container on exit
# ---------------------------------------------------------------------------
cleanup() {
    echo "--- Cleaning up container $CONTAINER_NAME ---"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Run inside Docker container
# ---------------------------------------------------------------------------
docker run --rm \
    "${GPU_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --network host \
    --shm-size=16g \
    -v "$GPMOL_WORK_DIR":/workspace \
    -v "$SCRIPT_DIR":/scripts:ro \
    -e SCAFFOLD="$SCAFFOLD" \
    -e NUM_BATCHES="$NUM_BATCHES" \
    -e OUTPUT_FILE="$OUTPUT_FILE" \
    "$GPMOL_IMAGE" \
    bash -c '
set -euo pipefail
source /opt/venv/bin/activate
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
cd /workspace

# Clone repo on first run
if [[ ! -d gp-molformer ]]; then
    echo "--- Cloning IBM/gp-molformer ---"
    git clone https://github.com/IBM/gp-molformer.git
fi

cd gp-molformer

echo "--- Installing dependencies (no requirements.txt — key deps from environment.yml) ---"
# Version pins from environment.yml target py3.10; tokenizers 0.13.x has no cp312
# wheel and requires Rust to build from source. Use transformers>=4.36 which pulls
# tokenizers>=0.15 — the first release with cp312 binary wheels.
pip install -q --no-cache-dir --prefer-binary \
    "accelerate>=0.26" \
    "datasets>=2.20" \
    networkx \
    pandas \
    "peft>=0.10" \
    scikit-learn \
    "transformers>=4.36,<4.41" 2>&1 | tail -5
pip install -q --no-cache-dir --prefer-binary rdkit-pypi 2>&1 | tail -3 \
    || echo "  rdkit-pypi unavailable — validity check in run_generation.sh will be skipped"

echo "--- Running generation ---"
bash /scripts/run_generation.sh
'

echo ""
echo "=== Done ==="
echo "  Host output: ${GPMOL_WORK_DIR}/$(basename "$OUTPUT_FILE")"
