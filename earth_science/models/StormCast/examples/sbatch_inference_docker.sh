#!/usr/bin/env bash
# StormCast deterministic inference on AMD Instinct via SLURM + Docker.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export SC_START=2025-01-01T06 SC_STEPS=6
#   sbatch sbatch_inference_docker.sh
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   1. Docker must be available on compute nodes.
#   2. Adjust #SBATCH directives below for your site's partition and account.
#   3. Internet access from compute nodes is required to pull the Docker image,
#      download StormCast weights from HuggingFace, and fetch NOAA HRRR/GFS
#      analysis data at runtime.
#
# ── Key environment variables ─────────────────────────────────────────────────
#   SC_IMAGE      Docker image (default: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0)
#   SC_START      Forecast start time ISO-8601, e.g. 2025-01-01T06 (default: 2025-01-01T06)
#   SC_STEPS      Number of 1-h inference steps (default: 6)
#   SC_OUTPUT     Output zarr path (default: outputs/pred-<start>.zarr next to script)
#
# ── GPU / ROCm compatibility ─────────────────────────────────────────────────
#   Default image covers MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950).
#   For older hardware (MI100/gfx908): override SC_IMAGE with a rocm6.x image.
#
# See: ../recipes/inference/README.md

#SBATCH --job-name=stormcast-docker
#SBATCH --partition=amd-arad
##SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=stormcast-docker-%j.out
#SBATCH --error=stormcast-docker-%j.out

set -euo pipefail

# SLURM copies the script to its spool dir before execution, so BASH_SOURCE[0]
# would resolve to the spool path. Use scontrol to recover the original path.
SCRIPT_DIR=$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")" && pwd)

# ---------------------------------------------------------------------------
# Configuration (override by exporting before sbatch)
# ---------------------------------------------------------------------------
SC_IMAGE="${SC_IMAGE:-rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
SC_START="${SC_START:-2025-01-01T06}"
SC_STEPS="${SC_STEPS:-6}"
SC_OUTPUT="${SC_OUTPUT:-}"

CONTAINER_NAME="stormcast-${SLURM_JOB_ID:-$$}"

# ---------------------------------------------------------------------------
# Resolve output path and wipe any previous run's zarr
# (ZarrBackend refuses to overwrite an existing store)
# ---------------------------------------------------------------------------
if [[ -z "${SC_OUTPUT}" ]]; then
    SAFE_START="${SC_START//[: T]/-}"
    SC_OUTPUT="${SCRIPT_DIR}/outputs/pred-${SAFE_START}.zarr"
fi
if [[ -e "${SC_OUTPUT}" ]]; then
    echo "  Removing previous output: ${SC_OUTPUT}"
    rm -rf "${SC_OUTPUT}"
fi

echo "=== StormCast deterministic inference (Docker) ==="
echo "  Image       : ${SC_IMAGE}"
echo "  Start       : ${SC_START}"
echo "  Steps       : ${SC_STEPS}"
echo "  Output zarr : ${SC_OUTPUT}"
echo "  Container   : ${CONTAINER_NAME}"
echo "  Job         : ${SLURM_JOB_ID:-local}"
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
    -v "${SCRIPT_DIR}":/workspace \
    "$SC_IMAGE" \
    bash -c '
set -euo pipefail
source /opt/venv/bin/activate
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

# Install deps (earth2studio + cartopy)
if ! python3 -c "import earth2studio, cartopy" 2>/dev/null; then
    echo "--- Installing earth2studio[stormcast] and cartopy ---"
    pip install -q --no-cache-dir "earth2studio[stormcast]" cartopy 2>&1 | tail -5

    # In Docker (unlike Apptainer SIF), the container filesystem is writable.
    # pip should keep the existing ROCm torch if it satisfies version constraints.
    # Do NOT strip torch here — it would remove the only copy.
    # Instead, verify the installed torch is still ROCm.
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    if [[ "$TORCH_VER" != *rocm* ]]; then
        echo "ERROR: Expected ROCm torch, got $TORCH_VER" >&2; exit 1
    fi
    echo "  ROCm torch verified: $TORCH_VER"
fi

python /workspace/run_inference.py \
    --start "'"${SC_START}"'" \
    --steps "'"${SC_STEPS}"'" \
    --output "'"${SC_OUTPUT}"'"
'

echo "=== Done ==="
echo "  Output: ${SC_OUTPUT}"
