#!/usr/bin/env bash
# docker_run.sh — Launch a SemlaFlow container on AMD Instinct
#
# OPTIONAL EDITS:
#   WORKSPACE_DIR — host directory mounted as /workspace (default: this examples/ dir)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms) — ROCm 6.4.1 or newer required
#   - AMD Container Toolkit (preferred) or device passthrough fallback
#
# IMPORTANT: ROCm 6.3.3 causes import errors with SemlaFlow — use 6.4.1+.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
IMAGE="rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0"
CONTAINER_NAME="semlaflow"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"

# --- Check for existing container ---
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    echo "  To attach : docker exec -it $CONTAINER_NAME bash"
    echo "  To remove  : docker rm -f $CONTAINER_NAME"
    exit 0
fi

# --- Check ROCm device ---
if [[ ! -e /dev/kfd ]]; then
    echo "WARNING: /dev/kfd not found — ROCm kernel driver may not be loaded."
fi

# --- Detect GPU access method ---
if docker info 2>/dev/null | grep -qi "amd"; then
    RUNTIME_ARGS=(--runtime=amd -e AMD_VISIBLE_DEVICES=all)
    echo "Using AMD Container Toolkit runtime."
else
    echo "AMD Container Toolkit not detected — using device passthrough."
    RENDER_DEVICES=()
    for node in /dev/dri/renderD*; do
        [[ -e "$node" ]] && RENDER_DEVICES+=(--device="$node")
    done
    RUNTIME_ARGS=(--device=/dev/kfd "${RENDER_DEVICES[@]}" --group-add video)
fi

echo ""
echo "Starting container: $CONTAINER_NAME"
echo "  Image     : $IMAGE  (ROCm 6.4.1 — 6.3.3 causes import errors)"
echo "  Workspace : $WORKSPACE_DIR → /workspace"
echo ""

docker run -d \
    "${RUNTIME_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size=256g \
    -v "$WORKSPACE_DIR":/workspace \
    "$IMAGE" \
    tail -f /dev/null

echo "Container started. Cloning and installing SemlaFlow ..."
echo ""

docker exec "$CONTAINER_NAME" bash -c '
    set -euo pipefail
    cd /workspace

    if [[ ! -d semla-flow ]]; then
        echo "Cloning rssrwn/semla-flow ..."
        git clone https://github.com/rssrwn/semla-flow.git
    fi

    cd semla-flow

    if ! python -c "import semlaflow" &>/dev/null 2>&1; then
        echo "Patching environment.yml: swapping pytorch-cuda → rocm-pytorch ..."
        sed -i "s/pytorch-cuda[^\"'\''<>]*/rocm-pytorch/g" environment.yml

        echo "Installing conda environment ..."
        conda env update --name base --file environment.yml --prune
    else
        echo "SemlaFlow already installed — skipping."
    fi

    echo ""
    echo "SemlaFlow is ready."
'

echo ""
echo "Done. Attach to the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "Then run:"
echo "  bash /workspace/run_inference.sh"
