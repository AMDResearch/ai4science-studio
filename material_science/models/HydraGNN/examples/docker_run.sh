#!/usr/bin/env bash
# docker_run.sh — Launch a HydraGNN container on AMD Instinct
#
# OPTIONAL EDITS:
#   AMD_VISIBLE_DEVICES — GPU indices to expose (default: all)
#   WORKSPACE_DIR       — host directory mounted as /workspace (default: this examples/ dir)
#   DATA_DIR            — host path for ADIOS datasets (default: /tmp/hydragnn-data)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms)
#   - AMD Container Toolkit (provides --runtime=amd); falls back to device passthrough
#
# Note: published pretraining used Frontier (OLCF) at very large scale.
# This script targets single-node AMD Instinct for inference and small-scale experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
IMAGE="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1"
CONTAINER_NAME="hydragnn"
AMD_VISIBLE_DEVICES="${AMD_VISIBLE_DEVICES:-all}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-/tmp/hydragnn-data}"

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
    RUNTIME_ARGS=(--runtime=amd -e AMD_VISIBLE_DEVICES="$AMD_VISIBLE_DEVICES")
    echo "Using AMD Container Toolkit runtime."
else
    echo "AMD Container Toolkit not detected — using device passthrough."
    RENDER_DEVICES=()
    for node in /dev/dri/renderD*; do
        [[ -e "$node" ]] && RENDER_DEVICES+=(--device="$node")
    done
    RUNTIME_ARGS=(--device=/dev/kfd "${RENDER_DEVICES[@]}" --group-add video)
fi

mkdir -p "$DATA_DIR" "$WORKSPACE_DIR"

echo ""
echo "Starting container: $CONTAINER_NAME"
echo "  Image     : $IMAGE"
echo "  Workspace : $WORKSPACE_DIR → /workspace"
echo "  Data      : $DATA_DIR → /data"
echo ""

docker run -d \
    "${RUNTIME_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --network host \
    --shm-size=16g \
    -v "$WORKSPACE_DIR":/workspace \
    -v "$DATA_DIR":/data \
    -v "$SCRIPT_DIR":/examples:ro \
    "$IMAGE" \
    tail -f /dev/null

echo "Container started. Installing HydraGNN ..."
echo "(Cloning ORNL/HydraGNN branch Predictive_GFM_2024 and installing dependencies)"
echo ""

docker exec "$CONTAINER_NAME" bash -c '
    set -euo pipefail
    cd /workspace

    if [[ ! -d HydraGNN ]]; then
        echo "Cloning ORNL/HydraGNN (branch Predictive_GFM_2024) ..."
        git clone --depth=1 --branch Predictive_GFM_2024 \
            https://github.com/ORNL/HydraGNN.git
    fi

    cd HydraGNN

    if ! python -c "import hydragnn" &>/dev/null 2>&1; then
        echo "Installing HydraGNN ..."
        pip install -e ".[dev]" --quiet
    else
        echo "HydraGNN already installed — skipping."
    fi

    echo ""
    echo "HydraGNN is ready."
'

echo ""
echo "Done. Attach to the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "Then run examples (from inside the container):"
echo "  bash /examples/run_inference.sh"
echo "  bash /examples/run_train.sh"
