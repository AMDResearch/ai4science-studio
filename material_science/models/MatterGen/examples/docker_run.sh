#!/usr/bin/env bash
# docker_run.sh — Launch a MatterGen container on AMD Instinct
#
# OPTIONAL EDITS:
#   AMD_VISIBLE_DEVICES — GPU indices to expose (default: all)
#   WORKSPACE_DIR       — host directory mounted as /workspace (default: this examples/ dir)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms)
#   - AMD Container Toolkit (provides --runtime=amd); falls back to device passthrough

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
IMAGE="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1"
CONTAINER_NAME="mattergen"
AMD_VISIBLE_DEVICES="${AMD_VISIBLE_DEVICES:-all}"
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

echo ""
echo "Starting container: $CONTAINER_NAME"
echo "  Image     : $IMAGE"
echo "  Workspace : $WORKSPACE_DIR → /workspace"
echo ""

docker run -d \
    "${RUNTIME_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --network host \
    --shm-size=16g \
    -v "$WORKSPACE_DIR":/workspace \
    "$IMAGE" \
    tail -f /dev/null

echo "Container started. Cloning and installing MatterGen ..."
echo "(This installs ROCm-compatible pytorch_scatter and pytorch_sparse — takes ~5 min)"
echo ""

docker exec "$CONTAINER_NAME" bash -c '
    set -euo pipefail
    cd /workspace

    if [[ ! -d mattergen ]]; then
        echo "Cloning microsoft/mattergen ..."
        git clone https://github.com/microsoft/mattergen.git
    fi

    cd mattergen

    if ! python -c "import mattergen" &>/dev/null 2>&1; then
        echo "Running setup (installs ROCm forks of pytorch_scatter / pytorch_sparse) ..."
        bash src/setup.bash
    else
        echo "MatterGen already installed — skipping setup."
    fi

    echo ""
    echo "MatterGen is ready."
'

echo ""
echo "Done. Attach to the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "Then run examples (from inside the container):"
echo "  bash /workspace/run_inference.sh"
echo "  bash /workspace/run_train.sh"
