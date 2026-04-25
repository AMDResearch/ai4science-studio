#!/usr/bin/env bash
# docker_run.sh — Launch a REINVENT4 container on AMD Instinct
#
# OPTIONAL EDITS:
#   WORKSPACE_DIR — host directory mounted as /workspace (default: this examples/ dir)
#   RENDER_ID     — specific /dev/dri/renderD node index (default: auto-detected)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms)
#   Note: REINVENT4 was validated without AMD Container Toolkit; uses device passthrough.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
IMAGE="rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.10_pytorch_release_2.2.1"
CONTAINER_NAME="reinvent4"
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

# --- Collect render devices ---
RENDER_DEVICES=()
for node in /dev/dri/renderD*; do
    [[ -e "$node" ]] && RENDER_DEVICES+=(--device="$node")
done

if [[ ${#RENDER_DEVICES[@]} -eq 0 ]]; then
    echo "WARNING: No /dev/dri/renderD* nodes found."
fi

echo ""
echo "Starting container: $CONTAINER_NAME"
echo "  Image     : $IMAGE"
echo "  Workspace : $WORKSPACE_DIR → /workspace"
echo ""

docker run -d \
    --device=/dev/kfd \
    "${RENDER_DEVICES[@]}" \
    --group-add video \
    --name "$CONTAINER_NAME" \
    --network host \
    --shm-size=16g \
    -v "$WORKSPACE_DIR":/workspace \
    "$IMAGE" \
    tail -f /dev/null

echo "Container started. Cloning and installing REINVENT4 ..."
echo ""

docker exec "$CONTAINER_NAME" bash -c '
    set -euo pipefail
    cd /workspace

    if [[ ! -d REINVENT4 ]]; then
        echo "Cloning MolecularAI/REINVENT4 ..."
        git clone https://github.com/MolecularAI/REINVENT4.git
    fi

    cd REINVENT4

    if ! command -v reinvent &>/dev/null; then
        echo "Stripping PyTorch deps from pyproject.toml (already in ROCm image) ..."
        sed -i "/^torch/d; /^torchvision/d" pyproject.toml

        echo "Installing REINVENT4 ..."
        python install.py
    else
        echo "REINVENT4 already installed — skipping."
    fi

    echo ""
    echo "REINVENT4 is ready. Try: reinvent --help"
'

echo ""
echo "Done. Attach to the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "Then run examples (from inside the container):"
echo "  bash /workspace/run_tl.sh"
