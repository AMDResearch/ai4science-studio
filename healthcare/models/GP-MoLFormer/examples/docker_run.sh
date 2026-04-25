#!/usr/bin/env bash
# docker_run.sh — Launch the GP-MoLFormer container on AMD Instinct
#
# OPTIONAL EDITS:
#   AMD_VISIBLE_DEVICES — GPU indices to expose (default: all)
#   WORKSPACE_DIR       — host directory mounted as /workspace (default: this examples/ dir)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms) installed
#   - AMD Container Toolkit installed (provides --runtime=amd)
#     OR fall back to --device=/dev/kfd --device=/dev/dri if not available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
IMAGE="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1"
CONTAINER_NAME="gp-molformer"
AMD_VISIBLE_DEVICES="${AMD_VISIBLE_DEVICES:-all}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"

# --- Check for existing container ---
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    echo "  To attach: docker exec -it $CONTAINER_NAME bash"
    echo "  To remove: docker rm -f $CONTAINER_NAME"
    exit 0
fi

# --- Detect GPU access method ---
if docker info 2>/dev/null | grep -q "amd"; then
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
    -v "$WORKSPACE_DIR":/workspace \
    "$IMAGE" \
    tail -f /dev/null

echo "Container started. Setting up GP-MoLFormer ..."
echo ""

# Clone repo and apply patch inside the container
docker exec "$CONTAINER_NAME" bash -c '
    set -euo pipefail
    cd /workspace

    if [[ ! -d gp-molformer ]]; then
        echo "Cloning IBM/gp-molformer ..."
        git clone https://github.com/IBM/gp-molformer.git
    fi

    cd gp-molformer

    # Apply pairtune compatibility patch if present
    PATCH_SRC="/workspace/pairtune_training.patch"
    if [[ -f "$PATCH_SRC" ]] && ! git apply --check "$PATCH_SRC" 2>/dev/null; then
        echo "Patch already applied or not applicable — skipping."
    elif [[ -f "$PATCH_SRC" ]]; then
        echo "Applying pairtune_training.patch ..."
        git apply "$PATCH_SRC"
    else
        echo "NOTE: pairtune_training.patch not found in /workspace."
        echo "  Pair-tuning may fail with newer transformers versions."
        echo "  See: https://github.com/IBM/gp-molformer for the patch."
    fi

    echo "Installing dependencies ..."
    pip install -q -r requirements.txt

    echo ""
    echo "GP-MoLFormer is ready."
'

echo ""
echo "Done. Attach to the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "Then run examples:"
echo "  bash /workspace/run_generation.sh"
echo "  bash /workspace/run_pairtune.sh"
