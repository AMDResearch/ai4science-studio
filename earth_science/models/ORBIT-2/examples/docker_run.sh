#!/usr/bin/env bash
# docker_run.sh — Launch an ORBIT-2 inference container on AMD Instinct
#
# Suitable for single-node inference and visualization.
# Multi-GPU distributed training is HPC-scale (see recipes/train/README.md).
#
# OPTIONAL EDITS:
#   AMD_VISIBLE_DEVICES — GPU indices to expose (default: all)
#   ORBIT2_ROOT         — path to a pre-cloned XiaoWang-Github/ORBIT-2 checkout;
#                         if unset the container starts without it (clone inside)
#   WORKSPACE_DIR       — host path mounted as /workspace (default: this examples/ dir)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms)
#   - AMD Container Toolkit (provides --runtime=amd); falls back to device passthrough

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-shell}"   # inference | shell

# --- Configuration ---
IMAGE="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1"
CONTAINER_NAME="orbit2"
AMD_VISIBLE_DEVICES="${AMD_VISIBLE_DEVICES:-all}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"
ORBIT2_ROOT="${ORBIT2_ROOT:-}"

# --- Validate mode ---
case "$MODE" in
    inference|shell) ;;
    *)
        echo "Usage: $0 [inference|shell]"
        echo "  inference — start a container and run run_visualize.py"
        echo "  shell     — start a container with an interactive bash shell"
        exit 1
        ;;
esac

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

VOLUME_ARGS=(-v "$WORKSPACE_DIR":/workspace -v "$SCRIPT_DIR":/examples:ro)
[[ -n "$ORBIT2_ROOT" ]] && VOLUME_ARGS+=(-v "$ORBIT2_ROOT":/orbit2)

echo ""
echo "Starting container: $CONTAINER_NAME  (mode: $MODE)"
echo "  Image     : $IMAGE"
echo "  Workspace : $WORKSPACE_DIR → /workspace"
[[ -n "$ORBIT2_ROOT" ]] && echo "  ORBIT-2   : $ORBIT2_ROOT → /orbit2"
echo ""

if [[ "$MODE" == "shell" ]]; then
    docker run -it --rm \
        "${RUNTIME_ARGS[@]}" \
        --name "$CONTAINER_NAME" \
        --network host \
        --shm-size=16g \
        "${VOLUME_ARGS[@]}" \
        "$IMAGE" bash
else
    # --- inference mode: validate ORBIT2_ROOT then run ---
    if [[ -z "$ORBIT2_ROOT" ]]; then
        echo "ERROR: ORBIT2_ROOT must be set for inference mode."
        echo "  Clone the repo first:"
        echo "    git clone https://github.com/XiaoWang-Github/ORBIT-2.git /path/to/orbit2"
        echo "  Then:"
        echo "    ORBIT2_ROOT=/path/to/orbit2 ./docker_run.sh inference"
        exit 1
    fi
    if [[ -z "${ORBIT2_CONFIG:-}" ]]; then
        echo "ERROR: ORBIT2_CONFIG must be set (path to a YAML config, e.g. interm_8m_ft.yaml)."
        exit 1
    fi
    docker run -it --rm \
        "${RUNTIME_ARGS[@]}" \
        --name "$CONTAINER_NAME" \
        --network host \
        --shm-size=16g \
        "${VOLUME_ARGS[@]}" \
        -e ORBIT2_ROOT=/orbit2 \
        "$IMAGE" \
        python /examples/run_visualize.py \
            --orbit2-root /orbit2 \
            "${ORBIT2_CONFIG}" \
            ${ORBIT2_CHECKPOINT:+--checkpoint "$ORBIT2_CHECKPOINT"}
fi
