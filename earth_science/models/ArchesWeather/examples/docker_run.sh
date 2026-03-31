#!/usr/bin/env bash
# docker_run.sh — Build the ArchesWeather image and launch an interactive container
#
# OPTIONAL EDITS:
#   AI_SAMPLES_DIR — where to clone silogen/ai-samples (default: alongside this script)
#   DATA_DIR       — host path to mount as /data for ERA5 data (default: /tmp/archesweather-data)
#   WORKSPACE_DIR  — host path to mount as /workspace for outputs (default: alongside this script)
#
# Prerequisites:
#   - ROCm kernel-mode driver (amdgpu-dkms)
#   - ~735 GB disk at DATA_DIR for full ERA5 training data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
AI_SAMPLES_DIR="${AI_SAMPLES_DIR:-$SCRIPT_DIR/ai-samples}"
DATA_DIR="${DATA_DIR:-/tmp/archesweather-data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"
IMAGE_NAME="pytorch_training_geoarches:latest"
CONTAINER_NAME="archesweather"

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

# --- Clone silogen/ai-samples if needed ---
if [[ ! -d "$AI_SAMPLES_DIR" ]]; then
    echo "Cloning silogen/ai-samples into $AI_SAMPLES_DIR ..."
    git clone --depth=1 https://github.com/silogen/ai-samples.git "$AI_SAMPLES_DIR"
fi

RECIPE_DIR="$AI_SAMPLES_DIR/ai4sciences/geoarches-training"

# --- Build image if not present ---
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building $IMAGE_NAME (this may take ~10 min) ..."
    docker build -t "$IMAGE_NAME" "$RECIPE_DIR"
else
    echo "Image $IMAGE_NAME already exists — skipping build."
fi

# --- Collect render devices ---
RENDER_DEVICES=()
for node in /dev/dri/renderD*; do
    [[ -e "$node" ]] && RENDER_DEVICES+=(--device="$node")
done

mkdir -p "$DATA_DIR" "$WORKSPACE_DIR/checkpoints" "$WORKSPACE_DIR/results"

echo ""
echo "Launching $IMAGE_NAME ..."
echo "  Data      : $DATA_DIR → /data  (ERA5, ~735 GB for full training)"
echo "  Workspace : $WORKSPACE_DIR → /workspace"
echo "  Examples  : $SCRIPT_DIR → /examples (read-only)"
echo ""

docker run -it --rm \
    --device=/dev/kfd \
    "${RENDER_DEVICES[@]}" \
    --group-add video \
    --shm-size=16g \
    --network host \
    -v "$DATA_DIR":/data \
    -v "$WORKSPACE_DIR":/workspace \
    -v "$SCRIPT_DIR":/examples:ro \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" bash
