#!/usr/bin/env bash
# docker_run.sh — Launch SwinUNETR containers via Docker Compose (silogen/ai-samples)
#
# Wraps the Docker Compose setup from silogen/ai-samples.
# Pass "train" or "infer" to select the appropriate image.
#
# Usage:
#   bash docker_run.sh train    # ROCm 6.4, training image
#   bash docker_run.sh infer    # ROCm 7.0, inference/optimized image
#   bash docker_run.sh shell    # Interactive shell in training image
#
# OPTIONAL EDITS:
#   AI_SAMPLES_DIR — where to clone silogen/ai-samples (default: alongside this script)
#   DATA_DIR       — host path for MONAI data cache (default: /tmp/swinunetr-data)
#   WORKSPACE_DIR  — host path for checkpoints and results (default: alongside this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-shell}"

# --- Configuration ---
AI_SAMPLES_DIR="${AI_SAMPLES_DIR:-$SCRIPT_DIR/ai-samples}"
DATA_DIR="${DATA_DIR:-/tmp/swinunetr-data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"

# Training image (ROCm 6.4 / PyTorch 2.6)
TRAIN_IMAGE="rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0"
# Inference image (ROCm 7.0 / PyTorch 2.6) — needed for torch.compile max-autotune
INFER_IMAGE="rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0"

# --- Check ROCm device ---
if [[ ! -e /dev/kfd ]]; then
    echo "WARNING: /dev/kfd not found — ROCm kernel driver may not be loaded."
fi

# --- Clone silogen/ai-samples if needed ---
if [[ ! -d "$AI_SAMPLES_DIR" ]]; then
    echo "Cloning silogen/ai-samples into $AI_SAMPLES_DIR ..."
    git clone --depth=1 https://github.com/silogen/ai-samples.git "$AI_SAMPLES_DIR"
fi

RECIPE_DIR="$AI_SAMPLES_DIR/life-science/medical-imaging/swinunetr"

# --- Detect GPU access method ---
if docker info 2>/dev/null | grep -qi "amd"; then
    RUNTIME_ARGS=(--runtime=amd -e AMD_VISIBLE_DEVICES=all)
    echo "Using AMD Container Toolkit runtime."
else
    RENDER_DEVICES=()
    for node in /dev/dri/renderD*; do
        [[ -e "$node" ]] && RENDER_DEVICES+=(--device="$node")
    done
    RUNTIME_ARGS=(--device=/dev/kfd "${RENDER_DEVICES[@]}" --group-add video)
    echo "Using device passthrough for GPU access."
fi

mkdir -p "$DATA_DIR" "$WORKSPACE_DIR/checkpoints" "$WORKSPACE_DIR/results"

# --- Try Docker Compose first (uses silogen's compose file) ---
if [[ -f "$RECIPE_DIR/docker-compose.yml" ]] || [[ -f "$RECIPE_DIR/compose.yml" ]]; then
    echo "Using Docker Compose from $RECIPE_DIR ..."
    cd "$RECIPE_DIR"
    DATA_DIR="$DATA_DIR" WORKSPACE_DIR="$WORKSPACE_DIR" \
        docker compose up --build -d
    echo ""
    echo "Container running. To attach:"
    echo "  docker compose -f $RECIPE_DIR/docker-compose.yml exec swinunetr bash"
    echo ""
    echo "Then run examples:"
    echo "  bash /examples/run_train.sh"
    echo "  bash /examples/run_inference.sh"
    exit 0
fi

# --- Fallback: plain docker run ---
case "$MODE" in
    train|shell)
        IMAGE="$TRAIN_IMAGE"
        CONTAINER_NAME="swinunetr-train"
        ;;
    infer)
        IMAGE="$INFER_IMAGE"
        CONTAINER_NAME="swinunetr-infer"
        ;;
    *)
        echo "Usage: $0 [train|infer|shell]"
        exit 1
        ;;
esac

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    echo "  To attach : docker exec -it $CONTAINER_NAME bash"
    echo "  To remove  : docker rm -f $CONTAINER_NAME"
    exit 0
fi

echo ""
echo "Launching $IMAGE ($MODE) ..."

docker run -d \
    "${RUNTIME_ARGS[@]}" \
    --name "$CONTAINER_NAME" \
    --network host \
    --shm-size=32g \
    -v "$DATA_DIR":/data \
    -v "$WORKSPACE_DIR":/workspace \
    -v "$RECIPE_DIR":/recipe:ro \
    -v "$SCRIPT_DIR":/examples:ro \
    "$IMAGE" tail -f /dev/null

echo "Container '$CONTAINER_NAME' started."
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""

if [[ "$MODE" == "train" ]]; then
    echo "Install dependencies inside the container:"
    echo "  pip install monai[all] nibabel"
    echo "Then: bash /examples/run_train.sh"
elif [[ "$MODE" == "infer" ]]; then
    echo "Install dependencies inside the container:"
    echo "  pip install monai[all] nibabel"
    echo "Then: bash /examples/run_inference.sh"
fi
