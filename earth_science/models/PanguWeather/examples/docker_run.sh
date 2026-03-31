#!/usr/bin/env bash
# docker_run.sh — Build the JAX weather container and launch an interactive session
#
# USER EDITS REQUIRED:
#   1. Copy env_file.template to env_file and set your CDSAPI_KEY
#      (free account at https://cds.climate.copernicus.eu/)
#
# OPTIONAL EDITS:
#   CACHE_DIR  — where to cache downloaded ERA5 data (default: /tmp/earthkit-cache)
#   AI_SAMPLES_DIR — where to clone silogen/ai-samples (default: alongside this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
CACHE_DIR="${CACHE_DIR:-/tmp/earthkit-cache}"
AI_SAMPLES_DIR="${AI_SAMPLES_DIR:-$SCRIPT_DIR/ai-samples}"
IMAGE_NAME="jaxweather:latest"

# --- Check env_file ---
ENV_FILE="$SCRIPT_DIR/env_file"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found."
    echo ""
    echo "  cp $SCRIPT_DIR/env_file.template $SCRIPT_DIR/env_file"
    echo "  # Then set CDSAPI_KEY in the file"
    echo "  # Get a free key at: https://cds.climate.copernicus.eu/"
    exit 1
fi

if grep -q "YOUR_CDS_API_KEY_HERE" "$ENV_FILE"; then
    echo "ERROR: CDSAPI_KEY is still set to the placeholder value in env_file."
    echo "  Edit $ENV_FILE and set your real API key."
    exit 1
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

RECIPE_DIR="$AI_SAMPLES_DIR/ai4sciences/ai-weather-forecasting"

# --- Build JAX image if not present ---
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building $IMAGE_NAME ..."
    docker build -t "$IMAGE_NAME" -f "$RECIPE_DIR/jax.dockerfile" "$RECIPE_DIR"
else
    echo "Image $IMAGE_NAME already exists — skipping build. (Remove it to rebuild.)"
fi

# --- Collect /dev/dri render nodes ---
RENDER_DEVICES=()
for node in /dev/dri/renderD*; do
    [[ -e "$node" ]] && RENDER_DEVICES+=(--device="$node")
done

mkdir -p "$CACHE_DIR"

echo ""
echo "Launching $IMAGE_NAME ..."
echo "  Data cache : $CACHE_DIR"
echo "  Examples   : $SCRIPT_DIR (mounted at /examples)"
echo "  Recipe dir : $RECIPE_DIR (mounted at /recipe)"
echo ""

docker run -it --rm \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_RAWIO \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/mem \
    "${RENDER_DEVICES[@]}" \
    --group-add video \
    --network host \
    --env-file "$ENV_FILE" \
    -e EARTHKIT_DATA_USER_CACHE_DIRECTORY=/cache \
    -v "$CACHE_DIR":/cache \
    -v "$SCRIPT_DIR":/examples:ro \
    -v "$RECIPE_DIR":/recipe:ro \
    -v "$MODEL_DIR/examples/predictions":/predictions \
    --name jaxrocm \
    "$IMAGE_NAME" bash
