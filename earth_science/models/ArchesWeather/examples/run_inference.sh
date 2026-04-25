#!/usr/bin/env bash
# run_inference.sh — Run ArchesWeather inference and evaluation
#
# Run INSIDE the container launched by docker_run.sh.
#
# OPTIONAL EDITS:
#   CHECKPOINT   — path to trained checkpoint (default: downloads from HF gcouairon/ArchesWeather)
#   MODEL_NAME   — archesweather-m-seed0 through seed3, or archesweathergen (default: archesweather-m-seed0)
#   DATA_PATH    — ERA5 dataset path inside container (default: /data/era5_240/full)
#   OUTPUT_PATH  — where to write predictions (default: /workspace/results/predictions)
#   YEAR         — test year (default: 2020)

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-archesweather-m-seed0}"
DATA_PATH="${DATA_PATH:-/data/era5_240/full}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/results/predictions}"
YEAR="${YEAR:-2020}"
CHECKPOINT="${CHECKPOINT:-}"

mkdir -p "$OUTPUT_PATH"

# --- Download checkpoint from HF if not provided ---
if [[ -z "$CHECKPOINT" ]]; then
    echo "No checkpoint specified — downloading $MODEL_NAME from HuggingFace ..."
    echo "  (gcouairon/ArchesWeather)"
    python - <<PYEOF
from huggingface_hub import snapshot_download
import os
local_dir = "/workspace/checkpoints"
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id="gcouairon/ArchesWeather", local_dir=local_dir)
print(f"Downloaded to {local_dir}")
PYEOF
    CHECKPOINT="/workspace/checkpoints/$MODEL_NAME"
fi

echo "=== ArchesWeather Inference ==="
echo "  Model      : $MODEL_NAME"
echo "  Checkpoint : $CHECKPOINT"
echo "  Data       : $DATA_PATH"
echo "  Test year  : $YEAR"
echo "  Output     : $OUTPUT_PATH"
echo ""

# --- Run inference ---
python -m geoarches.inference.encode_dataset \
    "++checkpoint_path=$CHECKPOINT" \
    "++dataloader.dataset.path=$DATA_PATH" \
    "++dataloader.dataset.start_year=$YEAR" \
    "++dataloader.dataset.end_year=$YEAR" \
    "++output_path=$OUTPUT_PATH"

echo ""
echo "Predictions written to: $OUTPUT_PATH"
echo ""

# --- Evaluate ---
echo "Running evaluation metrics (RMSE, CRPS, Brier Skill Score) ..."
python -m geoarches.evaluation.evaluate \
    "++predictions_path=$OUTPUT_PATH" \
    "++target_path=$DATA_PATH"

echo ""
echo "To generate visualizations:"
echo "  python -m geoarches.evaluation.plot \\"
echo "      ++predictions_path=$OUTPUT_PATH \\"
echo "      ++output_dir=/workspace/results/figures/"
