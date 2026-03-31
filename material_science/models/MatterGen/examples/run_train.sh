#!/usr/bin/env bash
# run_train.sh — Train MatterGen on the mp_20 dataset
#
# Run inside the container: docker exec -it mattergen bash /workspace/run_train.sh
#
# OPTIONAL EDITS:
#   DATA_DIR    — path to prepared dataset cache (default: /workspace/mattergen/datasets)
#   OUTPUT_DIR  — where to write checkpoints (default: /workspace/mattergen/checkpoints)
#   MAX_EPOCHS  — training epochs (default: 900, ~15h on single MI300X)

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/mattergen/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/mattergen/checkpoints}"
MAX_EPOCHS="${MAX_EPOCHS:-900}"

cd /workspace/mattergen

# --- Step 1: Download and prepare mp_20 dataset (first run only) ---
if [[ ! -d "$DATA_DIR/cache" ]]; then
    echo "=== Preparing mp_20 dataset ==="
    echo "Pulling data via Git LFS ..."
    git lfs pull -I data-release/mp-20/ --exclude=""

    mkdir -p "$DATA_DIR"
    unzip -q data-release/mp-20/mp_20.zip -d "$DATA_DIR"

    echo "Converting CSV to model dataset format ..."
    csv-to-dataset \
        --csv-folder "$DATA_DIR/mp_20/" \
        --dataset-name mp_20 \
        --cache-folder "$DATA_DIR/cache"

    echo "Dataset ready at $DATA_DIR/cache"
    echo ""
fi

# --- Step 2: Train ---
echo "=== MatterGen Training ==="
echo "  Dataset     : $DATA_DIR/cache"
echo "  Max epochs  : $MAX_EPOCHS"
echo "  Est. time   : ~15 hours on single MI300X"
echo ""

mattergen-train \
    data_module=mp_20 \
    ~trainer.logger \
    ++trainer.max_epochs="$MAX_EPOCHS" \
    ++data_module.dataset.cache_folder="$DATA_DIR/cache"

echo ""
echo "Training complete. Checkpoints in: $OUTPUT_DIR"
