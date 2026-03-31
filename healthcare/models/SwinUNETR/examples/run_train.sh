#!/usr/bin/env bash
# run_train.sh — Train SwinUNETR on NSCLC-Radiomics with AMD optimizations
#
# Run INSIDE the container launched by docker_run.sh (train image, ROCm 6.4).
#
# OPTIONAL EDITS:
#   DATA_DIR     — where MONAI downloads NSCLC-Radiomics (default: /data)
#   CKPT_DIR     — where to save checkpoints (default: /workspace/checkpoints)
#   MAX_EPOCHS   — training epochs (default: 700)
#   ROI_X/Y/Z   — patch size in voxels (default: 96 96 96; MI300X supports up to 480 480 96)
#   FEATURE_SIZE — encoder feature size (default: 48)
#   NUM_WORKERS  — DataLoader workers (default: 64; >32 eliminates data bottleneck)
#   AMP_DTYPE    — float16 or bfloat16 (default: float16 — bfloat16 underperforms here)

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints}"
MAX_EPOCHS="${MAX_EPOCHS:-700}"
ROI_X="${ROI_X:-96}"
ROI_Y="${ROI_Y:-96}"
ROI_Z="${ROI_Z:-96}"
FEATURE_SIZE="${FEATURE_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-64}"
AMP_DTYPE="${AMP_DTYPE:-float16}"

mkdir -p "$DATA_DIR" "$CKPT_DIR"

# MIOpen auto-tuning — default on ROCm 6.4+ / PyTorch 2.6+, but set explicitly for older stacks
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=3

echo "=== SwinUNETR Training ==="
echo "  Data dir     : $DATA_DIR  (MONAI auto-downloads NSCLC-Radiomics)"
echo "  Checkpoint   : $CKPT_DIR"
echo "  ROI size     : ${ROI_X}×${ROI_Y}×${ROI_Z}  (MI300X supports up to 480×480×96)"
echo "  Feature size : $FEATURE_SIZE"
echo "  Max epochs   : $MAX_EPOCHS"
echo "  Num workers  : $NUM_WORKERS  (>32 eliminates DataLoader bottleneck)"
echo "  AMP dtype    : $AMP_DTYPE  (bfloat16 underperforms — use float16)"
echo "  MIOpen tuning: MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE (3× speedup)"
echo ""

# Locate training script from silogen recipe or MONAI research-contributions
TRAIN_SCRIPT=""
for candidate in \
    /recipe/train.py \
    /workspace/SwinUNETR/BTCV/main.py \
    /workspace/research-contributions/SwinUNETR/BTCV/main.py; do
    [[ -f "$candidate" ]] && TRAIN_SCRIPT="$candidate" && break
done

if [[ -z "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found. Clone the recipe source:"
    echo "  Silogen: /recipe/train.py  (mounted if launched via docker_run.sh)"
    echo "  MONAI:   git clone https://github.com/Project-MONAI/research-contributions.git /workspace/research-contributions"
    exit 1
fi

echo "Using training script: $TRAIN_SCRIPT"
echo ""

python "$TRAIN_SCRIPT" \
    --data_dir="$DATA_DIR" \
    --logdir="$CKPT_DIR" \
    --max_epochs="$MAX_EPOCHS" \
    --roi_x="$ROI_X" \
    --roi_y="$ROI_Y" \
    --roi_z="$ROI_Z" \
    --feature_size="$FEATURE_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --amp \
    --use_checkpoint
