#!/usr/bin/env bash
# run_inference.sh — Sample 3D molecules with SemlaFlow
#
# Run inside the container: docker exec -it semlaflow bash /workspace/run_inference.sh
#
# OPTIONAL EDITS:
#   CHECKPOINT    — path to pretrained .ckpt file (download from upstream GitHub)
#   DATASET       — dataset name: qm9 | drugs (default: drugs)
#   OUTPUT_FILE   — where to write generated structures (default: /workspace/generated.sdf)
#   NUM_STEPS     — ODE solver steps (default: 20; minimum recommended)
#   USE_COMPILE   — set to 1 to enable torch.compile for ~44% speedup (default: 1)
#   NO_EMA        — set to 1 to skip EMA for ~8% additional speedup (default: 1)

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
DATASET="${DATASET:-drugs}"
OUTPUT_FILE="${OUTPUT_FILE:-/workspace/generated.sdf}"
NUM_STEPS="${NUM_STEPS:-20}"
USE_COMPILE="${USE_COMPILE:-1}"
NO_EMA="${NO_EMA:-1}"

cd /workspace/semla-flow

# --- Check checkpoint ---
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: CHECKPOINT is not set."
    echo ""
    echo "Download a pretrained checkpoint from the upstream GitHub repo:"
    echo "  https://github.com/rssrwn/semla-flow"
    echo "  (Google Drive links in the README)"
    echo ""
    echo "Then set the path:"
    echo "  CHECKPOINT=/workspace/checkpoints/semlaflow_drugs.ckpt bash run_inference.sh"
    exit 1
fi

# --- Build predict command ---
PREDICT_ARGS=(
    predict "$OUTPUT_FILE"
    --ckpt_path "$CHECKPOINT"
    --dataset "$DATASET"
    --integration_steps "$NUM_STEPS"
)

[[ "$NO_EMA" == "1" ]] && PREDICT_ARGS+=(--no_ema)

# --- Apply torch.compile if requested (set compiler cache limit) ---
if [[ "$USE_COMPILE" == "1" ]]; then
    export TORCH_COMPILE_CACHE_SIZE=1000
    PREDICT_ARGS+=(--compile)
fi

echo "=== SemlaFlow Inference ==="
echo "  Checkpoint    : $CHECKPOINT"
echo "  Dataset       : $DATASET"
echo "  ODE steps     : $NUM_STEPS"
echo "  torch.compile : $([ "$USE_COMPILE" == "1" ] && echo "enabled (~44% speedup)" || echo "disabled")"
echo "  No EMA        : $([ "$NO_EMA" == "1" ] && echo "yes (~8% speedup)" || echo "no")"
echo "  Output        : $OUTPUT_FILE"
echo ""

python -m semlaflow.scripts.eval "${PREDICT_ARGS[@]}"

echo ""
echo "Generated molecules written to: $OUTPUT_FILE"
