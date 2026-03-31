#!/usr/bin/env bash
# run_pairtune.sh — Fine-tune GP-MoLFormer toward a target molecular property
#
# Run this INSIDE the container (docker exec -it gp-molformer bash)
# OR pass it directly: docker exec gp-molformer bash /workspace/run_pairtune.sh
#
# OPTIONAL EDITS:
#   PROPERTY     — target property: qed | logp | drd2 (default: qed)
#   NUM_EPOCHS   — training epochs (default: 100)
#   EVAL_EPOCHS  — evaluate every N epochs (default: 10)
#   BATCH_SIZE   — batch size (default: 1200)
#
# NOTE: pairtune_training.patch must have been applied by docker_run.sh.
#       If you skipped that step, pair-tuning may fail.

set -euo pipefail

PROPERTY="${PROPERTY:-qed}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
EVAL_EPOCHS="${EVAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1200}"

cd /workspace/gp-molformer

echo "=== GP-MoLFormer Pair-Tuning ==="
echo "  Property    : $PROPERTY"
echo "  Epochs      : $NUM_EPOCHS"
echo "  Eval every  : $EVAL_EPOCHS epochs"
echo "  Batch size  : $BATCH_SIZE"
echo "  Output      : models/pairtune/$PROPERTY/"
echo ""
echo "Available properties: qed (drug-likeness), logp (penalized logP), drd2 (binding)"
echo ""

python -m scripts.pairtune_training "$PROPERTY" \
    --lamb \
    --num_epochs "$NUM_EPOCHS" \
    --eval_epochs "$EVAL_EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "Adapter checkpoint written to: models/pairtune/$PROPERTY/"
echo "  adapter_model.bin"
echo "  adapter_config.json"
echo ""
echo "To generate molecules with the fine-tuned model, load the adapter"
echo "on top of the base GP-MoLFormer checkpoint."
