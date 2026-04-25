#!/usr/bin/env bash
# run_train.sh — Pretrain and fine-tune ArchesWeather / ArchesWeatherGen
#
# Run INSIDE the container launched by docker_run.sh.
#
# OPTIONAL EDITS:
#   MODEL        — archesweather | archesweathergen (default: archesweather)
#   PHASE        — pretrain | finetune (default: pretrain)
#   SEED         — seed index 0-3 for ArchesWeather (default: 0)
#   PRECISION    — 32-true | 16-mixed | bf16-mixed (default: 16-mixed)
#   BATCH_SIZE   — per-GPU batch size (default: 8 for 16-mixed on MI300X)
#   MAX_STEPS    — training steps (default: 250000 for pretrain, 50000 for finetune)
#   DATA_PATH    — ERA5 dataset path inside container (default: /data/era5_240/full)
#   LOAD_FROM    — checkpoint path for fine-tuning phase (required if PHASE=finetune)
#   NAME         — run name used for checkpoint dir (default: auto-generated)

set -euo pipefail

MODEL="${MODEL:-archesweather}"
PHASE="${PHASE:-pretrain}"
SEED="${SEED:-0}"
PRECISION="${PRECISION:-16-mixed}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATA_PATH="${DATA_PATH:-/data/era5_240/full}"
LOAD_FROM="${LOAD_FROM:-}"

# Set default steps based on phase and model
if [[ "$PHASE" == "pretrain" ]]; then
    MAX_STEPS="${MAX_STEPS:-$([ "$MODEL" == "archesweathergen" ] && echo 200000 || echo 250000)}"
    NAME="${NAME:-${MODEL}-seed${SEED}}"
else
    MAX_STEPS="${MAX_STEPS:-$([ "$MODEL" == "archesweathergen" ] && echo 60000 || echo 50000)}"
    NAME="${NAME:-${MODEL}-seed${SEED}-finetuned}"
fi

# --- Validate ---
if [[ ! -d "$DATA_PATH" ]]; then
    echo "WARNING: DATA_PATH not found: $DATA_PATH"
    echo "  Download ERA5 data first:"
    echo "    python -m geoarches.download.dl_era --output_dir $DATA_PATH"
    echo "  (~735 GB, may take hours depending on connection)"
fi

if [[ "$PHASE" == "finetune" && -z "$LOAD_FROM" ]]; then
    echo "ERROR: LOAD_FROM must be set for fine-tuning phase."
    echo "  Set LOAD_FROM to the path of a pretrained checkpoint, e.g.:"
    echo "  LOAD_FROM=/workspace/checkpoints/${MODEL}-seed${SEED} bash run_train.sh"
    exit 1
fi

echo "=== ArchesWeather Training ==="
echo "  Model      : $MODEL"
echo "  Phase      : $PHASE"
echo "  Seed       : $SEED"
echo "  Precision  : $PRECISION  (MI300X: batch 5 at 32-true, 8 at 16-mixed/bf16)"
echo "  Batch size : $BATCH_SIZE"
echo "  Max steps  : $MAX_STEPS"
echo "  Data       : $DATA_PATH"
echo "  Run name   : $NAME"
[[ -n "$LOAD_FROM" ]] && echo "  Load from  : $LOAD_FROM"
echo ""

# Build the command
CMD=(
    python -m geoarches.main_hydra ++log=True
    dataloader=era5
    "module=$MODEL"
    "++name=$NAME"
    "++cluster.precision=$PRECISION"
    "++batch_size=$BATCH_SIZE"
    "++max_steps=$MAX_STEPS"
    "++save_step_frequency=50000"
    "++dataloader.dataset.path=$DATA_PATH"
)

[[ -n "$LOAD_FROM" ]] && CMD+=("++load_from=$LOAD_FROM")

"${CMD[@]}"
