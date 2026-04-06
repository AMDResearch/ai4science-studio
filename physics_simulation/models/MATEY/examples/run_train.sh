#!/usr/bin/env bash
# Run MATEY training (single-GPU or DDP via torchrun).
#
# Environment variables (all optional — defaults shown):
#   MATEY_RUN_NAME   my_run
#   MATEY_CONFIG     basic_config
#   MATEY_YAML       /matey/config/Demo_JHUTDB_TT.yaml
#   MATEY_DATA_DIR   /data/JHTDB
#   MATEY_EPOCHS     100
#   MATEY_BATCH_SIZE 4
#   MATEY_LR         1e-4
#   MATEY_USE_DDP    0   (set to 1 for multi-GPU DDP)
#   MATEY_NUM_GPUS   1   (used when MATEY_USE_DDP=1)
#
# Usage (inside Docker container):
#   bash run_train.sh
#   MATEY_USE_DDP=1 MATEY_NUM_GPUS=8 bash run_train.sh

set -euo pipefail

MATEY_RUN_NAME="${MATEY_RUN_NAME:-my_run}"
MATEY_CONFIG="${MATEY_CONFIG:-basic_config}"
MATEY_YAML="${MATEY_YAML:-/matey/config/Demo_JHUTDB_TT.yaml}"
MATEY_DATA_DIR="${MATEY_DATA_DIR:-/data/JHTDB}"
MATEY_EPOCHS="${MATEY_EPOCHS:-100}"
MATEY_BATCH_SIZE="${MATEY_BATCH_SIZE:-4}"
MATEY_LR="${MATEY_LR:-1e-4}"
MATEY_USE_DDP="${MATEY_USE_DDP:-0}"
MATEY_NUM_GPUS="${MATEY_NUM_GPUS:-1}"

echo "=== MATEY Training ==="
echo "  Run name   : ${MATEY_RUN_NAME}"
echo "  Config     : ${MATEY_CONFIG}"
echo "  YAML       : ${MATEY_YAML}"
echo "  Data dir   : ${MATEY_DATA_DIR}"
echo "  Epochs     : ${MATEY_EPOCHS}"
echo "  Batch size : ${MATEY_BATCH_SIZE}"
echo "  LR         : ${MATEY_LR}"
echo "  DDP        : ${MATEY_USE_DDP} (GPUs: ${MATEY_NUM_GPUS})"
echo ""

if [[ ! -f "${MATEY_YAML}" ]]; then
    echo "error: YAML config not found: ${MATEY_YAML}" >&2
    echo "  Set MATEY_YAML to point to a valid config file." >&2
    exit 1
fi

COMMON_ARGS=(
    --run_name   "${MATEY_RUN_NAME}"
    --config     "${MATEY_CONFIG}"
    --yaml_config "${MATEY_YAML}"
)

if [[ "${MATEY_USE_DDP}" == "1" ]]; then
    torchrun --nproc_per_node="${MATEY_NUM_GPUS}" \
        /matey/basic_usage.py \
        "${COMMON_ARGS[@]}" \
        --use_ddp
else
    python /matey/basic_usage.py \
        "${COMMON_ARGS[@]}"
fi
