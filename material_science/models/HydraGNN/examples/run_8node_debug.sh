#!/usr/bin/env bash
# Short 8-node RCCL/DDP debug smoke — fail fast, verbose NCCL logs.
#
# Usage:
#   export AI4S_SHARED_DIR=/lustre/orion/ven136/proj-shared/ashwinaji/ai4science
#   export SBATCH_PARTITION=batch SBATCH_ACCOUNT=ven136
#   ./run_8node_debug.sh
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AI4S_SHARED_DIR:?Set AI4S_SHARED_DIR}"

export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HYDRAGNN_VALTEST=0
export HG_NUM_EPOCH=1
export HYDRAGNN_MAX_NUM_BATCH=5
export HG_BATCH_SIZE=200
export HG_PRECISION=fp64
export HYDRAGNN_TRACE_LEVEL=1
export HG_DATASETS="${HG_DATASETS:-ANI1x,Alexandria}"

# RCCL debug — logs land in SLURM stdout (rank-interleaved).
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,COLL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

LOG_DIR="${AI4S_SHARED_DIR}/models/HydraGNN/logs"
mkdir -p "$LOG_DIR"
export HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-${AI4S_SHARED_DIR}/models/HydraGNN/outputs/debug-8n-rccl}"
mkdir -p "$HG_OUTPUT_DIR"

SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-ven136}"
SBATCH_TIME="${SBATCH_TIME:-00:30:00}"
NODES="${NODES:-8}"

echo "=== HydraGNN 8-node RCCL debug smoke ==="
echo "  Nodes      : $NODES"
echo "  Time limit : $SBATCH_TIME"
echo "  Epochs     : $HG_NUM_EPOCH"
echo "  Max batches: $HYDRAGNN_MAX_NUM_BATCH"
echo "  Output dir : $HG_OUTPUT_DIR"
echo "  NCCL_DEBUG : $NCCL_DEBUG (subsys=$NCCL_DEBUG_SUBSYS, timeout=${NCCL_TIMEOUT}s)"
echo ""

jid=$(sbatch --parsable \
  --partition="$SBATCH_PARTITION" \
  --account="$SBATCH_ACCOUNT" \
  --nodes="$NODES" \
  --time="$SBATCH_TIME" \
  --job-name="hydragnn-dbg-${NODES}N" \
  -o "${LOG_DIR}/hydragnn-train-%j.out" \
  --export=ALL,TORCH_NCCL_HIGH_PRIORITY,GPU_MAX_HW_QUEUES,HYDRAGNN_VALTEST,HG_NUM_EPOCH,HYDRAGNN_MAX_NUM_BATCH,HG_BATCH_SIZE,HG_PRECISION,HYDRAGNN_TRACE_LEVEL,HG_OUTPUT_DIR,HG_DATASETS,NCCL_DEBUG,NCCL_DEBUG_SUBSYS,NCCL_TIMEOUT,NCCL_ASYNC_ERROR_HANDLING,AI4S_SHARED_DIR \
  "$SCRIPT_DIR/sbatch_train_amd.sh")

echo "Submitted debug job: $jid"
echo "  Log : ${LOG_DIR}/hydragnn-train-${jid}.out"
echo ""
echo "Watch:"
echo "  tail -f ${LOG_DIR}/hydragnn-train-${jid}.out"
echo "  squeue -j $jid"
