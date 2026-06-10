#!/usr/bin/env bash
# Submit matched ORBIT-2 strong-scaling sweep (1/2/4/8 nodes).
#
# Usage:
#   export AI4S_SHARED_DIR=/path/to/shared
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin
#   ./run_scaling_study.sh
#   ./run_scaling_study.sh --nodes 1,2
#
# ERA5 1.0_deg same-dir (timing only):
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
#   export ORBIT2_CONFIG_TEMPLATE=interm_8m_lux_era5.yaml
#   export ORBIT2_SCALING_TAG=era5
#   ./run_scaling_study.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AI4S_SHARED_DIR:?Set AI4S_SHARED_DIR}"

export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export ORBIT2_DATA_TYPE=float32
# max_epochs=6 → trains epochs 0–4; collate uses steady epochs 2–4 for FOM
export ORBIT2_MAX_EPOCH=6
export ORBIT2_MAX_BATCHES=20
export ORBIT2_BATCH_SIZE=4
export ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/data/superres/prism/10.0_arcmin}"
export ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-interm_8m_lux.yaml}"
export ORBIT2_SCALING_TAG="${ORBIT2_SCALING_TAG:-prism}"
export ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR:-${AI4S_SHARED_DIR}/models/ORBIT-2/outputs/scaling-${ORBIT2_SCALING_TAG}}"

SBATCH_PARTITION="${SBATCH_PARTITION:-YOUR_PARTITION_HERE}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_ACCOUNT_HERE}"

NODES_LIST="1,2,4,8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES_LIST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOG_DIR="${ORBIT2_SLURM_LOG_DIR:-${AI4S_SHARED_DIR}/models/ORBIT-2/outputs/train/logs}"
mkdir -p "$LOG_DIR"

submit() {
  local n="$1" t="$2"
  sbatch --parsable \
    --partition="$SBATCH_PARTITION" \
    --account="$SBATCH_ACCOUNT" \
    --nodes="$n" \
    --time="$t" \
    --job-name="orbit2-scale-${n}N" \
    --output="${LOG_DIR}/orbit2-train-%j.out" \
    --error="${LOG_DIR}/orbit2-train-%j.out" \
    --export=ALL,TORCH_NCCL_HIGH_PRIORITY,GPU_MAX_HW_QUEUES,ORBIT2_DATA_TYPE,ORBIT2_MAX_EPOCH,ORBIT2_MAX_BATCHES,ORBIT2_BATCH_SIZE,ORBIT2_DATA_ROOT,ORBIT2_CONFIG_TEMPLATE,ORBIT2_SCALING_TAG,ORBIT2_OUTPUT_DIR,AI4S_SHARED_DIR \
    "$SCRIPT_DIR/sbatch_train_amd.sh"
}

declare -A TIME_LIMIT=( [1]="02:00:00" [2]="02:30:00" [4]="03:00:00" [8]="04:00:00" )
IFS=',' read -ra NODE_COUNTS <<< "$NODES_LIST"
echo "=== ORBIT-2 scaling sweep (${ORBIT2_SCALING_TAG}) ==="
echo "  Data root    : $ORBIT2_DATA_ROOT"
echo "  Config       : $ORBIT2_CONFIG_TEMPLATE"
echo "  Output base  : $ORBIT2_OUTPUT_DIR"
echo "  Epochs: $ORBIT2_MAX_EPOCH  Max batches: $ORBIT2_MAX_BATCHES"
echo ""

JOB_IDS=()
for n in "${NODE_COUNTS[@]}"; do
  export ORBIT2_OUTPUT_DIR="${AI4S_SHARED_DIR}/models/ORBIT-2/outputs/scaling-${ORBIT2_SCALING_TAG}/${n}node"
  jid=$(submit "$n" "${TIME_LIMIT[$n]:-02:00:00}")
  echo "  ${n}-node -> job $jid"
  JOB_IDS+=("$jid")
done

echo ""
echo "Submitted: ${JOB_IDS[*]}"
echo "After all complete:"
echo "  python3 collate_scaling_study.py --log-dir ${LOG_DIR} --jobs $(IFS=,; echo "${JOB_IDS[*]}") \\"
echo "    --steady-epoch-start 2 --warmup-batches-per-epoch 1 --require-loss-sanity \\"
echo "    -o ${ORBIT2_OUTPUT_DIR}/scaling_study"
