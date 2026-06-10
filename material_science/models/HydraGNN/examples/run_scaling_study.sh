#!/usr/bin/env bash
# Submit a matched HydraGNN strong-scaling sweep.
#
# All node counts share identical env vars. Timing is train-only
# (HYDRAGNN_VALTEST=0); use collate_scaling_study.py to extract steady-state
# s/batch (mean of epochs 2–5 by default).
#
# Usage:
#   export AI4S_SHARED_DIR=/path/to/shared
#   export SBATCH_PARTITION=... SBATCH_ACCOUNT=...
#   ./run_scaling_study.sh              # submit 1,2,4,8 nodes
#   ./run_scaling_study.sh --warmup     # 1-node warm ckpt, then timed runs
#   ./run_scaling_study.sh --nodes 1,2  # subset
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AI4S_SHARED_DIR:?Set AI4S_SHARED_DIR}"

# ---------------------------------------------------------------------------
# Locked scaling-study contract (do not vary between node counts)
# ---------------------------------------------------------------------------
export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export HYDRAGNN_VALTEST=0
export HG_NUM_EPOCH=6
export HYDRAGNN_MAX_NUM_BATCH=50
export HG_BATCH_SIZE=200
export HG_PRECISION=fp64
export HYDRAGNN_TRACE_LEVEL=1
export HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-${AI4S_SHARED_DIR}/models/HydraGNN/outputs/scaling}"

SBATCH_PARTITION="${SBATCH_PARTITION:-YOUR_GPU_PARTITION}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_ACCOUNT}"

WARMUP=0
NODES_LIST="1,2,4,8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --warmup) WARMUP=1; shift ;;
    --nodes) NODES_LIST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

submit() {
  local n="$1" t="$2"
  local extra=()
  if [[ "$n" -ge 8 && -n "${SCALING_EXCLUDE_NODES:-}" ]]; then
    extra+=( --exclude="$SCALING_EXCLUDE_NODES" )
  fi
  sbatch --parsable \
    --partition="$SBATCH_PARTITION" \
    --account="$SBATCH_ACCOUNT" \
    --nodes="$n" \
    --time="$t" \
    "${extra[@]}" \
    --job-name="hydragnn-scale-${n}N" \
    --export=ALL,TORCH_NCCL_HIGH_PRIORITY,GPU_MAX_HW_QUEUES,HYDRAGNN_VALTEST,HG_NUM_EPOCH,HYDRAGNN_MAX_NUM_BATCH,HG_BATCH_SIZE,HG_PRECISION,HYDRAGNN_TRACE_LEVEL,HG_OUTPUT_DIR,HG_STARTFROM,HG_WARM_CKPT,AI4S_SHARED_DIR \
    "$SCRIPT_DIR/sbatch_train_amd.sh"
}

if [[ "$WARMUP" -eq 1 ]]; then
  export HG_NUM_EPOCH=1
  export HYDRAGNN_VALTEST=0
  unset HG_STARTFROM HG_WARM_CKPT
  WJ=$(submit 1 "00:30:00")
  echo "Warmup job submitted: $WJ (1 epoch, saves checkpoint under HG_OUTPUT_DIR/logs/)"
  echo "After it completes, re-run with HG_WARM_CKPT and HG_STARTFROM set."
  exit 0
fi

if [[ -n "${HG_WARM_CKPT:-}" && -n "${HG_STARTFROM:-}" ]]; then
  export HG_NUM_EPOCH="${HG_NUM_EPOCH:-5}"
  echo "Using warm checkpoint: $HG_WARM_CKPT -> logs/$HG_STARTFROM/"
fi

IFS=',' read -ra NODE_COUNTS <<< "$NODES_LIST"
declare -A TIME_LIMIT=( [1]="01:00:00" [2]="01:30:00" [4]="02:30:00" [8]="03:00:00" )
echo "=== HydraGNN scaling sweep ==="
echo "  Output dir : $HG_OUTPUT_DIR"
echo "  Epochs     : $HG_NUM_EPOCH (steady-state: epochs 2–5 via collate_scaling_study.py)"
echo "  RCCL       : TORCH_NCCL_HIGH_PRIORITY=$TORCH_NCCL_HIGH_PRIORITY GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES"
echo "  Start from : ${HG_STARTFROM:-none}"
echo ""

JOB_IDS=()
for n in "${NODE_COUNTS[@]}"; do
  jid=$(submit "$n" "${TIME_LIMIT[$n]:-02:00:00}")
  echo "  ${n}-node -> job $jid"
  JOB_IDS+=("$jid")
done

echo ""
echo "Submitted: ${JOB_IDS[*]}"
echo "After all complete:"
echo "  python3 collate_scaling_study.py --log-dir . --jobs $(IFS=,; echo "${JOB_IDS[*]}") -o scaling_study"
