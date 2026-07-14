#!/usr/bin/env bash
# Submit a matched ORBIT-2 WEAK-scaling sweep (1/2/4/8 nodes).
#
# Weak scaling: per-rank batch is held FIXED at every node count (each GPU does the same
# work; global batch grows as batch x 8 x N). Ideal result is constant steady step time vs N;
# the efficiency reported by collate_scaling_study.py is then a comm-overlap (RCCL bandwidth)
# metric, not a classic strong-scaling speedup. Parallelism is HSDP: fsdp=8 within a node
# (XGMI), simple_ddp=N across nodes (IB/ANP) — see ORBIT2_SCALING_FSDP below.
#
# Usage (PRISM res_slimvit, default):
#   export AI4S_SHARED_DIR=/path/to/shared
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin
#   ./run_scaling_study.sh
#   ./run_scaling_study.sh --nodes 1,2
#
# ERA5 1.0_deg same-dir res_slimvit (timing only):
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
#   export ORBIT2_CONFIG_TEMPLATE=interm_8m_era5.yaml
#   export ORBIT2_SCALING_TAG=era5
#   ./run_scaling_study.sh
#
# Bayes-CAST EDM, compute-saturated (per-rank batch fills the GPU at every node count):
#   export ORBIT2_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/code/bayes-cast
#   export ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
#   export ORBIT2_ERA5_SPATIAL_RES=111 ORBIT2_DATA_TYPE=bfloat16 ORBIT2_FUSED_ATTN=DEFAULT
#   export ORBIT2_BATCH_SIZE=1024 ORBIT2_SCALING_TAG=edm-era5
#   ./run_scaling_study.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AI4S_SHARED_DIR:?Set AI4S_SHARED_DIR}"

export TORCH_NCCL_HIGH_PRIORITY=1
export GPU_MAX_HW_QUEUES=2
export ORBIT2_DATA_TYPE="${ORBIT2_DATA_TYPE:-bfloat16}"
# max_epochs=6 → trains epochs 0–4; collate uses steady epochs 2–4 for FOM
# All sizing knobs honor caller overrides (EDM weak-scaling uses larger batch).
export ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-6}"
export ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES:-20}"
export ORBIT2_BATCH_SIZE="${ORBIT2_BATCH_SIZE:-4}"
export ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/data/superres/prism/10.0_arcmin}"
export ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-interm_8m_prism.yaml}"
# Pass-through for the EDM (Bayes-CAST) path; harmless when unset for the PRISM/ERA5 res_slimvit path.
[[ -n "${ORBIT2_ROOT:-}" ]] && export ORBIT2_ROOT
[[ -n "${ORBIT2_ERA5_SPATIAL_RES:-}" ]] && export ORBIT2_ERA5_SPATIAL_RES
export ORBIT2_FUSED_ATTN="${ORBIT2_FUSED_ATTN:-DEFAULT}"
# Weak-scaling parallelism: keep FSDP sharding *within* a node (fast XGMI) and add a
# data-parallel replica *per node* across the IB fabric → per-rank work stays fixed at
# every node count (fsdp=8 simple_ddp=N), instead of the render default (fsdp=N simple_ddp=8)
# which would shard weights across slow inter-node links. Override fsdp/node via ORBIT2_SCALING_FSDP.
export ORBIT2_SCALING_FSDP="${ORBIT2_SCALING_FSDP:-8}"
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
  # Weak-scaling HSDP: fsdp = GPUs/node (intra-node), simple_ddp = node count (inter-node).
  export ORBIT2_FSDP="$ORBIT2_SCALING_FSDP"
  export ORBIT2_SIMPLE_DDP="$n"
  sbatch --parsable \
    --partition="$SBATCH_PARTITION" \
    --account="$SBATCH_ACCOUNT" \
    --nodes="$n" \
    --time="$t" \
    --job-name="orbit2-scale-${n}N" \
    --output="${LOG_DIR}/orbit2-train-%j.out" \
    --error="${LOG_DIR}/orbit2-train-%j.out" \
    --export=ALL,TORCH_NCCL_HIGH_PRIORITY,GPU_MAX_HW_QUEUES,ORBIT2_DATA_TYPE,ORBIT2_MAX_EPOCH,ORBIT2_MAX_BATCHES,ORBIT2_BATCH_SIZE,ORBIT2_DATA_ROOT,ORBIT2_CONFIG_TEMPLATE,ORBIT2_SCALING_TAG,ORBIT2_OUTPUT_DIR,AI4S_SHARED_DIR,ORBIT2_ROOT,ORBIT2_ERA5_SPATIAL_RES,ORBIT2_FUSED_ATTN,ORBIT2_FSDP,ORBIT2_SIMPLE_DDP,ORBIT2_DISABLE_CKPT=1 \
    "$SCRIPT_DIR/sbatch_train_amd.sh"
}

declare -A TIME_LIMIT=( [1]="02:00:00" [2]="02:30:00" [4]="03:00:00" [8]="04:00:00" )
IFS=',' read -ra NODE_COUNTS <<< "$NODES_LIST"
echo "=== ORBIT-2 scaling sweep (${ORBIT2_SCALING_TAG}) ==="
echo "  Data root    : $ORBIT2_DATA_ROOT"
echo "  Config       : $ORBIT2_CONFIG_TEMPLATE"
echo "  Output base  : $ORBIT2_OUTPUT_DIR"
echo "  Per-rank batch: $ORBIT2_BATCH_SIZE (fixed → weak scaling)  dtype: $ORBIT2_DATA_TYPE"
echo "  Parallelism  : fsdp=$ORBIT2_SCALING_FSDP/node, simple_ddp=N (HSDP)"
echo "  Epochs: $ORBIT2_MAX_EPOCH  Max batches: $ORBIT2_MAX_BATCHES"
[[ -n "${ORBIT2_ROOT:-}" ]] && echo "  ORBIT2_ROOT  : $ORBIT2_ROOT"
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
