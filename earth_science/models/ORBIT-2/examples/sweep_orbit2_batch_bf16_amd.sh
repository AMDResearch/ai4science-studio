#!/usr/bin/env bash
# Submit ORBIT-2 bf16 + SDPA batch-size probes for HBM saturation (1 node × 8 GPUs).
#
# Usage (login node, repo root):
#   export AI4S_SHARED_DIR=...
#   export OMNIHUB_TOOLS_DIR=...
#   export ORBIT2_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/code/bayes-cast
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
#   # Optional: export ORBIT2_ERA5_SPATIAL_RES=111
#   bash earth_science/models/ORBIT-2/examples/sweep_orbit2_batch_bf16_amd.sh 64 128 256
#
# Each job is independent; use the printed job IDs with:
#   python3 earth_science/models/ORBIT-2/examples/run_fom_extractor.py --job-dir "$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>"
#   python3 earth_science/models/ORBIT-2/examples/report_orbit2_gpu_baseline.py --job-dir ".../perf-runs/<jobid>"
#
# For binary search, pick two bracketing batch sizes from VRAM logs, then add midpoints.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
SBATCH="${SCRIPT_DIR}/sbatch_train_perf_amd.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch1> [batch2 ...]" >&2
  exit 1
fi

: "${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR}"
: "${OMNIHUB_TOOLS_DIR:?set OMNIHUB_TOOLS_DIR}"

export ORBIT2_DATA_TYPE="${ORBIT2_DATA_TYPE:-bfloat16}"
export ORBIT2_FUSED_ATTN="${ORBIT2_FUSED_ATTN:-DEFAULT}"
export ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-edm_8m_era5_1x8.yaml}"
export ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-6}"
export ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES:-20}"

for B in "$@"; do
  export ORBIT2_BATCH_SIZE="$B"
  echo "--- sbatch ORBIT2_BATCH_SIZE=$B ---"
  ( cd "$REPO_ROOT" && sbatch --export=ALL,ORBIT2_BATCH_SIZE="$B",ORBIT2_DATA_TYPE="$ORBIT2_DATA_TYPE",ORBIT2_FUSED_ATTN="$ORBIT2_FUSED_ATTN",ORBIT2_CONFIG_TEMPLATE="$ORBIT2_CONFIG_TEMPLATE",ORBIT2_MAX_EPOCH="$ORBIT2_MAX_EPOCH",ORBIT2_MAX_BATCHES="$ORBIT2_MAX_BATCHES" "$SBATCH" ) || true
done

echo "Done. Track jobs: squeue -u \$USER"
