#!/usr/bin/env bash
# Submit ORBIT-2 perf baseline: ERA5 1.0° same-dir + interm_8m_lux_era5.yaml (matches
# sbatch_train_perf_amd.sh defaults). Requires staged NPZ under ORBIT2_DATA_ROOT.
#
# Usage (from anywhere):
#   export AI4S_SHARED_DIR=/path/to/shared
#   export PERF_TOOLS_DIR=/path/to/perf-tools
#   ./submit_perf_baseline_era5_amd.sh
# Extra sbatch flags: ./submit_perf_baseline_era5_amd.sh --partition=lux --account=myacct

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AI4S_SHARED_DIR:?set AI4S_SHARED_DIR}"
: "${PERF_TOOLS_DIR:?set PERF_TOOLS_DIR}"

export ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/data/superres/era5/1.0_deg}"
export ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-edm_8m_era5_1x8.yaml}"
export ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-6}"

# examples -> ORBIT-2 -> models -> earth_science -> repo root
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
SBATCH_SCRIPT="$REPO_ROOT/earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh"
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "error: repo root not found from $SCRIPT_DIR (expected $SBATCH_SCRIPT)" >&2
  exit 2
fi

cd "$REPO_ROOT"
exec sbatch "$@" "$SBATCH_SCRIPT"
