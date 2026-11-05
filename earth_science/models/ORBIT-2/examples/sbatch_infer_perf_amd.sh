#!/usr/bin/env bash
# ORBIT-2 inference + Omnistat on AMD Instinct via SLURM.
# Thin wrapper around run_infer_perf_amd.sh.
#
#   export AI4S_SHARED_DIR=/path/to/shared
#   export PERF_TOOLS_DIR=/path/to/perf-tools
#   export ORBIT2_ROOT=/path/to/ORBIT-2-clone
#   export ORBIT2_USE_SYNTHETIC=1
#   sbatch sbatch_infer_perf_amd.sh

#SBATCH -J orbit2-infer-perf
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
#SBATCH -o orbit2-infer-perf-%j.out
#SBATCH -e orbit2-infer-perf-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

exec bash "$SCRIPT_DIR/run_infer_perf_amd.sh" "$@"
