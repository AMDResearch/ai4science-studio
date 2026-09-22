#!/usr/bin/env bash
#SBATCH -J hydragnn-adios2
#SBATCH --partition=batch
#SBATCH --account=ven136
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o hydragnn-adios2-build-%j.out
#SBATCH -e hydragnn-adios2-build-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

export SCRATCH_LOCAL="${SCRATCH_LOCAL:-/tmp}"
module purge 2>/dev/null || true
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH PKG_CONFIG_PATH
for k in https_proxy http_proxy HTTPS_PROXY HTTP_PROXY no_proxy NO_PROXY; do
  v="${!k:-}"
  [[ -n "$v" ]] && export "$k=$v"
done

exec bash "${SCRIPT_DIR}/build_adios2_amd.sh"
