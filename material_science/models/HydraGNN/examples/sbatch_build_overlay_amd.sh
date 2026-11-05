#!/usr/bin/env bash
# Build HydraGNN Apptainer overlay on an ORNL Lux compute node.
#
# Prerequisite: sbatch sbatch_build_adios2_amd.sh  (MPI ADIOS2 install prefix)
#
# Quick start:
#   sbatch sbatch_build_adios2_amd.sh
#   sbatch sbatch_build_overlay_amd.sh
#
# ORNL Lux uses the site modules rocm/7.2.4, amd-llvm/7.2.4, and
# openmpi/5.0.10. Override HG_LUX_*_MODULE only when testing a new module.

#SBATCH -J hydragnn-overlay
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o hydragnn-overlay-build-%j.out
#SBATCH -e hydragnn-overlay-build-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

export SCRATCH_LOCAL="${SCRATCH_LOCAL:-/tmp}"

echo "=== HydraGNN overlay build (compute node) ==="
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  AI4S_SHARED_DIR : ${AI4S_SHARED_DIR:-<unset>}"
echo ""

# Proxy passthrough for PyG wheel download inside build_overlay_amd.sh
for k in https_proxy http_proxy HTTPS_PROXY HTTP_PROXY no_proxy NO_PROXY; do
  v="${!k:-}"
  [[ -n "$v" ]] && export "$k=$v"
done

# shellcheck source=lux_mpi_env.sh
source "${SCRIPT_DIR}/lux_mpi_env.sh"
if lux_mpi_available; then
  lux_mpi_setup_binds
  echo "  MPI baseline : system ${HG_LUX_OPENMPI_MODULE} (ob1/tcp runtime)"
  echo "  HG_OMPI_DIR  : ${HG_OMPI_DIR}"
else
  echo "  MPI baseline : unavailable"
fi
echo ""

exec bash "${SCRIPT_DIR}/build_overlay_amd.sh"
