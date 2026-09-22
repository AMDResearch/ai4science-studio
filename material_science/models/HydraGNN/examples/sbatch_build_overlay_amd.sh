#!/usr/bin/env bash
# Build HydraGNN Apptainer overlay on an ORNL Lux compute node.
#
# Prerequisite: sbatch sbatch_build_adios2_amd.sh  (MPI ADIOS2 install prefix)
#
# Quick start:
#   sbatch sbatch_build_adios2_amd.sh
#   sbatch sbatch_build_overlay_amd.sh
#
# Override MPI baseline:
#   export HG_LUX_CONFIG_BUNDLE=/path/to/config-bundle
#   export HG_LUX_MPI_OPT=/path/to/mpi-build-tests/opt
#   export HG_USE_LUX_MPI=1          # force Shelby stack
#   export HG_USE_LUX_MPI=0          # fall back to /opt/ompi or spack

#SBATCH -J hydragnn-overlay
#SBATCH --partition=batch
#SBATCH --account=ven136
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
  echo "  MPI baseline : Shelby config-bundle (Lux)"
  echo "  HG_OMPI_DIR  : ${HG_LUX_OMPI_CONTAINER} (Lustre → /opt/lux-mpi in container)"
  echo "  Config bundle: ${HG_LUX_CONFIG_BUNDLE}"
else
  echo "  MPI baseline : /opt/ompi or spack (HG_USE_LUX_MPI=0 or bundle missing)"
fi
echo ""

exec bash "${SCRIPT_DIR}/build_overlay_amd.sh"
