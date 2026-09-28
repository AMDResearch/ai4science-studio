#!/usr/bin/env bash
# Ladder smoke: Apptainer + overlay + system OpenMPI with ob1/TCP (no training).
#
# Isolates the Phase-1 hang that stopped after "Launching training" with no
# rank banners (MPI_Init / overlay-exec never returned).
#
#   export AI4S_SHARED_DIR=/path/to/shared
#   sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT \
#     --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=00:20:00 \
#     earth_science/models/ORBIT-2/examples/sbatch_mpi_tcp_smoke_amd.sh

#SBATCH --job-name=orbit2-mpi-smoke
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH --output=orbit2-mpi-smoke-%j.out
#SBATCH --error=orbit2-mpi-smoke-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

: "${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}"
ORBIT2_SIF="${ORBIT2_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
ORBIT2_OVERLAY="${ORBIT2_OVERLAY:-${AI4S_SHARED_DIR}/models/ORBIT-2/overlays/orbit2-overlay.img}"
LOGDIR="${ORBIT2_SLURM_LOG_DIR:-${AI4S_SHARED_DIR}/models/ORBIT-2/logs}"
mkdir -p "$LOGDIR"

# shellcheck source=../../../../material_science/models/HydraGNN/examples/lux_mpi_env.sh
source "${SCRIPT_DIR}/../../../../material_science/models/HydraGNN/examples/lux_mpi_env.sh"
NODES="${SLURM_JOB_NUM_NODES:-1}"
lux_mpi_setup_binds
lux_mpi_runtime_envs

echo "=== MPI TCP smoke ${SLURM_JOB_ID:-} ==="
echo "  host    : $(hostname)  date: $(date)"
echo "  SIF     : $ORBIT2_SIF"
echo "  overlay : $ORBIT2_OVERLAY"
echo "  OMPI    : $HG_OMPI_DIR"
echo "  iface   : ${HG_LUX_MPI_TCP_IFACE:-${NCCL_SOCKET_IFNAME:-bond0}}"
echo ""

_NODE_TMP="/tmp/orbit2-mpi-smoke-${SLURM_JOB_ID}"
echo "--- Stage SIF+overlay to ${_NODE_TMP} ---"
srun -N 1 --ntasks-per-node=1 --cpus-per-task=4 --gres=none bash -c '
  set -euo pipefail
  dest="'"$_NODE_TMP"'"
  mkdir -p "$dest"
  cp -f "'"$ORBIT2_SIF"'" "$dest/$(basename "'"$ORBIT2_SIF"'")"
  cp -f "'"$ORBIT2_OVERLAY"'" "$dest/orbit2-overlay.img"
  echo "  $(hostname): stage ready"
'
SIF_USE="${_NODE_TMP}/$(basename "$ORBIT2_SIF")"
OV_USE="${_NODE_TMP}/orbit2-overlay.img"

run_step() {
  local ntasks="$1" overlay="$2" cmd="$3" label="$4"
  echo ""
  echo "=== STEP ${label}  ntasks=${ntasks} overlay=${overlay} ==="
  local extra=()
  if [[ "$overlay" == "yes" ]]; then
    extra+=( --overlay "${OV_USE}:ro" )
  fi
  srun --mpi=pmix -N 1 --ntasks="$ntasks" --ntasks-per-node="$ntasks" \
    apptainer exec --rocm \
      "${extra[@]}" \
      "${MPI_BINDS[@]}" \
      --env PMIX_MCA_gds=hash \
      --env PMIX_MCA_psec=native \
      "${LUX_MPI_RUNTIME_ENVS[@]}" \
      --env "HG_OMPI_DIR=${HG_OMPI_DIR}" \
      --env "HG_MPI_EXTRA_LDPATH=${HG_MPI_EXTRA_LDPATH}" \
      --env "MPI4PY_LIBMPI=${MPI4PY_LIBMPI}" \
      --env "HG_LUX_OPAL_PREFIX=${HG_LUX_OPAL_PREFIX}" \
      --env PYTHONUNBUFFERED=1 \
      --env PYTHONPATH="/opt/orbit2-pkgs" \
      "$SIF_USE" \
      bash -lc "$cmd"
  echo "=== STEP ${label} OK ==="
}

run_step 1 no \
  'echo "A $(hostname) procdid=$SLURM_PROCID sif-only"' \
  A-sif-1rank

run_step 1 yes \
  'echo "B $(hostname) procdid=$SLURM_PROCID overlay-ok"' \
  B-overlay-1rank

run_step 1 yes \
  'source /opt/venv/bin/activate
   export PYTHONPATH="/opt/orbit2-pkgs:${PYTHONPATH:-}"
   export LD_LIBRARY_PATH="/opt/orbit2-pkgs:${HG_OMPI_DIR}/lib:${HG_MPI_EXTRA_LDPATH:-}:${LD_LIBRARY_PATH:-}"
   export OPAL_PREFIX="${HG_LUX_OPAL_PREFIX:-$HG_OMPI_DIR}"
   echo "C ldd missing:"; ldd "$MPI4PY_LIBMPI" | grep "not found" || echo "C ldd ok"
   python3 -c "from mpi4py import MPI; print(\"C MPI_Init ok rank\", MPI.COMM_WORLD.Get_rank(), \"size\", MPI.COMM_WORLD.Get_size(), flush=True)"' \
  C-mpi4py-1rank

run_step 2 yes \
  'source /opt/venv/bin/activate
   export PYTHONPATH="/opt/orbit2-pkgs:${PYTHONPATH:-}"
   export LD_LIBRARY_PATH="/opt/orbit2-pkgs:${HG_OMPI_DIR}/lib:${HG_MPI_EXTRA_LDPATH:-}:${LD_LIBRARY_PATH:-}"
   export OPAL_PREFIX="${HG_LUX_OPAL_PREFIX:-$HG_OMPI_DIR}"
   python3 -c "from mpi4py import MPI; print(\"D hello rank\", MPI.COMM_WORLD.Get_rank(), \"size\", MPI.COMM_WORLD.Get_size(), \"host\", __import__(\"socket\").gethostname(), flush=True); MPI.COMM_WORLD.Barrier()"' \
  D-mpi4py-2rank

run_step 8 yes \
  'source /opt/venv/bin/activate
   export PYTHONPATH="/opt/orbit2-pkgs:${PYTHONPATH:-}"
   export LD_LIBRARY_PATH="/opt/orbit2-pkgs:${HG_OMPI_DIR}/lib:${HG_MPI_EXTRA_LDPATH:-}:${LD_LIBRARY_PATH:-}"
   export OPAL_PREFIX="${HG_LUX_OPAL_PREFIX:-$HG_OMPI_DIR}"
   python3 -c "from mpi4py import MPI; print(\"E hello rank\", MPI.COMM_WORLD.Get_rank(), \"size\", MPI.COMM_WORLD.Get_size(), flush=True); MPI.COMM_WORLD.Barrier()"' \
  E-mpi4py-8rank

echo ""
echo "=== all MPI TCP smoke steps passed ==="
