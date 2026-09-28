#!/usr/bin/env bash
# HydraGNN multi-dataset training on AMD Instinct MI355X via SLURM (Apptainer).
#
# Emulates the upstream HydraGNN-scaling-test.sh workflow:
#   https://github.com/ORNL/HydraGNN/blob/main/run-scripts/HydraGNN-scaling-test.sh
#
# Uses the repo-bundled gfm_mlip.json config with ANI1x and Alexandria ADIOS
# datasets via MPI-enabled AdiosMultiDataset (no DDStore).
#
# Prerequisites:
#   1. Apptainer SIF with PyTorch ROCm (default: rocm7.2.2 + py3.12)
#   2. Pre-built overlay with hydragnn, torch-geometric, mpi4py, adios2 (MPI)
#      Built via: sbatch sbatch_build_overlay_amd.sh  (ORNL Lux, compute node)
#                 or build_overlay_amd.sh             (Vultr Lux, /opt/ompi)
#   3. ADIOS datasets staged on shared filesystem
#
# Quick-start (1 node, 8 GPUs):
#   export AI4S_SHARED_DIR=/path/to/shared   # set via /init-cluster
#   sbatch sbatch_train_amd.sh
#
# Multi-node (4 nodes, 32 GPUs):
#   export AI4S_SHARED_DIR=/path/to/shared
#   sbatch --nodes=4 sbatch_train_amd.sh
#
# Key environment variables:
#   AI4S_SHARED_DIR   Base path for shared assets (required, no default)
#   HG_SIF            Path to Apptainer SIF image
#   HG_OVERLAY        Path to pre-built ext3 overlay
#   HG_DATASETS       Comma-separated dataset names (default: ANI1x,Alexandria)
#   HG_DATA_DIR       Directory containing <dataset>-v2.bp dirs
#   HG_BATCH_SIZE     Per-rank batch size (default: from config)
#   HG_NUM_EPOCH      Number of training epochs (default: 1)
#   HG_PRECISION      fp32 | fp64 | bf16 (default: fp64)
#   HG_OUTPUT_DIR     Working directory for logs/checkpoints
#   HG_REPO_DIR       HydraGNN source clone location
#   SCRATCH_LOCAL     Node-local fast storage root (default: /scratch)
#   HYDRAGNN_MAX_NUM_BATCH  Cap batches per epoch (sanity test: 20-50, full: unset)
#   HYDRAGNN_VALTEST  Run val/test during training (default: 0 = skip)
#   HG_STARTFROM      Resume from logs/<name>/<name>.pk (default: none)
#   HG_WARM_CKPT      Host path to .pk copied into logs/<HG_STARTFROM>/ before launch
#   TORCH_NCCL_HIGH_PRIORITY  Default 1 (RCCL high-priority queue)
#   GPU_MAX_HW_QUEUES         Default 2
#
# HydraGNN pinned SHA: 2fb0bd0157e3c85a74f9841887155095bd163303 (main)

#SBATCH --job-name=hydragnn-train
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=hydragnn-train-%j.out
#SBATCH --error=hydragnn-train-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# SCRIPT_DIR — works both at submit time and inside the SLURM spool
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# shellcheck source=lux_mpi_env.sh
source "${SCRIPT_DIR}/lux_mpi_env.sh"
HG_USE_SYSTEM_MPI=0

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
HG_BASE="${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}/models/HydraGNN"
HG_SIF="${HG_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
HG_OVERLAY="${HG_OVERLAY:-${HG_BASE}/overlays/hydragnn-overlay.img}"
HG_DATASETS="${HG_DATASETS:-ANI1x,Alexandria}"
HG_DATA_DIR="${HG_DATA_DIR:-${HG_BASE}/weights}"
HG_BATCH_SIZE="${HG_BATCH_SIZE:-200}"
HG_NUM_EPOCH="${HG_NUM_EPOCH:-1}"
HG_PRECISION="${HG_PRECISION:-fp64}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-${HG_BASE}/outputs}"
HG_REPO_DIR="${HG_REPO_DIR:-${HG_BASE}/code/HydraGNN}"

# Validated on MI355X perf-optimizer-loop: ~1.3% epoch-time improvement on 2-node GFM-MLIP.
TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE=8
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

# ---------------------------------------------------------------------------
# Validate required files
# ---------------------------------------------------------------------------
for var in HG_SIF HG_OVERLAY; do
  if [[ ! -f "${!var}" ]]; then
    echo "ERROR: $var not found: ${!var}" >&2
    exit 2
  fi
done

IFS=',' read -ra DATASET_ARRAY <<< "$HG_DATASETS"
for ds in "${DATASET_ARRAY[@]}"; do
  ds_path="${HG_DATA_DIR}/${ds}-v2.bp"
  if [[ ! -d "$ds_path" ]]; then
    echo "ERROR: Dataset not found: $ds_path" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------------------
# OpenMPI bind-mount (overlay adios2 links against host MPI)
#   Vultr Lux: /opt/ompi
#   ORNL Lux: system openmpi/5.0.10 module over ob1/TCP
# ---------------------------------------------------------------------------
MPI_BINDS=()
MPI4PY_LIBMPI=""
HG_MPI_EXTRA_LDPATH=""
if [[ -z "${HG_OMPI_DIR:-}" ]] && lux_mpi_setup; then
  HG_USE_SYSTEM_MPI=1
  MPI_BINDS=( "${MPI_BINDS[@]}" )
elif [[ -z "${HG_OMPI_DIR:-}" ]]; then
  if [[ -x /opt/ompi/bin/mpicc ]]; then
    HG_OMPI_DIR=/opt/ompi
    MPI_BINDS=( --bind "${HG_OMPI_DIR}:${HG_OMPI_DIR}:ro" )
  else
    if command -v module &>/dev/null; then
      module load openmpi/5.0.10 2>/dev/null || module load openmpi 2>/dev/null || true
    fi
    if command -v mpicc &>/dev/null; then
      HG_OMPI_DIR="$(cd "$(dirname "$(dirname "$(command -v mpicc)")")" && pwd)"
      MPI_BINDS=( --bind "${HG_OMPI_DIR}:${HG_OMPI_DIR}:ro" )
      MPI4PY_LIBMPI="${HG_OMPI_DIR}/lib/libmpi.so.40"
      _ucc_lib="$(ldd "${HG_OMPI_DIR}/lib/libmpi.so.40" 2>/dev/null | awk '/libucc\.so/ {print $3; exit}')" || true
      _ucx_lib=""
      [[ -n "${_ucc_lib}" ]] && _ucx_lib="$(ldd "${_ucc_lib}" 2>/dev/null | awk '/libucs\.so/ {print $3; exit}')" || true
      HG_UCC_LIB="${HG_UCC_LIB:-$(dirname "${_ucc_lib}" 2>/dev/null)}"
      HG_UCX_LIB="${HG_UCX_LIB:-$(dirname "${_ucx_lib}" 2>/dev/null)}"
      if [[ -d "${HG_UCC_LIB}" && -d "${HG_UCX_LIB}" ]]; then
        HG_MPI_EXTRA_LDPATH="${HG_UCC_LIB}:${HG_UCX_LIB}:/lib64"
        MPI_BINDS+=( --bind "${HG_UCC_LIB}:${HG_UCC_LIB}:ro" --bind "${HG_UCX_LIB}:${HG_UCX_LIB}:ro" )
        for _lib in /lib64/liblnetconfig.so.4 /lib64/liblustreapi.so.1 /lib64/libnl-genl-3.so.200; do
          [[ -f "$_lib" ]] && MPI_BINDS+=( --bind "${_lib}:${_lib}:ro" )
        done
      fi
    fi
  fi
fi
if [[ -z "${HG_OMPI_DIR:-}" || ${#MPI_BINDS[@]} -eq 0 ]]; then
  echo "ERROR: OpenMPI not found for ADIOS MPI. On ORNL Lux load the system OpenMPI module." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Clone HydraGNN source if not present (prefer pre-clone on login node; compute nodes have no egress)
HG_HYDRAGNN_SHA="${HG_HYDRAGNN_SHA:-2fb0bd0157e3c85a74f9841887155095bd163303}"
# Orion world-shared ADIOS stores (Alexandria-v3, legacy ANI1x) lack graph_attr.
export HG_DISABLE_GRAPH_ATTR="${HG_DISABLE_GRAPH_ATTR:-1}"
if [[ ! -d "${HG_REPO_DIR}/examples/multidataset_hpo_sc26" ]]; then
  echo "--- Cloning HydraGNN (pinned SHA: ${HG_HYDRAGNN_SHA}) ---"
  PROXY_ENVS=()
  for k in https_proxy http_proxy HTTPS_PROXY HTTP_PROXY no_proxy NO_PROXY; do
    v="${!k:-}"
    [[ -n "$v" ]] && PROXY_ENVS+=( --env "${k}=${v}" )
  done
  if [[ -n "${HG_SIF:-}" && -f "${HG_SIF:-}" ]]; then
    apptainer exec "${PROXY_ENVS[@]}" --bind "$(dirname "$HG_REPO_DIR")" \
      "$HG_SIF" bash -c "git clone https://github.com/ORNL/HydraGNN.git '$HG_REPO_DIR' && git -C '$HG_REPO_DIR' checkout '$HG_HYDRAGNN_SHA'"
  else
    git clone https://github.com/ORNL/HydraGNN.git "$HG_REPO_DIR"
    git -C "$HG_REPO_DIR" checkout "$HG_HYDRAGNN_SHA"
  fi
fi

# ---------------------------------------------------------------------------
# Symlink datasets into the expected location
# ---------------------------------------------------------------------------
EXAMPLE_DIR="${HG_REPO_DIR}/examples/multidataset_hpo_sc26"
DATASET_DIR="${EXAMPLE_DIR}/dataset"
mkdir -p "$DATASET_DIR"

for ds in "${DATASET_ARRAY[@]}"; do
  target="${DATASET_DIR}/${ds}-v2.bp"
  source="${HG_DATA_DIR}/${ds}-v2.bp"
  if [[ ! -e "$target" ]]; then
    ln -sfn "$source" "$target"
    echo "  Linked: $target -> $source"
  fi
done

# Bind external world-shared trees so Apptainer can follow weights/ symlinks.
# shellcheck source=hg_dataset_binds.sh
source "${SCRIPT_DIR}/hg_dataset_binds.sh"
hg_setup_dataset_binds

# Multi-node: expose host RDMA/RoCE sysfs to UCX inside Apptainer.
LUX_MPI_MULTINODE_BINDS=()
LUX_RCCL_MULTINODE_ENVS=()
if [[ "$NODES" -gt 1 && "${HG_USE_SYSTEM_MPI}" == 1 ]]; then
  lux_mpi_multinode_apptainer_binds
  lux_mpi_rccl_multinode_envs
  for _bind in "${LUX_MPI_MULTINODE_BINDS[@]}"; do
    echo "  Multi-node bind: ${_bind}"
  done
fi

# ---------------------------------------------------------------------------
# Prepare output directory
# ---------------------------------------------------------------------------
mkdir -p "$HG_OUTPUT_DIR"

# Optional warm checkpoint for scaling resume (HG_STARTFROM + HG_WARM_CKPT).
if [[ -n "${HG_STARTFROM:-}" && "${HG_STARTFROM}" != "none" && -n "${HG_WARM_CKPT:-}" ]]; then
  _WARM_DIR="${HG_OUTPUT_DIR}/logs/${HG_STARTFROM}"
  mkdir -p "$_WARM_DIR"
  if [[ ! -f "${_WARM_DIR}/${HG_STARTFROM}.pk" ]]; then
    cp -a "$HG_WARM_CKPT" "${_WARM_DIR}/${HG_STARTFROM}.pk"
    echo "  Warm ckpt  : $HG_WARM_CKPT -> ${_WARM_DIR}/${HG_STARTFROM}.pk"
  fi
fi

# ---------------------------------------------------------------------------
# Print job configuration
# ---------------------------------------------------------------------------
echo "=== HydraGNN Training (MPI-enabled ADIOS, no DDStore) ==="
echo "  Nodes        : $NODES"
echo "  GPUs/node    : $GPUS_PER_NODE"
echo "  Total ranks  : $TOTAL_RANKS"
echo "  SIF          : $HG_SIF"
echo "  Overlay      : $HG_OVERLAY"
echo "  Datasets     : $HG_DATASETS"
echo "  Data dir     : $HG_DATA_DIR"
echo "  Precision    : $HG_PRECISION"
echo "  Batch size   : ${HG_BATCH_SIZE:-200}"
echo "  Num epochs   : ${HG_NUM_EPOCH:-1}"
echo "  Max batches  : ${HYDRAGNN_MAX_NUM_BATCH:-unlimited}"
echo "  RCCL priority: TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY}"
echo "  HW queues    : GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES}"
echo "  Start from   : ${HG_STARTFROM:-none}"
echo "  Output dir   : $HG_OUTPUT_DIR"
echo "  Repo dir     : $HG_REPO_DIR"
echo "  OpenMPI      : $HG_OMPI_DIR"
[[ -n "${HG_UCC_LIB:-}" ]] && echo "  UCC lib      : $HG_UCC_LIB"
[[ -n "${HG_UCX_LIB:-}" ]] && echo "  UCX lib      : $HG_UCX_LIB"
echo "  Config       : ${EXAMPLE_DIR}/gfm_mlip.json"
echo "  Node(s)      : ${SLURM_NODELIST:-$(hostname)}"
echo "  Date         : $(date)"
echo ""

# ---------------------------------------------------------------------------
# Write rank script (runs inside the container on every rank)
# ---------------------------------------------------------------------------
RANK_SCRIPT="${HG_OUTPUT_DIR}/hydragnn-rank-${SLURM_JOB_ID:-$$}.sh"
cat > "$RANK_SCRIPT" << 'RANKEOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

export PYTHONPATH="/opt/hydragnn-pkgs:${PYTHONPATH:-}"
OMPI="${HG_OMPI_DIR:-/opt/ompi}"
if [[ -n "${HG_MPI_EXTRA_LDPATH:-}" ]]; then
  export LD_LIBRARY_PATH="/opt/hydragnn-pkgs/adios2:${OMPI}/lib:${HG_MPI_EXTRA_LDPATH}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="/opt/hydragnn-pkgs/adios2:${OMPI}/lib:${LD_LIBRARY_PATH:-}"
fi
[[ -n "${MPI4PY_LIBMPI:-}" ]] && export MPI4PY_LIBMPI
[[ -n "${HG_LUX_OPAL_PREFIX:-}" ]] && export OPAL_PREFIX="${HG_LUX_OPAL_PREFIX}"

# Use node-local fast storage for per-rank temp writes (MIOpen cache, etc.)
# SCRATCH_LOCAL comes from .cluster-config.yaml paths.scratch_local (e.g. /scratch, /local, /tmp)
SCRATCH_RANK="${SCRATCH_LOCAL:-/tmp}/${USER:?}/hydragnn-${SLURM_JOB_ID:-$$}/${SLURM_PROCID:-0}"
mkdir -p "$SCRATCH_RANK"
export TMPDIR="$SCRATCH_RANK"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="${SCRATCH_RANK}/miopen"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONNOUSERSITE=1

# HydraGNN runtime env vars (from upstream HydraGNN-scaling-test.sh)
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST="${HYDRAGNN_VALTEST:-0}"
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_TRACE_LEVEL="${HYDRAGNN_TRACE_LEVEL:-1}"

# Required for RCCL stability on MI355X
export HSA_NO_SCRATCH_RECLAIM=1

# Multi-node: pin one GPU per rank. MPI uses ob1/TCP on bond0; RCCL selects
# ionic HCAs independently through NCCL_IB_HCA.
if [[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]]; then
  _local_rank="${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}"
  export ROCR_VISIBLE_DEVICES="${_local_rank}"
fi

cd "$HG_OUTPUT_DIR"

# Monkey-patch AdiosMultiDataset to inject avg_num_neighbors, avoiding an
# expensive full-dataset scan of neighbour counts at init.
export HYDRAGNN_AVG_NUM_NEIGHBORS="${HYDRAGNN_AVG_NUM_NEIGHBORS:-13.735293601560318}"

exec python -u -c "
import sys, os, json, runpy

avg_nn = float(os.environ.get('HYDRAGNN_AVG_NUM_NEIGHBORS', '0'))
import hydragnn.utils.datasets.adiosdataset as adm
_orig_setkeys = adm.AdiosDataset.setkeys
def _patched_setkeys(self, keys):
    filtered = [k for k in keys if k in self.keys]
    missing = [k for k in keys if k not in self.keys]
    if missing and self.comm.Get_rank() == 0:
        print('  ADIOS keys skipped (not in file):', missing, flush=True)
    return _orig_setkeys(self, filtered)
adm.AdiosDataset.setkeys = _patched_setkeys
if avg_nn > 0:
    _orig_init = adm.AdiosMultiDataset.__init__
    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self.avg_num_neighbors = avg_nn
    adm.AdiosMultiDataset.__init__ = _patched_init

cfg_src = os.environ['HG_EXAMPLE_DIR'] + '/gfm_mlip.json'
cfg_dst = os.path.join(os.environ['HG_OUTPUT_DIR'], 'gfm_mlip_runtime.json')
with open(cfg_src) as f:
    cfg = json.load(f)
if os.environ.get('HG_DISABLE_GRAPH_ATTR', '0') == '1':
    cfg['NeuralNetwork']['Architecture']['use_graph_attr_conditioning'] = False
with open(cfg_dst, 'w') as f:
    json.dump(cfg, f)

batch_size = int(os.environ.get('HG_BATCH_SIZE', '0') or 200)
max_num_batch = int(os.environ.get('HYDRAGNN_MAX_NUM_BATCH', '0'))
num_samples_val = max_num_batch * batch_size if max_num_batch > 0 else 0

sys.argv = [
    os.environ['HG_EXAMPLE_DIR'] + '/gfm_mlip_all_mpnn.py',
    '--inputfile=' + cfg_dst,
    '--multi',
    '--everyone',
    '--multi_model_list=' + os.environ['HG_DATASETS'],
    '--precision=' + os.environ['HG_PRECISION'],
    '--batch_size=' + str(batch_size),
    '--num_epoch=' + os.environ.get('HG_NUM_EPOCH', '1'),
    '--startfrom=' + os.environ.get('HG_STARTFROM', 'none'),
    '--log=hydragnn-train-' + os.environ.get('SLURM_JOB_ID','0') + '-N' + os.environ.get('SLURM_JOB_NUM_NODES','1'),
]

if num_samples_val > 0:
    sys.argv += [
        '--num_samples=' + str(num_samples_val),
        '--oversampling',
        '--oversampling_num_samples=' + str(num_samples_val),
    ]

runpy.run_path(sys.argv[0], run_name='__main__')
"
RANKEOF
chmod +x "$RANK_SCRIPT"

# ---------------------------------------------------------------------------
# Multi-node network environment (required for >1 node)
#
# All cluster-specific values are read from .cluster-config.yaml (gitignored)
# or overridden via env vars:
#   NCCL_IB_HCA        — IB HCA device list (discover: ibstat | grep 'CA ')
#   NCCL_SOCKET_IFNAME — Management NIC for OOB (discover: ip -o link show up)
#   RCCL_ANP_PLUGIN    — Path to librccl-anp.so on host
#   LIBIONIC_PATH      — Path to libionic.so.1 on host
# ---------------------------------------------------------------------------
RCCL_MULTINODE_ENVS=()
MPI_MULTINODE_ENVS=()
if [[ "${HG_USE_SYSTEM_MPI}" == 1 ]]; then
  if lux_mpi_runtime_envs; then
    MPI_MULTINODE_ENVS=( "${LUX_MPI_RUNTIME_ENVS[@]}" )
    if [[ "$NODES" -gt 1 ]]; then
      echo "  Multi-node MPI: system openmpi/5.0.10 (ob1/tcp)"
    fi
  else
    echo "WARNING: system OpenMPI module unavailable — MPI tuning not applied" >&2
  fi
elif [[ "$NODES" -gt 1 ]]; then
  # Vultr Lux: ob1/tcp + RCCL ANP over ionic fabric
  _CLUSTER_CFG="${SCRIPT_DIR}/../../../../.cluster-config.yaml"
  if [[ -f "$_CLUSTER_CFG" ]]; then
    _yaml_get() { grep "^  $1:" "$_CLUSTER_CFG" 2>/dev/null | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | grep -v '^$'; }
    : "${NCCL_IB_HCA:=$(_yaml_get ib_hca)}"
    : "${NCCL_SOCKET_IFNAME:=$(_yaml_get mgmt_iface)}"
    : "${RCCL_ANP_PLUGIN:=$(_yaml_get rccl_anp_plugin)}"
    : "${LIBIONIC_PATH:=$(_yaml_get libionic_path)}"
  fi

  : "${NCCL_IB_HCA:?Set NCCL_IB_HCA or configure network.ib_hca in .cluster-config.yaml}"
  : "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME or configure network.mgmt_iface in .cluster-config.yaml}"
  : "${RCCL_ANP_PLUGIN:?Set RCCL_ANP_PLUGIN or configure network.rccl_anp_plugin in .cluster-config.yaml}"
  : "${LIBIONIC_PATH:?Set LIBIONIC_PATH or configure network.libionic_path in .cluster-config.yaml}"

  MPI_MULTINODE_ENVS=(
    --env OMPI_MCA_pml=ob1
    --env OMPI_MCA_btl=tcp,self
    --env OMPI_MCA_btl_tcp_if_include="$NCCL_SOCKET_IFNAME"
    --env MPI4PY_RC_THREADS=false
  )

  RCCL_MULTINODE_ENVS=(
    --bind "${RCCL_ANP_PLUGIN}:${RCCL_ANP_PLUGIN}:ro"
    --bind "${LIBIONIC_PATH}:${LIBIONIC_PATH}:ro"
    --env NCCL_NET_PLUGIN="$RCCL_ANP_PLUGIN"
    --env NCCL_IB_HCA="$NCCL_IB_HCA"
    --env NCCL_IB_GID_INDEX=1
    --env NCCL_GDR_FLUSH_DISABLE=1
    --env RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
    --env NCCL_GDRCOPY_ENABLE=0
    --env NCCL_IB_QPS_PER_CONNECTION=1
    --env HSA_NO_SCRATCH_RECLAIM=1
    --env NCCL_IB_TC=96
    --env NCCL_IB_FIFO_TC=192
    --env NCCL_IGNORE_CPU_AFFINITY=1
    --env NCCL_PXN_DISABLE=0
    --env NET_OPTIONAL_RECV_COMPLETION=1
    --env NCCL_IB_USE_INLINE=1
    --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
    --env RCCL_LL128_FORCE_ENABLE=1
    --env NCCL_IB_PCI_RELAXED_ORDERING=1
    --env NCCL_DMABUF_ENABLE=1
    --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  )
  echo "  Multi-node MPI transport: ob1/tcp (btl_tcp_if_include=$NCCL_SOCKET_IFNAME)"
  echo "  Multi-node RCCL env vars enabled"
  echo "    NCCL_IB_HCA=$NCCL_IB_HCA"
  echo "    NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
fi

# ---------------------------------------------------------------------------
# Launch distributed training via srun + apptainer
# ---------------------------------------------------------------------------
echo "--- Launching training: $TOTAL_RANKS ranks across $NODES nodes ---"
echo ""

# Optional env passthrough — HydraGNN's training code uses
# `os.getenv(KEY) is not None` to detect these knobs, so an empty string
# passes the check and then `int("")` blows up. Build a separate array of
# `--env` flags that are only added when the var is set non-empty.
OPT_ENVS=()
for k in HYDRAGNN_MAX_NUM_BATCH HYDRAGNN_NUM_WORKERS HYDRAGNN_PERSISTENT_WORKERS MPI4PY_LIBMPI HG_MPI_EXTRA_LDPATH HG_OMPI_DIR HG_LUX_OPAL_PREFIX OPAL_PREFIX NCCL_DEBUG_SUBSYS NCCL_TIMEOUT NCCL_ASYNC_ERROR_HANDLING TORCH_DISTRIBUTED_DEBUG; do
  v="${!k:-}"
  if [[ -n "$v" ]]; then
    OPT_ENVS+=( --env "${k}=${v}" )
  fi
done

# Stage ext3 overlay to node-local /tmp once per node. Concurrent fuse2fs mounts of
# the same Lustre .img from NGPU ranks starve/deadlock Apptainer startup on ORNL Lux.
HG_OVERLAY_USE="$HG_OVERLAY"
if [[ -n "${HG_OVERLAY:-}" && -f "${HG_OVERLAY}" && "${HG_STAGE_OVERLAY_LOCAL:-1}" != "0" ]]; then
  _LOCAL_OV="/tmp/hydragnn-overlay-${SLURM_JOB_ID}.img"
  echo "--- Staging overlay to node-local ${_LOCAL_OV} (1 task/node) ---"
  srun -N "$NODES" --ntasks-per-node=1 --cpus-per-task=2 --gres=none \
    bash -c '
      src="'"$HG_OVERLAY"'"
      dst="'"$_LOCAL_OV"'"
      if [[ ! -s "$dst" ]]; then
        echo "  $(hostname): cp -> $dst"
        cp -f "$src" "$dst"
      fi
      echo "  $(hostname): ready $(du -h "$dst" 2>/dev/null | cut -f1)"
    '
  HG_OVERLAY_USE="$_LOCAL_OV"
fi

srun --mpi=pmix \
    apptainer exec \
    --rocm \
    --overlay "${HG_OVERLAY_USE}:ro" \
    "${MPI_BINDS[@]}" \
    "${LUX_MPI_MULTINODE_BINDS[@]}" \
    --bind "${SCRATCH_LOCAL:-/tmp}:${SCRATCH_LOCAL:-/tmp}" \
    --bind "${HG_REPO_DIR}:${HG_REPO_DIR}:ro" \
    --bind "${HG_DATA_DIR}:${HG_DATA_DIR}:ro" \
    "${DATA_BINDS[@]}" \
    --bind "${HG_OUTPUT_DIR}:${HG_OUTPUT_DIR}" \
    --env PMIX_MCA_gds=hash \
    --env PMIX_MCA_psec=native \
    "${MPI_MULTINODE_ENVS[@]}" \
    --env SCRATCH_LOCAL="${SCRATCH_LOCAL:-/tmp}" \
    --env HG_DATASETS="$HG_DATASETS" \
    --env HG_PRECISION="$HG_PRECISION" \
    --env HG_BATCH_SIZE="${HG_BATCH_SIZE:-200}" \
    --env HG_NUM_EPOCH="${HG_NUM_EPOCH:-1}" \
    --env HG_STARTFROM="${HG_STARTFROM:-none}" \
    --env HYDRAGNN_VALTEST="${HYDRAGNN_VALTEST:-0}" \
    --env HYDRAGNN_TRACE_LEVEL="${HYDRAGNN_TRACE_LEVEL:-1}" \
    --env HG_EXAMPLE_DIR="$EXAMPLE_DIR" \
    --env HG_OUTPUT_DIR="$HG_OUTPUT_DIR" \
    --env TORCH_NCCL_HIGH_PRIORITY="$TORCH_NCCL_HIGH_PRIORITY" \
    --env GPU_MAX_HW_QUEUES="$GPU_MAX_HW_QUEUES" \
    "${OPT_ENVS[@]}" \
    "${RCCL_MULTINODE_ENVS[@]}" \
    "${LUX_RCCL_MULTINODE_ENVS[@]}" \
    "$HG_SIF" \
    bash "$RANK_SCRIPT"

echo ""
echo "=== HydraGNN training complete ==="
echo "  Logs: ${HG_OUTPUT_DIR}/"
