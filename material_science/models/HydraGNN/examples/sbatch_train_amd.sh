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
#      Built via: build_overlay_amd.sh
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
#
# HydraGNN pinned SHA: 6c45f1682783e66dc89e9e23009f61716186432b (main)

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
# Clone HydraGNN source if not present
# ---------------------------------------------------------------------------
HG_HYDRAGNN_SHA="${HG_HYDRAGNN_SHA:-2fb0bd0157e3c85a74f9841887155095bd163303}"
if [[ ! -d "${HG_REPO_DIR}/examples/multidataset_hpo_sc26" ]]; then
  echo "--- Cloning HydraGNN (pinned SHA: ${HG_HYDRAGNN_SHA}) ---"
  git clone https://github.com/ORNL/HydraGNN.git "$HG_REPO_DIR"
  git -C "$HG_REPO_DIR" checkout "$HG_HYDRAGNN_SHA"
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

# ---------------------------------------------------------------------------
# Prepare output directory
# ---------------------------------------------------------------------------
mkdir -p "$HG_OUTPUT_DIR"

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
echo "  Output dir   : $HG_OUTPUT_DIR"
echo "  Repo dir     : $HG_REPO_DIR"
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
export LD_LIBRARY_PATH="/opt/hydragnn-pkgs/adios2:/opt/ompi/lib:${LD_LIBRARY_PATH:-}"

# Use node-local fast storage for per-rank temp writes (MIOpen cache, etc.)
# SCRATCH_LOCAL comes from .cluster-config.yaml paths.scratch_local (e.g. /scratch, /local, /tmp)
SCRATCH_RANK="${SCRATCH_LOCAL:-/scratch}/${USER:?}/hydragnn-${SLURM_JOB_ID:-$$}/${SLURM_PROCID:-0}"
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

cd "$HG_OUTPUT_DIR"

# Monkey-patch AdiosMultiDataset to inject avg_num_neighbors, avoiding an
# expensive full-dataset scan of neighbour counts at init.
export HYDRAGNN_AVG_NUM_NEIGHBORS="${HYDRAGNN_AVG_NUM_NEIGHBORS:-13.735293601560318}"

exec python -u -c "
import sys, os, runpy

avg_nn = float(os.environ.get('HYDRAGNN_AVG_NUM_NEIGHBORS', '0'))
if avg_nn > 0:
    import hydragnn.utils.datasets.adiosdataset as adm
    _orig_init = adm.AdiosMultiDataset.__init__
    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self.avg_num_neighbors = avg_nn
    adm.AdiosMultiDataset.__init__ = _patched_init

batch_size = int(os.environ.get('HG_BATCH_SIZE', '0') or 200)
max_num_batch = int(os.environ.get('HYDRAGNN_MAX_NUM_BATCH', '0'))
num_samples_val = max_num_batch * batch_size if max_num_batch > 0 else 0

sys.argv = [
    os.environ['HG_EXAMPLE_DIR'] + '/gfm_mlip_all_mpnn.py',
    '--inputfile=' + os.environ['HG_EXAMPLE_DIR'] + '/gfm_mlip.json',
    '--multi',
    '--everyone',
    '--multi_model_list=' + os.environ['HG_DATASETS'],
    '--precision=' + os.environ['HG_PRECISION'],
    '--batch_size=' + str(batch_size),
    '--num_epoch=' + os.environ.get('HG_NUM_EPOCH', '1'),
    '--startfrom=none',
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
if [[ "$NODES" -gt 1 ]]; then
  # Read cluster-specific network settings from .cluster-config.yaml if not
  # already set via environment. Requires yq or falls back to grep parsing.
  _CLUSTER_CFG="${SCRIPT_DIR}/../../../../.cluster-config.yaml"
  if [[ -f "$_CLUSTER_CFG" ]]; then
    _yaml_get() { grep "^  $1:" "$_CLUSTER_CFG" 2>/dev/null | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | grep -v '^$'; }
    : "${NCCL_IB_HCA:=$(_yaml_get ib_hca)}"
    : "${NCCL_SOCKET_IFNAME:=$(_yaml_get mgmt_iface)}"
    : "${RCCL_ANP_PLUGIN:=$(_yaml_get rccl_anp_plugin)}"
    : "${LIBIONIC_PATH:=$(_yaml_get libionic_path)}"
  fi

  # Validate required settings
  : "${NCCL_IB_HCA:?Set NCCL_IB_HCA or configure network.ib_hca in .cluster-config.yaml}"
  : "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME or configure network.mgmt_iface in .cluster-config.yaml}"
  : "${RCCL_ANP_PLUGIN:?Set RCCL_ANP_PLUGIN or configure network.rccl_anp_plugin in .cluster-config.yaml}"
  : "${LIBIONIC_PATH:?Set LIBIONIC_PATH or configure network.libionic_path in .cluster-config.yaml}"

  # MPI transport: ob1/tcp over management NIC.
  # Pensando/ionic data NICs use /31 point-to-point subnets that don't route
  # between nodes, so IB verbs cannot work for MPI. Only RCCL (via ANP plugin)
  # bypasses IP routing on the data fabric. MPI uses TCP on the management NIC.
  MPI_MULTINODE_ENVS=(
    --env OMPI_MCA_pml=ob1
    --env OMPI_MCA_btl=tcp,self
    --env OMPI_MCA_btl_tcp_if_include="$NCCL_SOCKET_IFNAME"
    --env MPI4PY_RC_THREADS=false
  )

  # RCCL/NCCL env vars for ANP over ionic (from AMD MI355X documentation).
  # The ANP plugin and libionic must be bind-mounted into the container since
  # Apptainer --rocm does not expose them automatically.
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

srun --mpi=pmix \
    apptainer exec \
    --rocm \
    --overlay "${HG_OVERLAY}:ro" \
    --bind "/opt/ompi:/opt/ompi:ro" \
    --bind "${SCRATCH_LOCAL:-/scratch}:${SCRATCH_LOCAL:-/scratch}" \
    --bind "${HG_REPO_DIR}:${HG_REPO_DIR}:ro" \
    --bind "${HG_DATA_DIR}:${HG_DATA_DIR}:ro" \
    --bind "${HG_OUTPUT_DIR}:${HG_OUTPUT_DIR}" \
    --env PMIX_MCA_gds=hash \
    --env PMIX_MCA_psec=native \
    "${MPI_MULTINODE_ENVS[@]}" \
    --env SCRATCH_LOCAL="${SCRATCH_LOCAL:-/scratch}" \
    --env HG_DATASETS="$HG_DATASETS" \
    --env HG_PRECISION="$HG_PRECISION" \
    --env HG_BATCH_SIZE="${HG_BATCH_SIZE:-200}" \
    --env HG_NUM_EPOCH="${HG_NUM_EPOCH:-1}" \
    --env HYDRAGNN_MAX_NUM_BATCH="${HYDRAGNN_MAX_NUM_BATCH:-}" \
    --env HYDRAGNN_VALTEST="${HYDRAGNN_VALTEST:-0}" \
    --env HYDRAGNN_TRACE_LEVEL="${HYDRAGNN_TRACE_LEVEL:-1}" \
    --env HG_EXAMPLE_DIR="$EXAMPLE_DIR" \
    --env HG_OUTPUT_DIR="$HG_OUTPUT_DIR" \
    "${RCCL_MULTINODE_ENVS[@]}" \
    "$HG_SIF" \
    bash "$RANK_SCRIPT"

echo ""
echo "=== HydraGNN training complete ==="
echo "  Logs: ${HG_OUTPUT_DIR}/"
