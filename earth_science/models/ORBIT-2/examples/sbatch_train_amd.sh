#!/usr/bin/env bash
# ORBIT-2 multi-node training on AMD Instinct via SLURM (Apptainer + MPI).
#
# Wraps upstream training with Studio launchers and caps (gptl4py stub, batch cap,
# optional rank-0 profiler hook). Prefer Bayes-CAST `launch_diffusion.sh` when present;
# otherwise `run_orbit2_train.py` + `intermediate_downscaling.py`.
# Uses real 10.0_arcmin PRISM data in same-dir mode (see interm_8m_prism.yaml).
#
# Quick start (1 node, 8 GPUs):
#   export AI4S_SHARED_DIR=/path/to/shared
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin
#   export ORBIT2_MAX_EPOCH=1 ORBIT2_MAX_BATCHES=5
#   sbatch earth_science/models/ORBIT-2/examples/sbatch_train_amd.sh
#
# Key environment variables:
#   AI4S_SHARED_DIR      Shared storage root (required)
#   ORBIT2_DATA_ROOT     Data root (PRISM 10.0_arcmin or era5/1.0_deg for sanity)
#   ORBIT2_CONFIG_TEMPLATE  YAML template basename (default: interm_8m_prism.yaml;
#                        use interm_8m_era5.yaml for new ERA5 1.0_deg data)
#   ORBIT2_ROOT          When unset: **bayes-cast** clone if `.../code/bayes-cast` exists, else public ORBIT-2
#   ORBIT2_SIF           Apptainer SIF path
#   ORBIT2_OVERLAY       Pre-built ext3 overlay (required — avoids ~15 min pip/job)
#   ORBIT2_MAX_EPOCH     Cap trainer.max_epochs (default: 3; must be >= 2 —
#                        upstream loop is while (epoch_start+1) < max_epochs)
#   ORBIT2_MAX_BATCHES   Cap batches per epoch (default: 20; 0 = unlimited)
#   ORBIT2_BATCH_SIZE    Per-rank batch size (default: 8)
#   ORBIT2_FSDP / ORBIT2_SIMPLE_DDP  Parallelism for render (optional). When both unset
#                        and nodes=1 with 8 GPUs/job, defaults to fsdp=8, simple_ddp=1.
#   ORBIT2_LAUNCH_SCRIPT  Absolute path to launch script under ORBIT2_ROOT (optional).
#                        If unset, uses launch_diffusion.sh at repo root or examples/ when present.
#   ORBIT2_OUTPUT_DIR    Job output dir (default: .../outputs/train/<jobid>)
#   TORCH_NCCL_HIGH_PRIORITY / GPU_MAX_HW_QUEUES — RCCL tuning (default: 1 / 2)
#   ORBIT2_SKIP_NODE_HEALTH_PROBE=1 — skip mount probe (not recommended)

#SBATCH --job-name=orbit2-train
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:00:00
#SBATCH --output=orbit2-train-%j.out
#SBATCH --error=orbit2-train-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

ORBIT2_BASE="${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}/models/ORBIT-2"
if [[ -z "${ORBIT2_ROOT:-}" ]]; then
  if [[ -d "${ORBIT2_BASE}/code/bayes-cast" ]]; then
    ORBIT2_ROOT="${ORBIT2_BASE}/code/bayes-cast"
  else
    ORBIT2_ROOT="${ORBIT2_BASE}/code/ORBIT-2"
  fi
else
  ORBIT2_ROOT="${ORBIT2_ROOT}"
fi
ORBIT2_SIF="${ORBIT2_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
ORBIT2_OVERLAY="${ORBIT2_OVERLAY:-${ORBIT2_BASE}/overlays/orbit2-overlay.img}"
ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${ORBIT2_BASE}/data/superres/prism/10.0_arcmin}"
ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-3}"
ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES:-20}"
ORBIT2_BATCH_SIZE="${ORBIT2_BATCH_SIZE:-8}"
ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR:-${ORBIT2_BASE}/outputs/train/${SLURM_JOB_ID:-$$}}"

for var in ORBIT2_SIF ORBIT2_OVERLAY ORBIT2_DATA_ROOT; do
  if [[ ! -e "${!var}" ]]; then
    echo "ERROR: $var not found: ${!var}" >&2
    exit 2
  fi
done

TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE=8
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

# Parallelism: explicit env, else 1×8 GPU baseline fsdp=8 simple_ddp=1, else fsdp=nodes × simple_ddp=gpus
if [[ -n "${ORBIT2_FSDP:-}" && -n "${ORBIT2_SIMPLE_DDP:-}" ]]; then
  RENDER_PARALLEL=(--fsdp "$ORBIT2_FSDP" --simple-ddp "$ORBIT2_SIMPLE_DDP")
elif [[ "$NODES" -eq 1 && "$GPUS_PER_NODE" -eq 8 ]]; then
  RENDER_PARALLEL=(--fsdp 8 --simple-ddp 1)
else
  RENDER_PARALLEL=()
fi

# Bayes-CAST ships launch/launch_diffusion.sh as an OLCF/Crusher sbatch+conda script: it
# ignores argv, hardcodes a config, and nests srun — unusable inside Studio Apptainer+srun.
# When launch/train_edm.py exists, call it directly with the rendered /config/config.yaml.
LAUNCH_IC=""
LAUNCH_EDM_DIRECT=0
if [[ -n "${ORBIT2_LAUNCH_SCRIPT:-}" && -f "${ORBIT2_LAUNCH_SCRIPT}" ]]; then
  case "$ORBIT2_LAUNCH_SCRIPT" in
    "$ORBIT2_ROOT"/*) _rel="${ORBIT2_LAUNCH_SCRIPT#"$ORBIT2_ROOT"/}"; LAUNCH_IC="/orbit2/$_rel" ;;
    *) echo "ERROR: ORBIT2_LAUNCH_SCRIPT must be under ORBIT2_ROOT: $ORBIT2_LAUNCH_SCRIPT" >&2; exit 2 ;;
  esac
elif [[ -f "${ORBIT2_ROOT}/launch/train_edm.py" ]]; then
  LAUNCH_EDM_DIRECT=1
elif [[ -f "${ORBIT2_ROOT}/launch_diffusion.sh" ]]; then
  LAUNCH_IC="/orbit2/launch_diffusion.sh"
elif [[ -f "${ORBIT2_ROOT}/launch/launch_diffusion.sh" ]]; then
  LAUNCH_IC="/orbit2/launch/launch_diffusion.sh"
elif [[ -f "${ORBIT2_ROOT}/examples/launch_diffusion.sh" ]]; then
  LAUNCH_IC="/orbit2/examples/launch_diffusion.sh"
fi

_has_upstream=0
[[ "$LAUNCH_EDM_DIRECT" -eq 1 ]] && _has_upstream=1
[[ -n "$LAUNCH_IC" ]] && _has_upstream=1
[[ -f "${ORBIT2_ROOT}/examples/intermediate_downscaling.py" ]] && _has_upstream=1
if [[ "$_has_upstream" -eq 0 ]]; then
  echo "ERROR: ORBIT2_ROOT has no trainable entry (launch/train_edm.py, launch_diffusion.sh, or examples/intermediate_downscaling.py): ${ORBIT2_ROOT}" >&2
  exit 2
fi

mkdir -p "$ORBIT2_OUTPUT_DIR"

ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-interm_8m_prism.yaml}"
CONFIG_TEMPLATE="${SCRIPT_DIR}/${ORBIT2_CONFIG_TEMPLATE}"
if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
  echo "ERROR: ORBIT2_CONFIG_TEMPLATE not found: $CONFIG_TEMPLATE" >&2
  exit 2
fi

JOB_CONFIG="${ORBIT2_OUTPUT_DIR}/${ORBIT2_CONFIG_TEMPLATE%.yaml}_${SLURM_JOB_ID:-$$}.yaml"
python3 "$SCRIPT_DIR/render_orbit2_config.py" \
  --template "$CONFIG_TEMPLATE" \
  --nodes "$NODES" \
  --gpus-per-node "$GPUS_PER_NODE" \
  --data-root "$ORBIT2_DATA_ROOT" \
  --max-epochs "$ORBIT2_MAX_EPOCH" \
  --batch-size "$ORBIT2_BATCH_SIZE" \
  "${RENDER_PARALLEL[@]}" \
  -o "$JOB_CONFIG"

echo "=== ORBIT-2 Training (Apptainer + MPI) ==="
echo "  Nodes        : $NODES"
echo "  GPUs/node    : $GPUS_PER_NODE"
echo "  Total ranks  : $TOTAL_RANKS"
echo "  SIF          : $ORBIT2_SIF"
echo "  Overlay      : $ORBIT2_OVERLAY"
echo "  Data root    : $ORBIT2_DATA_ROOT"
echo "  Max epochs   : $ORBIT2_MAX_EPOCH"
echo "  Max batches  : $ORBIT2_MAX_BATCHES"
echo "  Batch size   : $ORBIT2_BATCH_SIZE"
echo "  Config       : $JOB_CONFIG"
echo "  RCCL priority: TORCH_NCCL_HIGH_PRIORITY=$TORCH_NCCL_HIGH_PRIORITY"
echo "  HW queues    : GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES"
echo "  Output dir   : $ORBIT2_OUTPUT_DIR"
echo "  ORBIT2_ROOT  : $ORBIT2_ROOT"
if ((${#RENDER_PARALLEL[@]})); then
  echo "  Parallelism  : ${RENDER_PARALLEL[*]}"
else
  echo "  Parallelism  : (render default fsdp=$NODES simple_ddp=$GPUS_PER_NODE)"
fi
if [[ "$LAUNCH_EDM_DIRECT" -eq 1 ]]; then
  echo "  Train entry  : python3 /orbit2/launch/train_edm.py (Bayes EDM; Studio bypasses launch_diffusion.sh)"
elif [[ -n "$LAUNCH_IC" ]]; then
  echo "  Train entry  : ${LAUNCH_IC}"
else
  echo "  Train entry  : run_orbit2_train.py (studio)"
fi
echo "  Node(s)      : ${SLURM_NODELIST:-$(hostname)}"
echo "  Date         : $(date)"
echo ""

# ---------------------------------------------------------------------------
# Per-node mount-health probe (exit 42 → retry with --exclude=<bad-node>)
# ---------------------------------------------------------------------------
ORBIT2_SKIP_NODE_HEALTH_PROBE="${ORBIT2_SKIP_NODE_HEALTH_PROBE:-0}"
if [[ "$ORBIT2_SKIP_NODE_HEALTH_PROBE" != "1" ]]; then
  echo "--- Per-node mount-health probe ---"
  _PROBE_OUT="${ORBIT2_OUTPUT_DIR}/node_health_probe.txt"
  srun --no-kill --kill-on-bad-exit=0 -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
       --cpus-per-task=1 --gres=none --output="${_PROBE_OUT}" --error="${_PROBE_OUT}" \
    bash -c '
      H=$(hostname)
      HM=$(test -d "/home/$USER" 2>/dev/null && echo OK || echo FAIL)
      SH=$(test -d "'"$AI4S_SHARED_DIR"'" 2>/dev/null && echo OK || echo FAIL)
      SIF_CHECK=$(test -f "'"$ORBIT2_SIF"'" 2>/dev/null && echo OK || echo FAIL)
      DATA_CHECK=$(test -d "'"$ORBIT2_DATA_ROOT"'" 2>/dev/null && echo OK || echo FAIL)
      echo "NODE_HEALTH $H home=$HM shared=$SH sif=$SIF_CHECK data=$DATA_CHECK"
    ' 2>&1 || true
  if [[ -s "$_PROBE_OUT" ]]; then
    grep "^NODE_HEALTH" "$_PROBE_OUT" | sed 's/^/    /'
    _BAD_NODES=$(grep "^NODE_HEALTH" "$_PROBE_OUT" | awk '/home=FAIL|shared=FAIL|sif=FAIL|data=FAIL/ {print $2}' | sort -u | tr '\n' ',' | sed 's/,$//')
  else
    _BAD_NODES=""
  fi
  if [[ -n "$_BAD_NODES" ]]; then
    echo "FATAL: NODE_HEALTH_PROBE detected broken mounts on: $_BAD_NODES" >&2
    echo "       Retry with: sbatch --exclude=$_BAD_NODES ..." >&2
    exit 42
  fi
  echo "  All $SLURM_JOB_NUM_NODES nodes healthy."
fi

LAUNCH_DIR=""
if [[ "$LAUNCH_EDM_DIRECT" -eq 1 ]]; then
  LAUNCH_DIR="/orbit2/launch"
elif [[ -n "$LAUNCH_IC" ]]; then
  LAUNCH_DIR=$(dirname "$LAUNCH_IC")
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT="${ORBIT2_MASTER_PORT:-29500}"

RANK_SCRIPT="${ORBIT2_OUTPUT_DIR}/orbit2_rank_${SLURM_JOB_ID:-$$}.sh"
cat > "$RANK_SCRIPT" << RANKEOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate
echo "[rank \$SLURM_PROCID / \$SLURM_NTASKS] GPU \$SLURM_LOCALID on \$(hostname)"
export PYTHONPATH="/opt/orbit2-pkgs:/orbit2/src:/orbit2:\${PYTHONPATH:-}"
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-7}"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="\${TMPDIR:-/tmp}/orbit2-miopen-\${SLURM_JOB_ID:-\$\$}-\${SLURM_PROCID:-0}"
mkdir -p "\$MIOPEN_USER_DB_PATH"
# ORNL Frontier-validated MIOpen conv flags (bayes-cast launch/launch_diffusion.sh). ORNL DISABLES
# Winograd and unbounds the multi-pass Winograd workspace; tested many times on Frontier (gfx90a /
# ROCm 7.1.1). Defaults below replicate ORNL exactly; override at submit to A/B on MI355X (gfx950 /
# ROCm 7.2.2), e.g. ORBIT2_MIOPEN_CONV_WINOGRAD=1 to re-enable Winograd.
export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX="${ORBIT2_MIOPEN_WINOGRAD_MPASS_WS_MAX:--1}"
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX="${ORBIT2_MIOPEN_MP_BD_WINOGRAD_WS_MAX:--1}"
export MIOPEN_DEBUG_CONV_WINOGRAD="${ORBIT2_MIOPEN_CONV_WINOGRAD:-0}"
export PYTHONNOUSERSITE=1
export HSA_NO_SCRATCH_RECLAIM=1
# ORNL Frontier-validated (launch_diffusion.sh): fine-grain PCIe coherence. Generic ROCm, portable.
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export ORBIT_USE_DDSTORE=0
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES}"
export ORBIT2_ROOT="/orbit2"
export ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES}"
export ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR}"
# Pass through checkpoint toggle. Default 0 here (general training script keeps checkpoints);
# throughput/scaling callers set ORBIT2_DISABLE_CKPT=1 to skip all checkpoint writes.
export ORBIT2_DISABLE_CKPT="${ORBIT2_DISABLE_CKPT:-0}"
# Writable inside the container (job dir is bind-mounted) for CK debug NDJSON.
export DEBUG_AGENT_LOG="\${DEBUG_AGENT_LOG:-${ORBIT2_OUTPUT_DIR}/agent-ck.ndjson}"
export ORBIT2_DATA_TYPE="${ORBIT2_DATA_TYPE:-bfloat16}"
export ORBIT2_FUSED_ATTN="${ORBIT2_FUSED_ATTN:-DEFAULT}"
export ORBIT2_RANK_PRE_TRAIN_HOOK="\${ORBIT2_RANK_PRE_TRAIN_HOOK:-}"
if [[ "${LAUNCH_EDM_DIRECT}" -eq 1 ]]; then
  cd /orbit2/launch
  python3 /examples/orbit2_rank_hook_runner.py
  # run_orbit2_train_edm.py wraps train_edm.main() to honour ORBIT2_MAX_BATCHES
  # (train_edm.py runs its batch loop inline, so there is no function to patch).
  exec python3 /examples/run_orbit2_train_edm.py /config/config.yaml
elif [[ -n "${LAUNCH_IC}" ]]; then
  cd /orbit2/examples
  python3 /examples/orbit2_rank_hook_runner.py
  cd "${LAUNCH_DIR}"
  exec bash "${LAUNCH_IC}" /config/config.yaml
else
  cd /orbit2/examples
  exec python3 /examples/run_orbit2_train.py /config/config.yaml
fi
RANKEOF
chmod +x "$RANK_SCRIPT"

CONFIG_BIND="$(realpath "$JOB_CONFIG"):/config/config.yaml"

# Multi-node RCCL/MPI (from .cluster-config.yaml)
RCCL_MULTINODE_ENVS=()
MPI_MULTINODE_ENVS=()
if [[ "$NODES" -gt 1 ]]; then
  _CLUSTER_CFG="${SCRIPT_DIR}/../../../../.cluster-config.yaml"
  if [[ -f "$_CLUSTER_CFG" ]]; then
    _yaml_get() { grep "^  $1:" "$_CLUSTER_CFG" 2>/dev/null | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | grep -v '^$'; }
    : "${NCCL_IB_HCA:=$(_yaml_get ib_hca)}"
    : "${NCCL_SOCKET_IFNAME:=$(_yaml_get mgmt_iface)}"
    : "${RCCL_ANP_PLUGIN:=$(_yaml_get rccl_anp_plugin)}"
    : "${LIBIONIC_PATH:=$(_yaml_get libionic_path)}"
  fi
  : "${NCCL_IB_HCA:?Set NCCL_IB_HCA or configure network.ib_hca in .cluster-config.yaml}"
  : "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME or configure network.mgmt_iface}"
  : "${RCCL_ANP_PLUGIN:?Set RCCL_ANP_PLUGIN or configure network.rccl_anp_plugin}"
  : "${LIBIONIC_PATH:?Set LIBIONIC_PATH or configure network.libionic_path}"

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
  echo "  Multi-node RCCL enabled (NCCL_IB_HCA=$NCCL_IB_HCA)"
fi

echo "--- Launching training: $TOTAL_RANKS ranks across $NODES nodes ---"
srun --mpi=pmix apptainer exec \
    --rocm \
    --overlay "${ORBIT2_OVERLAY}:ro" \
    --bind "/opt/ompi:/opt/ompi:ro" \
    --bind "$ORBIT2_ROOT":/orbit2 \
    --bind "$SCRIPT_DIR":/examples \
    --bind "$(dirname "$RANK_SCRIPT"):$(dirname "$RANK_SCRIPT")" \
    --bind "$ORBIT2_DATA_ROOT":"$ORBIT2_DATA_ROOT":ro \
    --bind "$ORBIT2_OUTPUT_DIR":"$ORBIT2_OUTPUT_DIR" \
    --bind "$CONFIG_BIND" \
    --env HOSTNAME="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env PMIX_MCA_gds=hash \
    --env PMIX_MCA_psec=native \
    --env PYTHONPATH="/opt/orbit2-pkgs:/orbit2/src:/orbit2" \
    --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
    "${MPI_MULTINODE_ENVS[@]}" \
    "${RCCL_MULTINODE_ENVS[@]}" \
    "$ORBIT2_SIF" \
    bash "$RANK_SCRIPT"

echo ""
echo "=== ORBIT-2 training complete ==="
echo "  Output: $ORBIT2_OUTPUT_DIR"
