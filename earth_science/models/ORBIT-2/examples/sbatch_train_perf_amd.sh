#!/usr/bin/env bash
# ORBIT-2 training with PyTorch profiling + Omnistat user-mode telemetry.
#
# Perf-analysis variant of sbatch_train_amd.sh. See recipes/perf-analysis/.
# Default allocation is **1 node × 8 GPUs**; use `sbatch --nodes=2 ...` for multi-node.
#
# Quick start (ERA5 same-dir):
#   export AI4S_SHARED_DIR=/path/to/shared
#   export OMNIHUB_TOOLS_DIR=/shared/omnihub/tools
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
#   export ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml
#   export ORBIT2_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/code/bayes-cast   # EDM preset + launcher
#   sbatch earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh

#SBATCH --job-name=orbit2-perf
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:30:00
#SBATCH --output=orbit2-train-%j.out
#SBATCH --error=orbit2-train-%j.out

set -euo pipefail

ORBIT2_PERF_START_TS=$(date +%s)

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
# Default ERA5 1.0° same-dir (111×111 latent) for heavier forwards / GPU saturation baselines.
# PRISM 10.0_arcmin (18×18): export ORBIT2_DATA_ROOT=$ORBIT2_BASE/data/superres/prism/10.0_arcmin
#   and ORBIT2_CONFIG_TEMPLATE=interm_8m_prism.yaml
ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${ORBIT2_BASE}/data/superres/era5/1.0_deg}"
# Default 6 so steady_batch_time_s (epoch >= 2) has a multi-epoch window; see perf-analysis/README.md
ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-6}"
ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES:-20}"
ORBIT2_BATCH_SIZE="${ORBIT2_BATCH_SIZE:-4}"
ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR:-${ORBIT2_BASE}/perf-runs/${SLURM_JOB_ID:-$$}}"
PROFILE_TARGET_EPOCH="${PROFILE_TARGET_EPOCH:-0}"
# These are throughput/speed runs ONLY — do NOT persist model checkpoints. The Bayes-CAST EDM
# trainer otherwise writes a ~71 MB interm_epoch_*.ckpt every epoch to launch/checkpoints/bayes-cast/
# (save_checkpoint(), train_edm.py:1390). ORBIT2_DISABLE_CKPT=1 makes it skip every write.
# Override with ORBIT2_DISABLE_CKPT=0 only if you actually need a checkpoint from a run.
ORBIT2_DISABLE_CKPT="${ORBIT2_DISABLE_CKPT:-1}"

# TunableOp (PyTorch GEMM autotuning) lever. Default OFF — baseline byte-identical.
#   off  : disabled (PYTORCH_TUNABLEOP_ENABLED=0)
#   tune : live-tune GEMMs and WRITE per-rank caches to ORBIT2_TUNABLEOP_DIR (ENABLED=1 TUNING=1)
#   use  : load pre-tuned caches, no further tuning (ENABLED=1 TUNING=0)
# Isolated 1-GPU probe (jobs 10595/10596) proved tuning is NOT blocked on this stack; the
# remaining unknown is the full 8-rank FSDP path — that is exactly what a `tune` run validates.
# Cache dir is a stable shared path so a later `use` run can read a prior `tune` run's caches.
ORBIT2_TUNABLEOP_MODE="${ORBIT2_TUNABLEOP_MODE:-off}"
ORBIT2_TUNABLEOP_DIR="${ORBIT2_TUNABLEOP_DIR:-${ORBIT2_BASE}/tunableop-cache}"

: "${OMNIHUB_TOOLS_DIR:?OMNIHUB_TOOLS_DIR must be set}"
OMNISTAT_VENV="${OMNISTAT_VENV:-${OMNIHUB_TOOLS_DIR}/omnihub-inspect}"
OMNISTAT_TEMPLATE="${OMNISTAT_TEMPLATE:-${SCRIPT_DIR}/../recipes/perf-analysis/omnistat.config.template}"
OMNISTAT_USERMODE_INTERVAL="${OMNISTAT_USERMODE_INTERVAL:-1}"

TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE=8
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

if [[ -n "${ORBIT2_FSDP:-}" && -n "${ORBIT2_SIMPLE_DDP:-}" ]]; then
  RENDER_PARALLEL=(--fsdp "$ORBIT2_FSDP" --simple-ddp "$ORBIT2_SIMPLE_DDP")
elif [[ "$NODES" -eq 1 && "$GPUS_PER_NODE" -eq 8 ]]; then
  RENDER_PARALLEL=(--fsdp 8 --simple-ddp 1)
else
  RENDER_PARALLEL=()
fi

# Bayes-CAST launch/launch_diffusion.sh is an OLCF/Crusher wrapper (nested srun, ignores argv).
# Prefer launch/train_edm.py + rendered /config/config.yaml when present.
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

LAUNCH_DIR=""
if [[ "$LAUNCH_EDM_DIRECT" -eq 1 ]]; then
  LAUNCH_DIR="/orbit2/launch"
elif [[ -n "$LAUNCH_IC" ]]; then
  LAUNCH_DIR=$(dirname "$LAUNCH_IC")
fi

for var in ORBIT2_SIF ORBIT2_OVERLAY ORBIT2_DATA_ROOT OMNISTAT_TEMPLATE; do
  if [[ ! -e "${!var}" ]]; then
    echo "ERROR: $var not found: ${!var}" >&2
    exit 2
  fi
done
if [[ ! -x "${OMNISTAT_VENV}/bin/omnistat-usermode" ]]; then
  echo "ERROR: omnistat-usermode not found at ${OMNISTAT_VENV}/bin/" >&2
  exit 2
fi

mkdir -p "$ORBIT2_OUTPUT_DIR"
OMNISTAT_CONFIG="${ORBIT2_OUTPUT_DIR}/omnistat.config"
sed -e "s|@JOB_DIR@|${ORBIT2_OUTPUT_DIR}|g" \
    -e "s|@OMNIHUB_TOOLS_DIR@|${OMNIHUB_TOOLS_DIR}|g" \
    "$OMNISTAT_TEMPLATE" > "$OMNISTAT_CONFIG"
if [[ -n "${OMNISTAT_ROCPROF_PROFILE:-}" ]]; then
  sed -i.bak "s/^profile = .*/profile = ${OMNISTAT_ROCPROF_PROFILE}/" "$OMNISTAT_CONFIG" && rm -f "${OMNISTAT_CONFIG}.bak"
fi

ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-edm_8m_era5_1x8.yaml}"
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

echo "=== ORBIT-2 Perf Analysis Run ==="
echo "  Nodes        : $NODES"
echo "  Total ranks  : $TOTAL_RANKS"
echo "  ORBIT2_ROOT  : $ORBIT2_ROOT"
if ((${#RENDER_PARALLEL[@]})); then
  echo "  Parallelism  : ${RENDER_PARALLEL[*]}"
else
  echo "  Parallelism  : (render default fsdp=$NODES simple_ddp=$GPUS_PER_NODE)"
fi
if [[ "$LAUNCH_EDM_DIRECT" -eq 1 ]]; then
  echo "  Train entry  : python3 /orbit2/launch/train_edm.py (Bayes EDM)"
elif [[ -n "$LAUNCH_IC" ]]; then
  echo "  Train entry  : ${LAUNCH_IC}"
else
  echo "  Train entry  : run_orbit2_train.py (studio)"
fi
echo "  Config       : $JOB_CONFIG"
echo "  Output dir   : $ORBIT2_OUTPUT_DIR"
echo "  Profile epoch: $PROFILE_TARGET_EPOCH"
echo "  Omnistat cfg : $OMNISTAT_CONFIG"
echo "  Node(s)      : ${SLURM_NODELIST:-$(hostname)}"
echo ""

# Mount-health probe (same as sbatch_train_amd.sh)
ORBIT2_SKIP_NODE_HEALTH_PROBE="${ORBIT2_SKIP_NODE_HEALTH_PROBE:-0}"
if [[ "$ORBIT2_SKIP_NODE_HEALTH_PROBE" != "1" ]]; then
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
  _BAD_NODES=$(grep "^NODE_HEALTH" "$_PROBE_OUT" 2>/dev/null | awk '/home=FAIL|shared=FAIL|sif=FAIL|data=FAIL/ {print $2}' | sort -u | tr '\n' ',' | sed 's/,$//' || true)
  if [[ -n "$_BAD_NODES" ]]; then
    echo "FATAL: NODE_HEALTH_PROBE failed on: $_BAD_NODES" >&2
    exit 42
  fi
fi

export PATH="${OMNISTAT_VENV}/bin:${PATH}"
echo "--- Starting Omnistat user-mode ---"
"${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --start --interval "$OMNISTAT_USERMODE_INTERVAL" \
    2>&1 | tee "${ORBIT2_OUTPUT_DIR}/omnistat_start.log" || true

cleanup_omnistat() {
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopexporters || true
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopserver || true
}
trap cleanup_omnistat EXIT

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT="${ORBIT2_MASTER_PORT:-29500}"
TRACE_DIR="${ORBIT2_OUTPUT_DIR}/traces"
# Bayes EDM: default no hook (res_slimvit profiler imports break); allow ORBIT2_RANK_PRE_TRAIN_HOOK for sysopt patches.
if [[ "${LAUNCH_EDM_DIRECT}" -eq 1 ]]; then
  _RANK_PRE_TRAIN_HOOK="${ORBIT2_RANK_PRE_TRAIN_HOOK:-}"
else
  _RANK_PRE_TRAIN_HOOK="${ORBIT2_RANK_PRE_TRAIN_HOOK:-/examples/orbit2_profiler_hook.py}"
fi

# Resolve TunableOp mode → env flags (consumed inside the rank script).
#   record : log every GEMM shape to an UNTUNED file WITHOUT tuning (free; for offline workflow)
_TUNABLEOP_RECORD=0
case "$ORBIT2_TUNABLEOP_MODE" in
  tune)   _TUNABLEOP_ENABLED=1; _TUNABLEOP_TUNING=1 ;;
  use)    _TUNABLEOP_ENABLED=1; _TUNABLEOP_TUNING=0 ;;
  record) _TUNABLEOP_ENABLED=1; _TUNABLEOP_TUNING=0; _TUNABLEOP_RECORD=1 ;;
  off)    _TUNABLEOP_ENABLED=0; _TUNABLEOP_TUNING=0 ;;
  *) echo "ERROR: ORBIT2_TUNABLEOP_MODE must be off|tune|use|record (got: $ORBIT2_TUNABLEOP_MODE)" >&2; exit 2 ;;
esac
if [[ "$ORBIT2_TUNABLEOP_MODE" != "off" ]]; then
  mkdir -p "$ORBIT2_TUNABLEOP_DIR"
  echo "  TunableOp    : mode=$ORBIT2_TUNABLEOP_MODE dir=$ORBIT2_TUNABLEOP_DIR (ENABLED=$_TUNABLEOP_ENABLED TUNING=$_TUNABLEOP_TUNING)"
fi

RANK_SCRIPT="${ORBIT2_OUTPUT_DIR}/orbit2_rank_${SLURM_JOB_ID:-$$}.sh"
cat > "$RANK_SCRIPT" << RANKEOF
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate
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
# Perf experiment: widen path2/refine internal conv channels to a tile-friendly N (>=4) to A/B the
# starved low-channel conv GEMM (edm.py reads this). 0 = original architecture.
export ORBIT2_CONV_PAD="${ORBIT2_CONV_PAD:-0}"
# Perf experiment: run path2/refine convs in NHWC (channels_last) so MIOpen uses implicit-GEMM and
# skips the im2col buffer. 1 = on. FIND_MODE=1 (normal search) lets MIOpen pick the NHWC kernel.
export ORBIT2_CHANNELS_LAST="${ORBIT2_CHANNELS_LAST:-0}"
[[ "\${ORBIT2_CHANNELS_LAST}" == "1" ]] && export MIOPEN_FIND_MODE="\${MIOPEN_FIND_MODE:-1}"
# Default bfloat16 for production-like perf; use float32 if xformers.ops / CK path is unstable.
export ORBIT2_DATA_TYPE="${ORBIT2_DATA_TYPE:-bfloat16}"
export ORBIT2_FUSED_ATTN="${ORBIT2_FUSED_ATTN:-DEFAULT}"
# torch.compile lever (perf-optimizer-loop): in-process patch in train_edm.py.
export ORBIT2_TORCH_COMPILE="${ORBIT2_TORCH_COMPILE:-0}"
export ORBIT2_COMPILE_MODE="${ORBIT2_COMPILE_MODE:-default}"
export ORBIT2_COMPILE_DYNAMIC="${ORBIT2_COMPILE_DYNAMIC:-}"
# BLAS backend lever: bf16 large-M GEMMs (var_agg F.linear) throw hipErrorInvalidValue
# under hipBLASLt on MI355X+ROCm7.2.2; TORCH_BLAS_PREFER_HIPBLASLT=0 falls back to rocBLAS.
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"
# TunableOp GEMM autotuning (default off). Per-rank cache file (each rank owns 1 GPU; all see
# device 0 in-process, so key on SLURM_PROCID to avoid 8 ranks clobbering one results0.csv).
export PYTORCH_TUNABLEOP_ENABLED=${_TUNABLEOP_ENABLED}
export PYTORCH_TUNABLEOP_TUNING=${_TUNABLEOP_TUNING}
# Filename: 'use' reads the offline tuner's merged per-device cache (read-only → all ranks share
# the device file, torch appends the device id to '..._dev' → _dev0.csv.._dev7.csv). tune/record
# WRITE, so key on SLURM_PROCID to stop 8 ranks clobbering one file.
if [[ "${ORBIT2_TUNABLEOP_MODE}" == "use" ]]; then
  export PYTORCH_TUNABLEOP_FILENAME="${ORBIT2_TUNABLEOP_DIR}/tunableop_results_dev.csv"
else
  export PYTORCH_TUNABLEOP_FILENAME="${ORBIT2_TUNABLEOP_DIR}/tunableop_results_rank\${SLURM_PROCID:-0}.csv"
fi
export PYTORCH_TUNABLEOP_VERBOSE="${PYTORCH_TUNABLEOP_VERBOSE:-1}"
# Bounded tuning — ORBIT-2's patch-embed/var_agg GEMMs have HUGE M (e.g. tn_256_5308416_256,
# M=5.3M rows). The default rotating buffer (-1 → sized to defeat L2) allocates/cycles multi-GB
# input copies per candidate → a single op can take many minutes. ROTATING_BUFFER_SIZE=0 disables
# rotation (reuse one buffer); capped iterations keep per-op tuning to seconds. Override at submit.
export PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE="${PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE:-0}"
export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS="${PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS:-10}"
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS="${PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS:-30}"
# Offline "record" workflow: log every untuned GEMM shape to a file WITHOUT tuning it live, so a
# separate offline tuner (mgpu_tune_gemm_in_file) can tune a SIZE-FILTERED subset (skip giant ops).
export PYTORCH_TUNABLEOP_RECORD_UNTUNED=${_TUNABLEOP_RECORD}
export PYTORCH_TUNABLEOP_UNTUNED_FILENAME="${ORBIT2_TUNABLEOP_DIR}/tunableop_untuned_rank\${SLURM_PROCID:-0}.csv"
# Optional debug passthroughs (empty unless set in submit env). Synchronous kernel
# execution pins the true failing kernel for async HIP errors.
export AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-}"
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-}"
export ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR}"
# Throughput runs do not persist checkpoints (save_checkpoint() early-returns on this).
export ORBIT2_DISABLE_CKPT="${ORBIT2_DISABLE_CKPT}"
# Writable inside the container (perf run dir bind-mounted) for CK debug NDJSON.
export DEBUG_AGENT_LOG="\${DEBUG_AGENT_LOG:-${ORBIT2_OUTPUT_DIR}/agent-ck.ndjson}"
export ORBIT2_PROFILE_DIR="${TRACE_DIR}"
export PROFILE_TARGET_EPOCH="${PROFILE_TARGET_EPOCH}"
export PROFILE_RANK0_ONLY=1
export ORBIT2_RANK_PRE_TRAIN_HOOK="${_RANK_PRE_TRAIN_HOOK}"
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
  : "${NCCL_IB_HCA:?NCCL_IB_HCA required for multi-node}"
  : "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME required}"
  : "${RCCL_ANP_PLUGIN:?RCCL_ANP_PLUGIN required}"
  : "${LIBIONIC_PATH:?LIBIONIC_PATH required}"
  MPI_MULTINODE_ENVS=(
    --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=tcp,self
    --env OMPI_MCA_btl_tcp_if_include="$NCCL_SOCKET_IFNAME"
    --env MPI4PY_RC_THREADS=false
  )
  # GID-INDEX CAVEAT (MI355X RoCE): NCCL_IB_GID_INDEX=1 only works when every node
  # exposes the fabric ULA (RoCEv2) at GID index 1. On some clusters a subset of nodes also
  # carries a *global* IPv6 at idx1, which shifts the fabric ULA to idx2 → cross-node
  # localGid/remoteGid mismatch → "ionic_comp cqe error 12 / status=12 RETRY_EXC" abort.
  # Pin jobs to nodes with consistent GID tables via --exclude/--nodelist, or override
  # NCCL_IB_GID_INDEX. Enumerate: cat /sys/class/infiniband/<hca>/ports/1/gids/{1,2}.
  # See run_2node_scaleout_loop.sh (O2_EXCLUDE_PATTERN) for auto-excluding a known-bad range.
  RCCL_MULTINODE_ENVS=(
    --bind "${RCCL_ANP_PLUGIN}:${RCCL_ANP_PLUGIN}:ro"
    --bind "${LIBIONIC_PATH}:${LIBIONIC_PATH}:ro"
    --env NCCL_NET_PLUGIN="$RCCL_ANP_PLUGIN"
    --env NCCL_IB_HCA="$NCCL_IB_HCA"
    --env NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
    --env NCCL_GDR_FLUSH_DISABLE=1
    --env RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
    --env NCCL_GDRCOPY_ENABLE=0
    --env NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-1}"
    --env HSA_NO_SCRATCH_RECLAIM=1
    --env NCCL_IB_TC=96
    --env NCCL_IB_FIFO_TC=192
    --env NCCL_IGNORE_CPU_AFFINITY=1
    --env NCCL_PXN_DISABLE=0
    --env NET_OPTIONAL_RECV_COMPLETION=1
    --env NCCL_IB_USE_INLINE=1
    --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
    --env RCCL_LL128_FORCE_ENABLE="${RCCL_LL128_FORCE_ENABLE:-1}"
    --env NCCL_IB_PCI_RELAXED_ORDERING=1
    --env NCCL_DMABUF_ENABLE=1
    --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  )
  # Optional comm-tuning levers (perf-optimizer-loop): only injected when set at submit time,
  # so the baseline keeps upstream defaults. These are real multi-node levers (FSDP all-gather
  # over IB is the new cost at N>1) — see recipes/perf-optimizer-loop/agents/lever_catalog.yaml.
  [[ -n "${NCCL_MIN_NCHANNELS:-}" ]] && RCCL_MULTINODE_ENVS+=(--env NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS")
  [[ -n "${NCCL_NCHANNELS_PER_PEER:-}" ]] && RCCL_MULTINODE_ENVS+=(--env NCCL_NCHANNELS_PER_PEER="$NCCL_NCHANNELS_PER_PEER")
  echo "  Multi-node RCCL enabled (NCCL_IB_HCA=$NCCL_IB_HCA, QPS=${NCCL_IB_QPS_PER_CONNECTION:-1}, LL128=${RCCL_LL128_FORCE_ENABLE:-1}, MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-default})"
fi

# Bind the TunableOp cache dir rw so `tune` runs can persist caches and `use` runs can read them
# across jobs (it lives outside the per-job OUTPUT_DIR).
TUNABLEOP_BIND=()
if [[ "$ORBIT2_TUNABLEOP_MODE" != "off" ]]; then
  TUNABLEOP_BIND=(--bind "$ORBIT2_TUNABLEOP_DIR":"$ORBIT2_TUNABLEOP_DIR")
fi

echo "--- Launching perf training: $TOTAL_RANKS ranks ---"
set +e
srun --mpi=pmix apptainer exec \
    --rocm --overlay "${ORBIT2_OVERLAY}:ro" \
    --bind "/opt/ompi:/opt/ompi:ro" \
    "${TUNABLEOP_BIND[@]}" \
    --bind "$ORBIT2_ROOT":/orbit2 \
    --bind "$SCRIPT_DIR":/examples \
    --bind "$(dirname "$RANK_SCRIPT"):$(dirname "$RANK_SCRIPT")" \
    --bind "$ORBIT2_DATA_ROOT":"$ORBIT2_DATA_ROOT":ro \
    --bind "$ORBIT2_OUTPUT_DIR":"$ORBIT2_OUTPUT_DIR" \
    --bind "$CONFIG_BIND" \
    --env HOSTNAME="$MASTER_ADDR" --env MASTER_PORT="$MASTER_PORT" \
    --env PMIX_MCA_gds=hash --env PMIX_MCA_psec=native \
    --env PYTHONPATH="/opt/orbit2-pkgs:/orbit2/src:/orbit2" \
    --env LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/rocm/lib:/usr/lib/x86_64-linux-gnu" \
    "${MPI_MULTINODE_ENVS[@]}" "${RCCL_MULTINODE_ENVS[@]}" \
    "$ORBIT2_SIF" bash "$RANK_SCRIPT"
TRAIN_RC=$?
set -e

_SLURM_LOG="${ORBIT2_SLURM_LOG:-${SLURM_SUBMIT_DIR:-.}/orbit2-train-${SLURM_JOB_ID}.out}"
if [[ ! -f "$_SLURM_LOG" ]]; then
  _SLURM_LOG="${AI4S_SHARED_DIR}/models/ORBIT-2/outputs/train/logs/orbit2-train-${SLURM_JOB_ID}.out"
fi
cp -f "$_SLURM_LOG" "${ORBIT2_OUTPUT_DIR}/orbit2-train-${SLURM_JOB_ID}.out" 2>/dev/null || true

M_FSDP="$NODES"
M_SIMPLE="$GPUS_PER_NODE"
if ((${#RENDER_PARALLEL[@]} == 4)); then
  M_FSDP="${RENDER_PARALLEL[1]}"
  M_SIMPLE="${RENDER_PARALLEL[3]}"
fi

RUNTIME_S=$(( $(date +%s) - ORBIT2_PERF_START_TS ))
WORKLOAD_TAG="intermediate_downscaling"
if [[ "${LAUNCH_EDM_DIRECT:-0}" -eq 1 ]]; then
  WORKLOAD_TAG="bayes_edm"
fi

python3 - "$ORBIT2_OUTPUT_DIR" "${SLURM_JOB_ID}" "$TRAIN_RC" "$ORBIT2_ROOT" "$JOB_CONFIG" "$M_FSDP" "$M_SIMPLE" \
  "$RUNTIME_S" "$ORBIT2_BATCH_SIZE" "$TOTAL_RANKS" "$ORBIT2_MAX_EPOCH" "${ORBIT2_DATA_TYPE:-bfloat16}" "$WORKLOAD_TAG" <<'PYEOF'
import json, subprocess
import sys
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


job_dir = Path(sys.argv[1])
job_id, train_rc = sys.argv[2], int(sys.argv[3])
orbit2_root = Path(sys.argv[4])
job_config = Path(sys.argv[5])
m_fsdp, m_simple = int(sys.argv[6]), int(sys.argv[7])
runtime_s = int(sys.argv[8])
batch = int(sys.argv[9])
total_ranks = int(sys.argv[10])
max_epochs = int(sys.argv[11])
data_type = sys.argv[12]
workload = sys.argv[13]
global_batch = batch * m_fsdp * m_simple

manifest = {
    "job_id": job_id,
    "model": "ORBIT-2",
    "workload": workload,
    "job_dir": str(job_dir),
    "slurm_log": str(job_dir / f"orbit2-train-{job_id}.out"),
    "omnistat_config": str(job_dir / "omnistat.config"),
    "omnistat_db": str(job_dir / "omnistat-db"),
    "trace_dir": str(job_dir / "traces"),
    "state": "complete" if train_rc == 0 else "failed",
    "exit_code": train_rc,
    "orbit2_root": str(orbit2_root),
    "rendered_config": str(job_config),
    "parallelism": {"fsdp": m_fsdp, "simple_ddp": m_simple},
    "orbit2_batch_size": batch,
    "total_ranks": total_ranks,
    "max_epochs": max_epochs,
    "data_type": data_type,
    "global_batch_size": global_batch,
    "runtime_seconds": runtime_s,
    "git_sha": _git(orbit2_root, "rev-parse", "HEAD"),
    "git_branch": _git(orbit2_root, "rev-parse", "--abbrev-ref", "HEAD"),
    "git_remote_origin": _git(orbit2_root, "remote", "get-url", "origin"),
}
(job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote {job_dir / 'manifest.json'}")
PYEOF

echo "=== ORBIT-2 perf run complete (rc=$TRAIN_RC) ==="
exit $TRAIN_RC
