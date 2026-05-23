#!/usr/bin/env bash
# HydraGNN 2-node training with PyTorch profiling + Omnistat user-mode telemetry.
#
# This is a thin variant of sbatch_train_amd.sh tailored for the perf-analysis
# recipe at material_science/models/HydraGNN/recipes/perf-analysis/.
#
# Differences from sbatch_train_amd.sh:
#   1. Wraps srun with omnistat-usermode --start/--stopexporters/--stopserver.
#   2. Generates a per-job copy of gfm_mlip.json with a "Profile" block injected
#      so HydraGNN's built-in torch.profiler captures rank-0 of node-0 only.
#   3. Hard-codes the lux partition / vultr_lux account so this script works
#      out-of-the-box on Lux. Override via SBATCH_PARTITION/SBATCH_ACCOUNT if
#      submitting to a different cluster.
#   4. Defaults are tuned for a quick perf run: --nodes=2, NUM_EPOCH=2,
#      MAX_NUM_BATCH=30, time=00:30:00 — enough for the wait=5/warmup=3/active=3
#      profiler schedule plus a clean epoch 0 for warm caches.
#
# Quick start:
#   export AI4S_SHARED_DIR=/shared/aaji
#   sbatch material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh
#
# Required:
#   AI4S_SHARED_DIR — shared base path
#   /shared/aaji/tools/omnistat-pr271/    (created by the launcher subagent)
#   /shared/aaji/tools/victoriametrics/   (created by the launcher subagent)
#
# Optional env-var overrides (with defaults):
#   HG_NUM_EPOCH=2
#   HYDRAGNN_MAX_NUM_BATCH=30
#   HG_PRECISION=fp64
#   HG_BATCH_SIZE=200
#   PROFILE_TARGET_EPOCH=1
#   OMNISTAT_USERMODE_INTERVAL=1   # seconds

#SBATCH --job-name=hydragnn-perf
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
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
HG_NUM_EPOCH="${HG_NUM_EPOCH:-2}"
HG_PRECISION="${HG_PRECISION:-fp64}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-${HG_BASE}/perf-runs/${SLURM_JOB_ID:-$$}}"
HG_REPO_DIR="${HG_REPO_DIR:-${HG_BASE}/code/HydraGNN}"
HYDRAGNN_MAX_NUM_BATCH="${HYDRAGNN_MAX_NUM_BATCH:-30}"
PROFILE_TARGET_EPOCH="${PROFILE_TARGET_EPOCH:-1}"

OMNISTAT_VENV="${OMNISTAT_VENV:-/shared/aaji/tools/omnistat-pr271}"
# Read the omnistat config from the in-repo recipe (single source of truth).
# Override with OMNISTAT_TEMPLATE=/path/to/file if you need a custom probe config.
# A staged copy under ${HG_BASE}/perf-runs/ is intentionally NOT used as the
# default — it has drifted in the past from the in-repo template, silently
# disabling rocprofiler counters. See ai4science-studio SKILL.md for context.
OMNISTAT_TEMPLATE="${OMNISTAT_TEMPLATE:-${SCRIPT_DIR}/../recipes/perf-analysis/omnistat-lux.config.template}"
OMNISTAT_USERMODE_INTERVAL="${OMNISTAT_USERMODE_INTERVAL:-1}"

# Kernel-dispatch tracing (per-kernel name + duration histogram from every
# rank). OFF by default — opt in with OMNISTAT_KERNEL_TRACE=1. When enabled
# we auto-flip enable_kernel_trace=True in the rendered config and load
# libomnistat_trace.so via ROCP_TOOL_LIBRARIES inside the container.
# See material_science/models/HydraGNN/recipes/perf-analysis/agents/launcher.md
# for how to build libomnistat_trace.so on a compute node.
OMNISTAT_KERNEL_TRACE="${OMNISTAT_KERNEL_TRACE:-0}"
OMNISTAT_TRACE_LIB="${OMNISTAT_TRACE_LIB:-/shared/aaji/tools/omnistat-src/build-trace/libomnistat_trace.so}"
# IMPORTANT: the kernel-trace collector registers /kernel_trace on the SAME
# Flask app as the other omnistat endpoints, so the tool library must POST to
# the [omnistat.collectors] `port` (8101 here), NOT the library's own default
# of 8001. Mismatch = silent zero kernel metrics. Auto-derive from the
# template so we can never drift.
_OMNI_PORT=$(awk -F'[= \t]+' '/^[[:space:]]*port[[:space:]]*=/ {print $2; exit}' "$OMNISTAT_TEMPLATE")
OMNISTAT_TRACE_ENDPOINT_PORT="${OMNISTAT_TRACE_ENDPOINT_PORT:-${_OMNI_PORT:-8101}}"

NODES="${SLURM_JOB_NUM_NODES:-2}"
GPUS_PER_NODE=8
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

# ---------------------------------------------------------------------------
# Validate required files
# ---------------------------------------------------------------------------
for var in HG_SIF HG_OVERLAY OMNISTAT_TEMPLATE; do
  if [[ ! -e "${!var}" ]]; then
    echo "ERROR: $var not found: ${!var}" >&2
    exit 2
  fi
done

# Surface the rocprofiler state of the chosen template so a silent
# "counters off" never happens again. Computed here, printed in the banner.
_ROCPROF_STATE=$(awk -F'[= \t]+' '/^enable_rocprofiler/ {print $2; exit}' "$OMNISTAT_TEMPLATE")
_ROCPROF_PROFILE=$(awk -F'[= \t]+' '/^profile/ {print $2; exit}' "$OMNISTAT_TEMPLATE")

if [[ ! -x "${OMNISTAT_VENV}/bin/omnistat-usermode" ]]; then
  echo "ERROR: omnistat-usermode not found at ${OMNISTAT_VENV}/bin/omnistat-usermode" >&2
  echo "       Run the launcher subagent first to install (see recipes/perf-analysis/agents/launcher.md)" >&2
  exit 2
fi

IFS=',' read -ra DATASET_ARRAY <<< "$HG_DATASETS"
for ds in "${DATASET_ARRAY[@]}"; do
  ds_path="${HG_DATA_DIR}/${ds}-v2.bp"
  if [[ ! -d "$ds_path" ]]; then
    echo "ERROR: Dataset not found: $ds_path" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------------------
# Clone HydraGNN source if not present (mirrors sbatch_train_amd.sh)
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
  fi
done

# ---------------------------------------------------------------------------
# Render the per-job omnistat config from the template (substitute @JOB_DIR@)
# This sidesteps configparser's lack of os.environ interpolation.
# ---------------------------------------------------------------------------
mkdir -p "$HG_OUTPUT_DIR"
OMNISTAT_CONFIG="${HG_OUTPUT_DIR}/omnistat.config"
sed -e "s|@JOB_DIR@|${HG_OUTPUT_DIR}|g" "$OMNISTAT_TEMPLATE" > "$OMNISTAT_CONFIG"

# Opt-in kernel tracing: flip enable_kernel_trace=True in the rendered config
# and verify the rocprofiler-sdk tool library exists. Bail out loudly if
# the user asked for it but the .so isn't present — silent fall-through is
# exactly the "rocprofiler counters disabled" failure mode we already burned
# a day on (see SKILL.md §11).
if [[ "$OMNISTAT_KERNEL_TRACE" == "1" ]]; then
  if [[ ! -f "$OMNISTAT_TRACE_LIB" ]]; then
    echo "ERROR: OMNISTAT_KERNEL_TRACE=1 but tool library not found:" >&2
    echo "       $OMNISTAT_TRACE_LIB" >&2
    echo "       Build it on a compute node:" >&2
    echo "         salloc -p lux -A vultr_lux -N 1 --time=00:15:00 --gpus-per-node=1 \\" >&2
    echo "           apptainer exec --rocm \"\$HG_SIF\" bash -c '" >&2
    echo "             cd /shared/aaji/tools/omnistat-src && \\" >&2
    echo "             cmake -S rocprofiler-sdk/ -B build-trace/ -DBUILD_KERNEL_TRACE_LIB=ON && \\" >&2
    echo "             cmake --build build-trace/ -j 8'" >&2
    exit 2
  fi
  sed -i 's/^enable_kernel_trace = False/enable_kernel_trace = True/' "$OMNISTAT_CONFIG"
fi
_KTRACE_STATE=$(awk -F'[= \t]+' '/^enable_kernel_trace/ {print $2; exit}' "$OMNISTAT_CONFIG")

# ---------------------------------------------------------------------------
# Generate the per-job profile config (inject "Profile" block)
# Done on the submit/host side (login or compute), uses python3 if available.
# ---------------------------------------------------------------------------
HG_CONFIG_OVERRIDE="${HG_OUTPUT_DIR}/gfm_mlip_profile.json"
SRC_CONFIG="${EXAMPLE_DIR}/gfm_mlip.json"

python3 - "$SRC_CONFIG" "$HG_CONFIG_OVERRIDE" "$PROFILE_TARGET_EPOCH" <<'PYEOF'
import json, sys
src, dst, epoch = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = json.load(f)
# HydraGNN's train_validate_test() is called with config["NeuralNetwork"];
# its Profiler reads config["Profile"] from THAT scope, so the block must
# go under NeuralNetwork, not at the top level.
cfg.setdefault("NeuralNetwork", {})["Profile"] = {"enable": 1, "target_epoch": epoch}
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote {dst} with NeuralNetwork.Profile.enable=1, target_epoch={epoch}")
PYEOF

# ---------------------------------------------------------------------------
# Print job configuration
# ---------------------------------------------------------------------------
echo "=== HydraGNN Perf Analysis Run ==="
echo "  Nodes        : $NODES"
echo "  GPUs/node    : $GPUS_PER_NODE"
echo "  Total ranks  : $TOTAL_RANKS"
echo "  SIF          : $HG_SIF"
echo "  Overlay      : $HG_OVERLAY"
echo "  Datasets     : $HG_DATASETS"
echo "  Precision    : $HG_PRECISION"
echo "  Batch size   : $HG_BATCH_SIZE"
echo "  Num epochs   : $HG_NUM_EPOCH"
echo "  Max batches  : $HYDRAGNN_MAX_NUM_BATCH"
echo "  Profile epoch: $PROFILE_TARGET_EPOCH (rank-0 only)"
echo "  Output dir   : $HG_OUTPUT_DIR"
echo "  Repo dir     : $HG_REPO_DIR"
echo "  Profile cfg  : $HG_CONFIG_OVERRIDE"
echo "  Omnistat venv: $OMNISTAT_VENV"
echo "  Omnistat tmpl: $OMNISTAT_TEMPLATE"
echo "  Omnistat cfg : $OMNISTAT_CONFIG (rendered)"
echo "  Rocprofiler  : enable_rocprofiler=${_ROCPROF_STATE:-?} profile=${_ROCPROF_PROFILE:-?}"
if [[ "$_ROCPROF_STATE" != "True" ]]; then
  echo "  WARN         : rocprofiler counters DISABLED — HBM/FLOP counters will NOT be collected" >&2
fi
echo "  KernelTrace  : enable_kernel_trace=${_KTRACE_STATE:-?} (opt-in via OMNISTAT_KERNEL_TRACE=1)"
if [[ "$OMNISTAT_KERNEL_TRACE" == "1" ]]; then
  echo "  TraceLib     : $OMNISTAT_TRACE_LIB"
  echo "  TracePort    : $OMNISTAT_TRACE_ENDPOINT_PORT"
fi
echo "  Node(s)      : ${SLURM_NODELIST:-$(hostname)}"
echo "  Date         : $(date)"
echo ""

# ---------------------------------------------------------------------------
# Per-node mount-health probe (lesson from loop-43b33ec1, iter-1, job 7088)
#
# Background: an allocation can land on a node where the per-user autofs/NFS
# mount of /home (or a shared dir under /shared) silently fails. The job then
# wedges in a pmix collective fence until SLURM kills it at the wall, wasting
# the iteration budget. We surface that condition immediately so the caller
# can retry with `sbatch --exclude=<bad-node>` (NEVER blanket-exclude a node
# class — see story.md Lessons #1 in any optimizer-loop dir).
#
# Override probe by setting HG_SKIP_NODE_HEALTH_PROBE=1 (not recommended).
# ---------------------------------------------------------------------------
HG_SKIP_NODE_HEALTH_PROBE="${HG_SKIP_NODE_HEALTH_PROBE:-0}"
if [[ "$HG_SKIP_NODE_HEALTH_PROBE" != "1" ]]; then
  echo "--- Per-node mount-health probe ($(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ',' | sed 's/,$//')) ---"
  _PROBE_OUT="${HG_OUTPUT_DIR}/node_health_probe.txt"
  # One task per node, no-kill so we get reports from healthy nodes even if some fail.
  # Each task reports: hostname, /home/$USER existence, /shared/$USER existence, SIF readability.
  srun --no-kill --kill-on-bad-exit=0 -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
       --cpus-per-task=1 --gres=none --output="${_PROBE_OUT}" --error="${_PROBE_OUT}" \
    bash -c '
      H=$(hostname)
      HM=$(test -d "/home/$USER" 2>/dev/null && echo OK || echo FAIL)
      SH=$(test -d "'"$AI4S_SHARED_DIR"'/$USER" 2>/dev/null || test -d "'"$AI4S_SHARED_DIR"'" 2>/dev/null && echo OK || echo FAIL)
      SIF_CHECK=$(test -f "'"$HG_SIF"'" 2>/dev/null && echo OK || echo FAIL)
      echo "NODE_HEALTH $H home=$HM shared=$SH sif=$SIF_CHECK"
    ' 2>&1 || true
  echo ""
  echo "  Probe results:"
  if [[ -s "$_PROBE_OUT" ]]; then
    grep "^NODE_HEALTH" "$_PROBE_OUT" | sed 's/^/    /'
    _BAD_NODES=$(grep "^NODE_HEALTH" "$_PROBE_OUT" | awk '/home=FAIL|shared=FAIL|sif=FAIL/ {print $2}' | sort -u | tr '\n' ',' | sed 's/,$//')
  else
    echo "    (no output captured; assuming all nodes healthy)"
    _BAD_NODES=""
  fi
  _MISSING_REPORTS=$(( SLURM_JOB_NUM_NODES - $(grep -c "^NODE_HEALTH" "$_PROBE_OUT" 2>/dev/null || echo 0) ))
  if [[ -n "$_BAD_NODES" ]]; then
    echo "" >&2
    echo "FATAL: NODE_HEALTH_PROBE detected broken mounts on: $_BAD_NODES" >&2
    echo "       Retry with:  sbatch --exclude=$_BAD_NODES $0" >&2
    echo "       Do NOT blanket-exclude the node class (a* or b*); only specific bad nodes." >&2
    echo "       Notify cluster admin so the broken node can be repaired." >&2
    exit 42
  fi
  if (( _MISSING_REPORTS > 0 )); then
    echo "WARN: $_MISSING_REPORTS of $SLURM_JOB_NUM_NODES nodes did not return a health-probe result; proceeding anyway" >&2
  fi
  echo "  All $SLURM_JOB_NUM_NODES nodes healthy."
fi

# ---------------------------------------------------------------------------
# Start Omnistat user-mode (datadir from omnistat-lux.config: under HG_OUTPUT_DIR)
# ---------------------------------------------------------------------------
echo "--- Starting Omnistat user-mode (interval=${OMNISTAT_USERMODE_INTERVAL}s) ---"
export PATH="${OMNISTAT_VENV}/bin:${PATH}"
"${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --start --interval "$OMNISTAT_USERMODE_INTERVAL" \
    2>&1 | tee "${HG_OUTPUT_DIR}/omnistat_start.log" || {
  echo "WARN: omnistat-usermode --start returned nonzero; continuing without telemetry" >&2
}

# Make sure we tear down even if the training fails
cleanup_omnistat() {
  echo "--- Stopping Omnistat user-mode ---"
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopexporters || true
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopserver || true
}
trap cleanup_omnistat EXIT

# ---------------------------------------------------------------------------
# Write rank script (runs inside the container on every rank)
# Mirrors sbatch_train_amd.sh but uses HG_CONFIG_OVERRIDE and gates the
# profiler to rank 0 only via PROFILE_RANK0_ONLY.
# ---------------------------------------------------------------------------
RANK_SCRIPT="${HG_OUTPUT_DIR}/hydragnn-rank-${SLURM_JOB_ID:-$$}.sh"
cat > "$RANK_SCRIPT" << 'RANKEOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

# The overlay's /opt/hydragnn-pkgs is rebuilt from HG_HYDRAGNN_SHA + patches/
# (see build_overlay_amd.sh), so we import directly from there. Keep the
# bind-mounted clone available for `examples/multidataset_hpo_sc26/*.py`.
export PYTHONPATH="/opt/hydragnn-pkgs:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/hydragnn-pkgs/adios2:/opt/ompi/lib:${LD_LIBRARY_PATH:-}"

SCRATCH_RANK="${SCRATCH_LOCAL:-/scratch}/${USER:?}/hydragnn-${SLURM_JOB_ID:-$$}/${SLURM_PROCID:-0}"
mkdir -p "$SCRATCH_RANK"
export TMPDIR="$SCRATCH_RANK"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="${SCRATCH_RANK}/miopen"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONNOUSERSITE=1

export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST="${HYDRAGNN_VALTEST:-0}"
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_TRACE_LEVEL="${HYDRAGNN_TRACE_LEVEL:-1}"

export HSA_NO_SCRATCH_RECLAIM=1

cd "$HG_OUTPUT_DIR"
# NOTE: do NOT `import hydragnn` from a separate rank-0-only python here.
# That import triggers `from mpi4py import MPI` which calls MPI_Init() in
# rank 0 only and leaves ranks 1..N-1 hanging in their later MPI_Init()
# because the PMIX rendezvous requires all ranks concurrently. We log the
# imported path inside the main python's first stanza instead.

export HYDRAGNN_AVG_NUM_NEIGHBORS="${HYDRAGNN_AVG_NUM_NEIGHBORS:-13.735293601560318}"

# Profiler is enabled for the whole world (the JSON config already has
# Profile.enable=1) but we want trace files from RANK 0 only so we don't
# stomp on each other in $HG_OUTPUT_DIR/logs/. We disable it in the JSON
# at runtime for non-zero ranks via a tiny monkey-patch in HydraGNN's
# Profiler class.
exec python -u -c "
import sys, os, runpy

# Rank 0 logs which HydraGNN copy is being imported and from what SHA.
# Done inside the main python (after MPI ranks align) — see the comment
# above about why this cannot live in a separate rank-0-only python.
if os.environ.get('SLURM_PROCID', '0') == '0':
    import inspect, hydragnn, subprocess
    pkg_dir = os.path.dirname(hydragnn.__file__)
    src = inspect.getsource(__import__('hydragnn.preprocess.load_data', fromlist=['_']))
    print('[hydragnn] imported from:', pkg_dir, flush=True)
    print('[hydragnn] persistent_workers patch present:', 'HYDRAGNN_PERSISTENT_WORKERS' in src, flush=True)

# Rank 0 keeps the Profile block; others zero it out so HydraGNN's Profiler
# falls through to the null context. Block lives under NeuralNetwork because
# that is the dict train_validate_test() receives.
if os.environ.get('PROFILE_RANK0_ONLY', '0') == '1' and os.environ.get('SLURM_PROCID', '0') != '0':
    import json
    cfg_path = os.environ['HG_CONFIG_OVERRIDE']
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.setdefault('NeuralNetwork', {}).setdefault('Profile', {})['enable'] = 0
    # Write to a per-rank scratch path to avoid races with rank 0
    rank_cfg = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'gfm_mlip_profile_rank.json')
    with open(rank_cfg, 'w') as f:
        json.dump(cfg, f)
    cfg_arg = rank_cfg
else:
    cfg_arg = os.environ['HG_CONFIG_OVERRIDE']

avg_nn = float(os.environ.get('HYDRAGNN_AVG_NUM_NEIGHBORS', '0'))
if avg_nn > 0:
    import hydragnn.utils.datasets.adiosdataset as adm
    _orig_init = adm.AdiosMultiDataset.__init__
    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self.avg_num_neighbors = avg_nn
    adm.AdiosMultiDataset.__init__ = _patched_init

# Optional pre-train hook used by the perf-optimizer-loop recipe to apply a
# rank_script_patch lever (e.g. torch.compile wrapping). The orchestrator writes
# a small python file into the loop dir and exports HG_RANK_PRE_TRAIN_HOOK to
# its absolute path; the hook executes in this rank's interpreter BEFORE
# train_validate_test() runs, so it can monkey-patch hydragnn modules.
# Hook is expected to be small + idempotent; errors are logged on rank 0 but
# never abort the training (so a hook bug doesn't waste a whole 2-node sbatch).
_hook = os.environ.get('HG_RANK_PRE_TRAIN_HOOK', '')
if _hook and os.path.isfile(_hook):
    if os.environ.get('SLURM_PROCID', '0') == '0':
        print(f'[rank_hook] executing pre-train hook: {_hook}', flush=True)
    try:
        with open(_hook) as _hf:
            exec(_hf.read(), {'__name__': '__hg_rank_hook__'})
    except Exception as _e:
        if os.environ.get('SLURM_PROCID', '0') == '0':
            print(f'[rank_hook] FAILED: {type(_e).__name__}: {_e}', flush=True)

batch_size = int(os.environ.get('HG_BATCH_SIZE', '0') or 200)
max_num_batch = int(os.environ.get('HYDRAGNN_MAX_NUM_BATCH', '0'))
num_samples_val = max_num_batch * batch_size if max_num_batch > 0 else 0

sys.argv = [
    os.environ['HG_EXAMPLE_DIR'] + '/gfm_mlip_all_mpnn.py',
    '--inputfile=' + cfg_arg,
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
# Multi-node network environment (read from .cluster-config.yaml)
# Identical to sbatch_train_amd.sh — kept verbatim to avoid drift.
# ---------------------------------------------------------------------------
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
fi

# ---------------------------------------------------------------------------
# Optional: kernel-trace tool library bind + env (loaded by rocprofiler-sdk
# when ROCP_TOOL_LIBRARIES points at the .so). The library POSTs dispatch
# records to the omnistat collector on localhost:OMNISTAT_TRACE_ENDPOINT_PORT.
# ---------------------------------------------------------------------------
KTRACE_BIND_ENVS=()
if [[ "$OMNISTAT_KERNEL_TRACE" == "1" ]]; then
  KTRACE_BIND_ENVS=(
    --bind "${OMNISTAT_TRACE_LIB}:${OMNISTAT_TRACE_LIB}:ro"
    --env ROCP_TOOL_LIBRARIES="${OMNISTAT_TRACE_LIB}"
    --env OMNISTAT_TRACE_ENDPOINT_PORT="${OMNISTAT_TRACE_ENDPOINT_PORT}"
    --env OMNISTAT_TRACE_LOG="${OMNISTAT_TRACE_LOG:-1}"
  )
fi

# ---------------------------------------------------------------------------
# Optional env passthrough — only forward HYDRAGNN_NUM_WORKERS /
# HYDRAGNN_PERSISTENT_WORKERS / HYDRAGNN_MAX_NUM_BATCH when set NON-EMPTY.
# HydraGNN's load_data.py uses `os.getenv(KEY) is not None` to detect the
# var, so an empty string passes the check and then `int("")` blows up.
# Building a separate array keeps `apptainer exec` from receiving stale
# `--env KEY=` flags. (See SKILL §11; observed on probe job 7033.)
#
# The same conditional-forward pattern applies to lever knobs from the
# perf-optimizer-loop recipe (HG_TORCH_COMPILE, TORCH_NCCL_HIGH_PRIORITY,
# PYTORCH_TUNABLEOP_ENABLED, etc.) — most of these are read via plain
# `os.environ.get(K, default)` so an empty string would override the default
# back to "" rather than the desired default. Forward only when non-empty.
# ---------------------------------------------------------------------------
OPT_ENVS=()
for k in \
    HYDRAGNN_NUM_WORKERS HYDRAGNN_PERSISTENT_WORKERS HYDRAGNN_MAX_NUM_BATCH \
    HG_TORCH_COMPILE HG_TORCH_COMPILE_MODE TORCHINDUCTOR_MAX_AUTOTUNE \
    TORCH_NCCL_HIGH_PRIORITY GPU_MAX_HW_QUEUES \
    PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING PYTORCH_TUNABLEOP_VERBOSE \
    NCCL_MIN_NCHANNELS; do
  v="${!k:-}"
  if [[ -n "$v" ]]; then
    OPT_ENVS+=( --env "${k}=${v}" )
  fi
done

# ---------------------------------------------------------------------------
# Optional pre-train hook (perf-optimizer-loop rank_script_patch lever).
# When HG_RANK_PRE_TRAIN_HOOK points at a host-side .py file, bind-mount its
# parent dir read-only and forward the env var. The rank script's main
# python -c block sources this file before train_validate_test().
# ---------------------------------------------------------------------------
HOOK_BIND_ENVS=()
if [[ -n "${HG_RANK_PRE_TRAIN_HOOK:-}" ]]; then
  if [[ -f "$HG_RANK_PRE_TRAIN_HOOK" ]]; then
    _HOOK_DIR=$(cd "$(dirname "$HG_RANK_PRE_TRAIN_HOOK")" && pwd)
    HOOK_BIND_ENVS=(
      --bind "${_HOOK_DIR}:${_HOOK_DIR}:ro"
      --env HG_RANK_PRE_TRAIN_HOOK="$HG_RANK_PRE_TRAIN_HOOK"
    )
    echo "  PreTrainHook : $HG_RANK_PRE_TRAIN_HOOK (bind-mounted ${_HOOK_DIR})"
  else
    echo "WARN: HG_RANK_PRE_TRAIN_HOOK=$HG_RANK_PRE_TRAIN_HOOK is set but file not found; continuing without hook" >&2
  fi
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
    "${KTRACE_BIND_ENVS[@]}" \
    --env PMIX_MCA_gds=hash \
    --env PMIX_MCA_psec=native \
    "${MPI_MULTINODE_ENVS[@]}" \
    --env SCRATCH_LOCAL="${SCRATCH_LOCAL:-/scratch}" \
    --env HG_DATASETS="$HG_DATASETS" \
    --env HG_PRECISION="$HG_PRECISION" \
    --env HG_BATCH_SIZE="${HG_BATCH_SIZE:-200}" \
    --env HG_NUM_EPOCH="${HG_NUM_EPOCH:-1}" \
    --env HYDRAGNN_VALTEST="${HYDRAGNN_VALTEST:-0}" \
    --env HYDRAGNN_TRACE_LEVEL="${HYDRAGNN_TRACE_LEVEL:-1}" \
    --env HG_EXAMPLE_DIR="$EXAMPLE_DIR" \
    --env HG_OUTPUT_DIR="$HG_OUTPUT_DIR" \
    --env HG_REPO_DIR="$HG_REPO_DIR" \
    --env HG_CONFIG_OVERRIDE="$HG_CONFIG_OVERRIDE" \
    "${OPT_ENVS[@]}" \
    "${HOOK_BIND_ENVS[@]}" \
    --env PROFILE_RANK0_ONLY=1 \
    "${RCCL_MULTINODE_ENVS[@]}" \
    "$HG_SIF" \
    bash "$RANK_SCRIPT"

TRAIN_RC=$?

echo ""
echo "=== HydraGNN perf-analysis training complete (rc=$TRAIN_RC) ==="
echo "  Logs: ${HG_OUTPUT_DIR}/"
echo "  Trace dir: ${HG_OUTPUT_DIR}/logs/"
echo "  Omnistat DB: ${HG_OUTPUT_DIR}/omnistat-db/"

exit $TRAIN_RC
