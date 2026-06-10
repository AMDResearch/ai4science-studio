#!/usr/bin/env bash
# ORBIT-2 2-node training with PyTorch profiling + Omnistat user-mode telemetry.
#
# Perf-analysis variant of sbatch_train_amd.sh. See recipes/perf-analysis/.
#
# Quick start:
#   export AI4S_SHARED_DIR=/path/to/shared
#   export OMNIHUB_TOOLS_DIR=/path/to/omnihub/tools
#   export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin
#   sbatch earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh

#SBATCH --job-name=orbit2-perf
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:30:00
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
ORBIT2_ROOT="${ORBIT2_ROOT:-${ORBIT2_BASE}/code/ORBIT-2}"
ORBIT2_SIF="${ORBIT2_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
ORBIT2_OVERLAY="${ORBIT2_OVERLAY:-${ORBIT2_BASE}/overlays/orbit2-overlay.img}"
ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${ORBIT2_BASE}/data/superres/prism/10.0_arcmin}"
ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-3}"
ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES:-20}"
ORBIT2_BATCH_SIZE="${ORBIT2_BATCH_SIZE:-4}"
ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR:-${ORBIT2_BASE}/perf-runs/${SLURM_JOB_ID:-$$}}"
PROFILE_TARGET_EPOCH="${PROFILE_TARGET_EPOCH:-0}"

: "${OMNIHUB_TOOLS_DIR:?OMNIHUB_TOOLS_DIR must be set}"
OMNISTAT_VENV="${OMNISTAT_VENV:-${OMNIHUB_TOOLS_DIR}/omnihub-inspect}"
OMNISTAT_TEMPLATE="${OMNISTAT_TEMPLATE:-${SCRIPT_DIR}/../recipes/perf-analysis/omnistat.config.template}"
OMNISTAT_USERMODE_INTERVAL="${OMNISTAT_USERMODE_INTERVAL:-1}"

TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"

NODES="${SLURM_JOB_NUM_NODES:-2}"
GPUS_PER_NODE=8
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

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

JOB_CONFIG="${ORBIT2_OUTPUT_DIR}/interm_8m_lux_${SLURM_JOB_ID:-$$}.yaml"
python3 "$SCRIPT_DIR/render_orbit2_config.py" \
  --nodes "$NODES" \
  --gpus-per-node "$GPUS_PER_NODE" \
  --data-root "$ORBIT2_DATA_ROOT" \
  --max-epochs "$ORBIT2_MAX_EPOCH" \
  --batch-size "$ORBIT2_BATCH_SIZE" \
  -o "$JOB_CONFIG"

echo "=== ORBIT-2 Perf Analysis Run ==="
echo "  Nodes        : $NODES"
echo "  Total ranks  : $TOTAL_RANKS"
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
      echo "NODE_HEALTH $H home=$HM shared=$SH sif=$SIF_CHECK"
    ' 2>&1 || true
  _BAD_NODES=$(grep "^NODE_HEALTH" "$_PROBE_OUT" 2>/dev/null | awk '/FAIL/ {print $2}' | sort -u | tr '\n' ',' | sed 's/,$//' || true)
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
PROFILER_HOOK="${SCRIPT_DIR}/orbit2_profiler_hook.py"
TRACE_DIR="${ORBIT2_OUTPUT_DIR}/traces"

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
export PYTHONNOUSERSITE=1
export HSA_NO_SCRATCH_RECLAIM=1
export ORBIT_USE_DDSTORE=0
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES}"
export ORBIT2_ROOT="/orbit2"
export ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES}"
export ORBIT2_DATA_TYPE="${ORBIT2_DATA_TYPE:-float32}"
export ORBIT2_FUSED_ATTN="${ORBIT2_FUSED_ATTN:-DEFAULT}"
export ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR}"
export ORBIT2_PROFILE_DIR="${TRACE_DIR}"
export PROFILE_TARGET_EPOCH="${PROFILE_TARGET_EPOCH}"
export PROFILE_RANK0_ONLY=1
export ORBIT2_RANK_PRE_TRAIN_HOOK="${PROFILER_HOOK}"
cd /orbit2/examples
exec python3 /examples/run_orbit2_train.py /config/config.yaml
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

echo "--- Launching perf training: $TOTAL_RANKS ranks ---"
set +e
srun --mpi=pmix apptainer exec \
    --rocm --overlay "${ORBIT2_OVERLAY}:ro" \
    --bind "/opt/ompi:/opt/ompi:ro" \
    --bind "$ORBIT2_ROOT":/orbit2 \
    --bind "$SCRIPT_DIR":/examples \
    --bind "$(dirname "$RANK_SCRIPT"):$(dirname "$RANK_SCRIPT")" \
    --bind "$ORBIT2_DATA_ROOT":"$ORBIT2_DATA_ROOT":ro \
    --bind "$ORBIT2_OUTPUT_DIR":"$ORBIT2_OUTPUT_DIR" \
    --bind "$CONFIG_BIND" \
    --env HOSTNAME="$MASTER_ADDR" --env MASTER_PORT="$MASTER_PORT" \
    --env PMIX_MCA_gds=hash --env PMIX_MCA_psec=native \
    --env PYTHONPATH="/opt/orbit2-pkgs:/orbit2/src:/orbit2" \
    --env LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib \
    "${MPI_MULTINODE_ENVS[@]}" "${RCCL_MULTINODE_ENVS[@]}" \
    "$ORBIT2_SIF" bash "$RANK_SCRIPT"
TRAIN_RC=$?
set -e

_SLURM_LOG="${ORBIT2_SLURM_LOG:-${SLURM_SUBMIT_DIR:-.}/orbit2-train-${SLURM_JOB_ID}.out}"
if [[ ! -f "$_SLURM_LOG" ]]; then
  _SLURM_LOG="${AI4S_SHARED_DIR}/models/ORBIT-2/outputs/train/logs/orbit2-train-${SLURM_JOB_ID}.out"
fi
cp -f "$_SLURM_LOG" "${ORBIT2_OUTPUT_DIR}/orbit2-train-${SLURM_JOB_ID}.out" 2>/dev/null || true

python3 - "$ORBIT2_OUTPUT_DIR" "${SLURM_JOB_ID}" "$TRAIN_RC" <<'PYEOF'
import json, sys
from pathlib import Path
job_dir, job_id, train_rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
manifest = {
    "job_id": job_id,
    "model": "ORBIT-2",
    "workload": "intermediate_downscaling",
    "job_dir": str(job_dir),
    "slurm_log": str(job_dir / f"orbit2-train-{job_id}.out"),
    "omnistat_config": str(job_dir / "omnistat.config"),
    "omnistat_db": str(job_dir / "omnistat-db"),
    "trace_dir": str(job_dir / "traces"),
    "state": "complete" if train_rc == 0 else "failed",
    "exit_code": train_rc,
}
(job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote {job_dir / 'manifest.json'}")
PYEOF

echo "=== ORBIT-2 perf run complete (rc=$TRAIN_RC) ==="
exit $TRAIN_RC
