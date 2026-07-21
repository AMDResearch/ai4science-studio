#!/bin/bash
# 8-GPU HydraGNN DDP training on Alexandria DFT, wrapped with AMD Omnistat GPU
# telemetry. Self-owned: every tool + dataset path lives under /shared/spannala,
# so this does NOT depend on /shared/aaji or /shared/omnihub.
#
# Merges two proven pieces:
#   - the production 8-GPU launch from train_work/scripts/hg_ddp8_v3.sbatch
#     (Alexandria-only, corr 0.89 — the model the studio demo tells its story about)
#   - the Omnistat user-mode start/stop lifecycle + per-job VictoriaMetrics DB
#     from material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh
#
# Telemetry lands in $HG_OUTPUT_DIR/omnistat-db (a VictoriaMetrics TSDB) and a
# manifest.json the studio backend reads to range-query the metrics after the run.
#
# Env (all have defaults; studio backend overrides the first three):
#   PERF_TOOLS_DIR   /shared/spannala/perf-tools           (omnistat venv + VM binary)
#   HG_DATA_DIR      /shared/spannala/models/HydraGNN/weights
#   HG_NUM_EPOCH     200                                   (studio epochs slider)
#   HG_OUTPUT_DIR    <work>/perf-runs/<jobid>              (per-run telemetry dir)
#   N_SAMPLES        600000
#SBATCH --job-name=hg-tele8
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/shared/spannala/models/HydraGNN/train_work/logs/hg_tele8_%j.log
#SBATCH --error=/shared/spannala/models/HydraGNN/train_work/logs/hg_tele8_%j.log

set -uo pipefail

# ── Self-owned paths ────────────────────────────────────────────────────────
PERF_TOOLS_DIR="${PERF_TOOLS_DIR:-/shared/spannala/perf-tools}"
OMNISTAT_VENV="${OMNISTAT_VENV:-${PERF_TOOLS_DIR}/omnistat-venv}"
WORK=/shared/spannala/models/HydraGNN/train_work
HG_DATA_DIR="${HG_DATA_DIR:-/shared/spannala/models/HydraGNN/weights}"
HG_DATASET_BP="${HG_DATASET_BP:-${HG_DATA_DIR}/Alexandria-v2.bp}"
HG_NUM_EPOCH="${HG_NUM_EPOCH:-200}"
# Precision: fp32 = production-accuracy model (default); fp64 = double precision
# to exercise the MI355X FP64 units and populate the FP64 telemetry tile.
HG_PRECISION="${HG_PRECISION:-fp32}"
N_SAMPLES="${N_SAMPLES:-600000}"
SCAN_LIMIT="${SCAN_LIMIT:-1500000}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-${WORK}/perf-runs/${SLURM_JOB_ID:-$$}}"
# Omnistat config template. SLURM copies this script into its spool dir before
# running, so BASH_SOURCE does NOT point at the repo — use an absolute default.
# A self-contained copy lives beside this script in studio/telemetry/.
OMNISTAT_TEMPLATE="${OMNISTAT_TEMPLATE:-/home/spannala/Projects/ai4science-studio/studio/telemetry/omnistat.config.template}"
OMNISTAT_USERMODE_INTERVAL="${OMNISTAT_USERMODE_INTERVAL:-1}"

export HG_SIF=/shared/spannala/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
export HG_OVERLAY=/shared/spannala/models/HydraGNN/overlays/hydragnn-overlay.img
export HG_INFER_REPO=/shared/spannala/models/HydraGNN/outputs/HydraGNN-infer

mkdir -p "$HG_OUTPUT_DIR" "$WORK/logs" "$WORK/scripts"

echo "=== HydraGNN 8-GPU DDP training + Omnistat telemetry ==="
echo "Node        : $(hostname)"
echo "Date        : $(date)"
echo "JobID       : ${SLURM_JOB_ID:-n/a}"
echo "Epochs      : $HG_NUM_EPOCH"
echo "Precision   : $HG_PRECISION"
echo "N_SAMPLES   : $N_SAMPLES"
echo "Dataset     : $HG_DATASET_BP"
echo "PerfTools   : $PERF_TOOLS_DIR"
echo "OutputDir   : $HG_OUTPUT_DIR"
echo ""

# ── Preflight ───────────────────────────────────────────────────────────────
for p in "$HG_SIF" "$HG_OVERLAY" "$HG_DATASET_BP" "$OMNISTAT_TEMPLATE"; do
  [[ -e "$p" ]] || { echo "ERROR: required path missing: $p" >&2; exit 2; }
done
_TELEMETRY_ON=1
if [[ ! -x "${OMNISTAT_VENV}/bin/omnistat-usermode" ]]; then
  echo "WARN: omnistat-usermode not found at ${OMNISTAT_VENV}/bin — running WITHOUT telemetry" >&2
  _TELEMETRY_ON=0
fi

# ── Render per-job omnistat config (substitute @JOB_DIR@ + @PERF_TOOLS_DIR@) ──
OMNISTAT_CONFIG="${HG_OUTPUT_DIR}/omnistat.config"
if [[ "$_TELEMETRY_ON" == "1" ]]; then
  sed -e "s|@JOB_DIR@|${HG_OUTPUT_DIR}|g" \
      -e "s|@PERF_TOOLS_DIR@|${PERF_TOOLS_DIR}|g" \
      "$OMNISTAT_TEMPLATE" > "$OMNISTAT_CONFIG"
  echo "--- Starting Omnistat user-mode (interval=${OMNISTAT_USERMODE_INTERVAL}s) ---"
  export PATH="${OMNISTAT_VENV}/bin:${PATH}"
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" \
      --start --interval "$OMNISTAT_USERMODE_INTERVAL" \
      2>&1 | tee "${HG_OUTPUT_DIR}/omnistat_start.log" || {
    echo "WARN: omnistat-usermode --start returned nonzero; continuing without telemetry" >&2
    _TELEMETRY_ON=0
  }
fi

cleanup_omnistat() {
  [[ "$_TELEMETRY_ON" == "1" ]] || return 0
  echo "--- Stopping Omnistat user-mode ---"
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopexporters || true
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopserver || true
}
trap cleanup_omnistat EXIT

# ── Per-rank launcher (one container per task, bound to its local GPU) ────────
cat > "$WORK/scripts/tele8_rank.sh" << 'RANKEOF'
#!/bin/bash
source /opt/venv/bin/activate
# All 8 GPUs stay visible to every rank (RCCL/XGMI topology); device chosen from
# SLURM_LOCALID. Do NOT set ROCR_VISIBLE_DEVICES per rank.
export HSA_NO_SCRATCH_RECLAIM=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8899
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export HYDRAGNN_MASTER_ADDR=127.0.0.1
export HYDRAGNN_MASTER_PORT=8899
export HYDRAGNN_BACKEND=nccl
cd /shared/spannala/models/HydraGNN/train_work/training
python3 hg_train_ddp.py
RANKEOF
chmod +x "$WORK/scripts/tele8_rank.sh"

# Strip outer PMIX/PMI/OMPI env (MPI_Init collision inside container otherwise).
_U=(); while IFS= read -r v; do _U+=(-u "$v"); done \
  < <(env | grep -oE '^(PMIX_|PMI_|OMPI_)[A-Za-z0-9_]+')

_T0=$(date +%s)
srun --ntasks=8 --gpus-per-node=8 --mpi=pmix --cpu-bind=none \
  env "${_U[@]}" apptainer exec --rocm \
  --overlay "${HG_OVERLAY}:ro" \
  --bind "${HG_DATA_DIR}:${HG_DATA_DIR}:ro" \
  --bind "$WORK:$WORK" \
  --bind "$HG_INFER_REPO:$HG_INFER_REPO" \
  --env HG_INFER_REPO="$HG_INFER_REPO" \
  --env HG_DATASET_BP="$HG_DATASET_BP" \
  --env HG_NUM_EPOCH="$HG_NUM_EPOCH" \
  --env HG_PRECISION="$HG_PRECISION" \
  --env TRAIN_CONFIG="$WORK/training/hg_train_config_v3.json" \
  --env VAL_OUT="${HG_OUTPUT_DIR}/validation.json" \
  --env MODEL_OUT="${HG_OUTPUT_DIR}/hg_model.pk" \
  --env SCAN_LIMIT="$SCAN_LIMIT" \
  --env N_SAMPLES="$N_SAMPLES" \
  --env PYTHONPATH=/shared/spannala/models/HydraGNN/outputs/HydraGNN-infer:/opt/hydragnn-pkgs \
  --env LD_LIBRARY_PATH=/opt/hydragnn-pkgs/adios2:/opt/venv/lib/python3.12/site-packages/torch/lib \
  "$HG_SIF" bash "$WORK/scripts/tele8_rank.sh"

rc=$?
_T1=$(date +%s)
_RUNTIME=$((_T1 - _T0))
echo ""
echo "=== srun exit code: $rc  runtime: ${_RUNTIME}s ==="

# ── Manifest the studio backend reads to harvest telemetry ───────────────────
cat > "${HG_OUTPUT_DIR}/manifest.json" << EOF
{
  "jobid": "${SLURM_JOB_ID:-$$}",
  "runtime_s": ${_RUNTIME},
  "epochs": ${HG_NUM_EPOCH},
  "precision": "${HG_PRECISION}",
  "n_gpus": 8,
  "n_samples": ${N_SAMPLES},
  "telemetry_enabled": ${_TELEMETRY_ON},
  "omnistat_db_path": "${HG_OUTPUT_DIR}/omnistat-db",
  "victoria_binary": "${PERF_TOOLS_DIR}/victoriametrics/victoria-metrics-prod",
  "validation_json": "${HG_OUTPUT_DIR}/validation.json",
  "model_out": "${HG_OUTPUT_DIR}/hg_model.pk"
}
EOF
echo "=== Wrote manifest: ${HG_OUTPUT_DIR}/manifest.json ==="
echo "=== Done at $(date) ==="
exit $rc
