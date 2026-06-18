#!/usr/bin/env bash
# run_2node_scaleout_loop.sh — unattended ORBIT-2 2-node scale-out lever loop.
#
# Submits a 2-node bf16 baseline + 5 lever iterations (one SLURM job each, serial),
# extracts HONEST throughput (real per-step batch; see run_fom_extractor.py) after each,
# and writes a cross-node REPORT.md comparing 1-node vs 2-node.
#
# Designed to run unattended in tmux (survives disconnect):
#   tmux new -s o2scale -d 'bash earth_science/models/ORBIT-2/examples/run_2node_scaleout_loop.sh'
#
# Prereqs (the loop sets sane defaults but you can override via env):
#   - 5x ERA5 staging present (stage_era5_3x_symlink.sh DATA_ROOT 5 -> 100 train shards),
#     so per_worker stays >=1 and full batches refill at 2 nodes (data_par_size=16).
#   - .cluster-config.yaml present (multi-node RCCL: ib_hca, mgmt_iface, plugin, libionic).
#
# Levers tested at 2 nodes (compute-bound at 1 node; FSDP all-gather over IB is the new
# cost at N>1, so comm-tuning is where multi-node wins, if any, live):
#   iter-0 baseline | iter-1 IB QPS=2 | iter-2 IB QPS=4 | iter-3 GPU_MAX_HW_QUEUES=4
#   iter-4 torch.compile | iter-5 NCCL_MIN_NCHANNELS=112

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
SBATCH="${SCRIPT_DIR}/sbatch_train_perf_amd.sh"

# ---- environment (override-able) -------------------------------------------
export AI4S_SHARED_DIR="${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}"
export PERF_TOOLS_DIR="${PERF_TOOLS_DIR:?PERF_TOOLS_DIR must be set (perf_tools.dir in .cluster-config.yaml)}"
export ORBIT2_ROOT="${ORBIT2_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/code/bayes-cast}"
export ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/data/superres/era5/1.0_deg}"
export ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-edm_8m_era5_1x8.yaml}"

PARTITION="${SBATCH_PARTITION:?SBATCH_PARTITION must be set (slurm.partition in .cluster-config.yaml)}"
ACCOUNT="${SBATCH_ACCOUNT:?SBATCH_ACCOUNT must be set (slurm.account in .cluster-config.yaml)}"
NODES=2
# Node placement (all optional, all portable defaults = no pinning):
#   O2_NODELIST       pin to specific nodes (--nodelist)
#   O2_NODE_EXCLUDE   exclude specific nodes (--exclude); set directly, or
#   O2_EXCLUDE_PATTERN  grep -E pattern matched against `sinfo` node names to auto-build the
#                       exclude list (use when a known-bad node range must be skipped — e.g. a
#                       sub-range with an inconsistent RoCE GID table that aborts multi-node RCCL).
O2_NODELIST="${O2_NODELIST:-}"
if [[ -z "${O2_NODE_EXCLUDE:-}" && -n "${O2_EXCLUDE_PATTERN:-}" ]]; then
  O2_NODE_EXCLUDE=$(sinfo -p "$PARTITION" -N -h -o "%n" 2>/dev/null | sort -u | grep -E "$O2_EXCLUDE_PATTERN" | paste -sd, || true)
fi
O2_NODE_EXCLUDE="${O2_NODE_EXCLUDE:-}"
BATCH=4096
MAXEPOCH=6
MAXBATCH=20
POLL_S="${POLL_S:-30}"
TIME_LIMIT="${TIME_LIMIT:-02:30:00}"

LOOP_ID="${LOOP_ID:-loop-2node-scaleout-001}"
LOOP_DIR="${AI4S_SHARED_DIR}/models/ORBIT-2/perf-runs/${LOOP_ID}"
mkdir -p "$LOOP_DIR"
RESULTS_CSV="${LOOP_DIR}/results.csv"
STATUS="${LOOP_DIR}/STATUS.txt"
REPORT="${LOOP_DIR}/REPORT.md"
EXTRACT="${SCRIPT_DIR}/run_fom_extractor.py"

# 1-node honest reference (clean num_workers test, batch 4096, 100 shards).
ONE_NODE_NW1_JOB="${ONE_NODE_NW1_JOB:-10504}"
ONE_NODE_NW4_JOB="${ONE_NODE_NW4_JOB:-10505}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

# ---- lever definitions (name | extra KEY=VALUE env, space-separated) -------
LEVER_NAME=(  "baseline"  "ib_qps_2"                        "ib_qps_4"                        "gpu_hw_queues_4"        "torch_compile"                                   "nccl_min_nchannels_112" )
LEVER_ENV=(   ""          "NCCL_IB_QPS_PER_CONNECTION=2"    "NCCL_IB_QPS_PER_CONNECTION=4"    "GPU_MAX_HW_QUEUES=4"    "ORBIT2_TORCH_COMPILE=1 ORBIT2_COMPILE_MODE=default" "NCCL_MIN_NCHANNELS=112" )

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "iter,lever,job_id,nodes,state,throughput_sps,method,partial_frac,realized_dims,hbm_pct,steady_batch_time_s,max_batches_per_epoch,verdict" > "$RESULTS_CSV"
fi

log "=== ${LOOP_ID} START (2-node scale-out, ${#LEVER_NAME[@]} configs incl. baseline) ==="
log "loop dir: $LOOP_DIR"

# Verify 5x staging (>=96 shards). Stage if missing.
N_SHARDS=$(find "${ORBIT2_DATA_ROOT}/train" -maxdepth 1 -name '*_*.npz' ! -name 'climatology.npz' 2>/dev/null | wc -l)
if [[ "$N_SHARDS" -lt 96 ]]; then
  log "staging 5x ERA5 (have $N_SHARDS shards, need >=96)"
  bash "${SCRIPT_DIR}/stage_era5_3x_symlink.sh" "$ORBIT2_DATA_ROOT" 5 2>&1 | tee -a "$STATUS"
  N_SHARDS=$(find "${ORBIT2_DATA_ROOT}/train" -maxdepth 1 -name '*_*.npz' ! -name 'climatology.npz' | wc -l)
fi
log "train shards: $N_SHARDS (per_worker @2node nw=1 = $((N_SHARDS / 16)))"

GLOBAL_BATCH=$((BATCH * NODES * 8))   # fsdp=nodes, simple_ddp=8 -> ranks = nodes*8
BASELINE_TP=""

read_fom() {  # $1=job_dir  -> sets FOM_* globals
  python3 - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"foms.json"
d=json.loads(p.read_text()) if p.is_file() else {}
def g(k,default=""):
    v=d.get(k); return default if v is None else v
dims=g("steady_realized_batch_dims","")
if isinstance(dims,list): dims="/".join(str(x) for x in dims)
# single-quote every value so eval is safe regardless of content (dims, etc.)
print("\n".join([
  f"FOM_TP='{g('throughput_samples_per_s','')}'",
  f"FOM_METHOD='{g('throughput_method','')}'",
  f"FOM_PARTIAL='{g('partial_step_fraction','')}'",
  f"FOM_DIMS='{dims}'",
  f"FOM_HBM='{g('hbm_reserved_pct_288','')}'",
  f"FOM_BT='{g('steady_batch_time_s','')}'",
  f"FOM_MAXB='{g('max_batches_per_epoch','')}'",
]))
PY
}

# Resume: skip iters already recorded in results.csv; reseed baseline from iter-0 row.
declare -A DONE_ITER=()
while IFS=, read -r ri rl rj rn rstate rtp rest; do
  [[ "$ri" == "iter" || -z "$ri" ]] && continue
  DONE_ITER["$ri"]=1
  if [[ "$ri" == "0" && -n "$rtp" ]]; then
    BASELINE_TP="$rtp"
    bj=$(ls -d "${AI4S_SHARED_DIR}/models/ORBIT-2/perf-runs/${rj}" 2>/dev/null)
    if [[ -n "$bj" ]]; then eval "$(read_fom "$bj")"; BASELINE_DIMS="$FOM_DIMS"; fi
  fi
done < "$RESULTS_CSV"
[[ -n "${BASELINE_TP:-}" ]] && log "resume: baseline throughput=${BASELINE_TP} dims=[${BASELINE_DIMS:-}] (iter-0 already done)"

for i in "${!LEVER_NAME[@]}"; do
  name="${LEVER_NAME[$i]}"
  extra="${LEVER_ENV[$i]}"
  if [[ -n "${DONE_ITER[$i]:-}" ]]; then
    log "iter-$i ($name) already in results.csv — skipping"
    continue
  fi
  log "--- iter-$i lever='$name' extra='${extra:-<none>}' ---"

  # Build --export list: base config + lever extras.
  EXP="ALL"
  EXP+=",ORBIT2_BATCH_SIZE=${BATCH},ORBIT2_NUM_WORKERS=1,ORBIT2_DATA_TYPE=bfloat16,ORBIT2_FUSED_ATTN=DEFAULT"
  EXP+=",ORBIT2_CONFIG_TEMPLATE=${ORBIT2_CONFIG_TEMPLATE},ORBIT2_MAX_EPOCH=${MAXEPOCH},ORBIT2_MAX_BATCHES=${MAXBATCH}"
  EXP+=",ORBIT2_DATA_ROOT=${ORBIT2_DATA_ROOT},ORBIT2_ROOT=${ORBIT2_ROOT},AI4S_SHARED_DIR=${AI4S_SHARED_DIR},PERF_TOOLS_DIR=${PERF_TOOLS_DIR}"
  EXP+=",TORCH_NCCL_HIGH_PRIORITY=1,GPU_MAX_HW_QUEUES=2"
  for kv in $extra; do EXP+=",${kv}"; done

  PLACE=()
  if [[ -n "$O2_NODELIST" ]]; then PLACE=(--nodelist="$O2_NODELIST")
  elif [[ -n "$O2_NODE_EXCLUDE" ]]; then PLACE=(--exclude="$O2_NODE_EXCLUDE"); fi
  JID=$( cd "$REPO_ROOT" && sbatch --parsable \
      --partition="$PARTITION" --account="$ACCOUNT" \
      --nodes="$NODES" --time="$TIME_LIMIT" \
      --job-name="o2-2n-${name}" "${PLACE[@]}" \
      --export="$EXP" "$SBATCH" 2>>"$STATUS" )
  if [[ -z "$JID" ]]; then
    log "iter-$i SUBMIT FAILED"
    echo "$i,$name,SUBMIT_FAIL,$NODES,submit_fail,,,,,,,,error" >> "$RESULTS_CSV"
    continue
  fi
  log "iter-$i submitted job $JID; polling every ${POLL_S}s"

  # Poll until job leaves the queue (initial sleep avoids the submit->squeue race).
  sleep 15
  while squeue -h -j "$JID" 2>/dev/null | grep -q .; do sleep "$POLL_S"; done
  sleep 20  # let perf-sbatch finish omnistat cleanup + write manifest/log

  JOB_DIR="${AI4S_SHARED_DIR}/models/ORBIT-2/perf-runs/${JID}"
  STATE=$(python3 -c "import json,sys;print(json.load(open('${JOB_DIR}/manifest.json')).get('state','unknown'))" 2>/dev/null || echo "no_manifest")
  log "iter-$i job $JID finished (state=$STATE)"

  python3 "$EXTRACT" --job-dir "$JOB_DIR" >/dev/null 2>>"$STATUS" || log "iter-$i FOM extract warning"
  eval "$(read_fom "$JOB_DIR")"

  # Verdict (honest throughput + integrity).
  verdict="n/a"
  if [[ "$i" -eq 0 ]]; then
    BASELINE_TP="$FOM_TP"; BASELINE_DIMS="$FOM_DIMS"; verdict="baseline"
  elif [[ "$STATE" != "complete" ]]; then
    verdict="failed"
  elif [[ -n "$FOM_TP" && -n "$BASELINE_TP" ]]; then
    verdict=$(python3 -c "
b=float('$BASELINE_TP'); t=float('$FOM_TP')
dimic = ('$FOM_DIMS'=='$BASELINE_DIMS')
d=(t-b)/b*100
if not dimic: print(f'REJECT-artifact({d:+.1f}%,dims_differ)')
elif d>2: print(f'ACCEPT(+{d:.1f}%)')
elif d<-2: print(f'REGRESS({d:.1f}%)')
else: print(f'neutral({d:+.1f}%)')
" 2>/dev/null || echo "parse_err")
  fi
  log "iter-$i throughput=${FOM_TP} s/s method=${FOM_METHOD} partial=${FOM_PARTIAL} dims=[${FOM_DIMS}] hbm=${FOM_HBM}% verdict=$verdict"
  echo "$i,$name,$JID,$NODES,$STATE,$FOM_TP,$FOM_METHOD,$FOM_PARTIAL,\"$FOM_DIMS\",$FOM_HBM,$FOM_BT,$FOM_MAXB,$verdict" >> "$RESULTS_CSV"
done

# ---- report ----------------------------------------------------------------
log "=== generating REPORT.md ==="
python3 - "$RESULTS_CSV" "$REPORT" "$ONE_NODE_NW1_JOB" "$ONE_NODE_NW4_JOB" "$AI4S_SHARED_DIR" "$BATCH" <<'PY'
import csv,json,sys
from pathlib import Path
csv_path,report,j1,j4,shared,batch=sys.argv[1:7]
batch=int(batch)
def fom(job):
    p=Path(shared)/"models/ORBIT-2/perf-runs"/str(job)/"foms.json"
    return json.loads(p.read_text()) if p.is_file() else {}
one_nw1=fom(j1); one_nw4=fom(j4)
rows=list(csv.DictReader(open(csv_path)))
base=next((r for r in rows if r["iter"]=="0"), None)
base_tp=float(base["throughput_sps"]) if base and base["throughput_sps"] else None
one_tp=one_nw1.get("throughput_samples_per_s")

L=[]
L.append("# ORBIT-2 Scale-out Report — 1-node vs 2-node (Bayes-CAST EDM, MI355X, bf16)\n")
L.append("All throughput is **honest** (real per-step batch × ranks ÷ real step time), so partial\n")
L.append("trailing batches do not inflate it. batch=%d/rank, EFFICIENT SDPA, 100 ERA5 shards (5× staged).\n" % batch)

L.append("\n## 1-node reference (clean num_workers test)\n")
L.append("| config | job | throughput (s/s) | method | realized batch dims | partial frac | HBM%% |")
L.append("|---|---|---:|---|---|---:|---:|")
for lbl,f,j in [("1-node nw=1",one_nw1,j1),("1-node nw=4",one_nw4,j4)]:
    L.append(f"| {lbl} | {j} | {f.get('throughput_samples_per_s','-'):.0f} | {f.get('throughput_method','-')} | {f.get('steady_realized_batch_dims','-')} | {f.get('partial_step_fraction','-')} | {f.get('hbm_reserved_pct_288','-')} |" if f else f"| {lbl} | {j} | - | - | - | - | - |")

L.append("\n## 2-node lever loop\n")
L.append("| iter | lever | job | state | throughput (s/s) | Δ vs 2N base | scaling vs 1N | partial | dims | HBM%% | verdict |")
L.append("|---:|---|---|---|---:|---:|---:|---:|---|---:|---|")
for r in rows:
    tp=r["throughput_sps"]
    tpf=float(tp) if tp else None
    dvb=f"{(tpf-base_tp)/base_tp*100:+.1f}%" if (tpf and base_tp) else "-"
    scal=f"{tpf/one_tp:.2f}×" if (tpf and one_tp) else "-"
    tps=f"{tpf:.0f}" if tpf else "-"
    L.append(f"| {r['iter']} | {r['lever']} | {r['job_id']} | {r['state']} | {tps} | {dvb} | {scal} | {r['partial_frac']} | {r['realized_dims']} | {r['hbm_pct']} | {r['verdict']} |")

# headline
if base_tp and one_tp:
    eff=base_tp/one_tp/2*100
    L.append(f"\n## Headline\n")
    L.append(f"- **2-node baseline: {base_tp:.0f} s/s vs 1-node: {one_tp:.0f} s/s → {base_tp/one_tp:.2f}× ({eff:.0f}% strong-scaling efficiency).**")
    best=max((r for r in rows if r['throughput_sps']), key=lambda r: float(r['throughput_sps']), default=None)
    if best:
        L.append(f"- Best 2-node config: **{best['lever']}** ({float(best['throughput_sps']):.0f} s/s, {best['verdict']}).")
    L.append(f"- Accepted levers (>+2%% over 2N baseline, integrity-matched): " +
             (", ".join(r['lever'] for r in rows if r['verdict'].startswith('ACCEPT')) or "none — workload stays compute-bound at 2 nodes."))
Path(report).write_text("\n".join(L)+"\n")
print(f"wrote {report}")
PY

log "=== ${LOOP_ID} COMPLETE — see $REPORT ==="
cat "$REPORT" 2>/dev/null | tee -a "$STATUS"
