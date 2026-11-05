#!/usr/bin/env bash
# Allocation-neutral ORBIT-2 inference with Omnistat user-mode telemetry.
#
# Wraps run_infer_amd.sh: starts Omnistat immediately before the timed
# inference window, writes manifest.json + foms.json under ORBIT2_OUTPUT_DIR.
# Never nests sbatch.
#
# Required:
#   AI4S_SHARED_DIR, PERF_TOOLS_DIR, ORBIT2_ROOT
# Optional:
#   ORBIT2_OUTPUT_DIR, ORBIT2_SIF, ORBIT2_OVERLAY, ORBIT2_USE_SYNTHETIC=1,
#   ORBIT2_INFER_REPEATS (default 1), ORBIT2_INFER_WARMUP (default 0)

set -euo pipefail

ORBIT2_PERF_START_TS=$(date +%s)

# Prefer BASH_SOURCE for nested campaign/runtime calls; scontrol only for spool copies.
_SRC="${BASH_SOURCE[0]:-}"
if [[ -n "$_SRC" && "$_SRC" != *"/var/spool/slurmd/"* && "$_SRC" != *"/spool/slurmd/"* ]]; then
  SCRIPT_DIR=$(cd "$(dirname "$_SRC")" && pwd)
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

: "${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}"
: "${PERF_TOOLS_DIR:?PERF_TOOLS_DIR must be set}"
: "${ORBIT2_ROOT:?ORBIT2_ROOT must be set}"

ORBIT2_BASE="${AI4S_SHARED_DIR}/models/ORBIT-2"
RUN_ID="${AI4S_RUN_ID:-${SLURM_JOB_ID:-$$}}"
ORBIT2_OUTPUT_DIR="${ORBIT2_OUTPUT_DIR:-${AI4S_OUTPUT_DIR:-${ORBIT2_BASE}/infer-perf-runs/${RUN_ID}}}"
OMNISTAT_VENV="${OMNISTAT_VENV:-${PERF_TOOLS_DIR}/perf-inspect}"
OMNISTAT_TEMPLATE="${OMNISTAT_TEMPLATE:-${SCRIPT_DIR}/../recipes/perf-analysis/omnistat.config.template}"
OMNISTAT_USERMODE_INTERVAL="${OMNISTAT_USERMODE_INTERVAL:-1}"
ORBIT2_INFER_REPEATS="${ORBIT2_INFER_REPEATS:-1}"
ORBIT2_INFER_WARMUP="${ORBIT2_INFER_WARMUP:-0}"

for var in OMNISTAT_TEMPLATE; do
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
    -e "s|@PERF_TOOLS_DIR@|${PERF_TOOLS_DIR}|g" \
    "$OMNISTAT_TEMPLATE" > "$OMNISTAT_CONFIG"
if [[ -n "${OMNISTAT_ROCPROF_PROFILE:-}" ]]; then
  sed -i.bak "s/^profile = .*/profile = ${OMNISTAT_ROCPROF_PROFILE}/" "$OMNISTAT_CONFIG" && rm -f "${OMNISTAT_CONFIG}.bak"
fi

export PATH="${OMNISTAT_VENV}/bin:${PATH}"
export ORBIT2_OUTPUT_DIR
export AI4S_OUTPUT_DIR="$ORBIT2_OUTPUT_DIR"

echo "=== ORBIT-2 Inference Perf Run ==="
echo "  ORBIT2_ROOT     : $ORBIT2_ROOT"
echo "  Output dir      : $ORBIT2_OUTPUT_DIR"
echo "  Omnistat cfg    : $OMNISTAT_CONFIG"
echo "  Warmup/repeats  : ${ORBIT2_INFER_WARMUP}/${ORBIT2_INFER_REPEATS}"
echo "  Node(s)         : ${SLURM_NODELIST:-$(hostname)}"
echo ""

# Omnistat starts immediately before timed inference (not during dep install).
# run_infer_amd.sh may still do synth/deps first — we start collectors right
# before invoking it and document that prep may appear in the window unless
# ORBIT2_PERF_SKIP_PREP=1 and prep was done earlier. For MVP we mark the
# timed window around the runner invocation with explicit markers.
echo "--- Starting Omnistat user-mode ---"
"${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --start --interval "$OMNISTAT_USERMODE_INTERVAL" \
    2>&1 | tee "${ORBIT2_OUTPUT_DIR}/omnistat_start.log" || true

cleanup_omnistat() {
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopexporters || true
  "${OMNISTAT_VENV}/bin/omnistat-usermode" --configfile "$OMNISTAT_CONFIG" --stopserver || true
}
trap cleanup_omnistat EXIT

INFER_RC=0
INFER_START=$(date +%s.%N)
echo "ORBIT2_PERF_MARKER inference_start $(date -Is) ${INFER_START}" | tee -a "${ORBIT2_OUTPUT_DIR}/timing_markers.log"

# Warmup + timed repeats (each full visualize invocation).
_total=$((ORBIT2_INFER_WARMUP + ORBIT2_INFER_REPEATS))
for ((i=1; i<=_total; i++)); do
  echo "--- Infer invocation $i / ${_total} ---"
  if ! bash "$SCRIPT_DIR/run_infer_amd.sh" "$@"; then
    INFER_RC=$?
    echo "ERROR: run_infer_amd.sh failed on invocation $i (rc=$INFER_RC)" >&2
    break
  fi
done

INFER_END=$(date +%s.%N)
echo "ORBIT2_PERF_MARKER inference_end $(date -Is) ${INFER_END}" | tee -a "${ORBIT2_OUTPUT_DIR}/timing_markers.log"

# Stop Omnistat before writing manifests so DB is flushed.
cleanup_omnistat
trap - EXIT

# Copy SLURM log if present
JOB_ID="${SLURM_JOB_ID:-}"
if [[ -n "$JOB_ID" ]]; then
  for cand in "orbit2-vis-${JOB_ID}.out" "orbit2-infer-perf-${JOB_ID}.out"; do
    if [[ -f "$cand" ]]; then
      cp -f "$cand" "${ORBIT2_OUTPUT_DIR}/$cand" || true
    fi
  done
fi

RUNTIME_SECONDS=$(( $(date +%s) - ORBIT2_PERF_START_TS ))
python3 - "$ORBIT2_OUTPUT_DIR" "$RUN_ID" "$INFER_RC" "$RUNTIME_SECONDS" "$ORBIT2_ROOT" \
  "$ORBIT2_INFER_WARMUP" "$ORBIT2_INFER_REPEATS" "$INFER_START" "$INFER_END" <<'PY'
import json, os, subprocess, sys
from pathlib import Path

out, run_id, rc, runtime, root = sys.argv[1:6]
warmup, repeats, t0, t1 = sys.argv[6:10]
job_dir = Path(out)
rc = int(rc)
runtime = int(runtime)
warmup = int(warmup)
repeats = int(repeats)
wall = None
try:
    wall = float(t1) - float(t0)
except ValueError:
    pass

git_sha = None
try:
    git_sha = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    pass

state = "ok" if rc == 0 else "failed"
manifest = {
    "job_id": os.environ.get("SLURM_JOB_ID") or run_id,
    "run_id": run_id,
    "job_dir": str(job_dir),
    "omnistat_db": str(job_dir / "omnistat-db"),
    "omnistat_config": str(job_dir / "omnistat.config"),
    "state": state,
    "exit_code": rc,
    "runtime_seconds": runtime,
    "inference_wall_seconds": wall,
    "orbit2_root": root,
    "git_sha": git_sha,
    "workload": "visualize",
    "warmup_repeats": warmup,
    "timed_repeats": repeats,
    "use_synthetic": os.environ.get("ORBIT2_USE_SYNTHETIC", "0"),
    "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
    "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
}
(job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {job_dir / 'manifest.json'}")
PY

# Deterministic FOMs (no training-log parser).
python3 "$SCRIPT_DIR/run_infer_fom_extractor.py" --job-dir "$ORBIT2_OUTPUT_DIR" || true

# Stage result.json for the campaign runner.
python3 - "$ORBIT2_OUTPUT_DIR" "$RUN_ID" "$INFER_RC" <<'PY'
import json, sys
from pathlib import Path
job_dir = Path(sys.argv[1])
run_id = sys.argv[2]
rc = int(sys.argv[3])
manifest = job_dir / "manifest.json"
foms = job_dir / "foms.json"
omni = job_dir / "omnistat-db"
status = "pass" if rc == 0 and manifest.is_file() else ("partial" if rc == 0 else "error")
artifacts = {}
for key, path in (("manifest", manifest), ("foms", foms), ("omnistat_db", omni), ("omnistat_config", job_dir / "omnistat.config")):
    if path.exists():
        artifacts[key] = {"path": str(path), "sha256": None, "required_for_next": key in {"manifest", "foms"}}
result = {
    "schema_version": "1",
    "run_id": run_id,
    "name": "orbit_inference",
    "status": status,
    "checks": [{"name": "exit_status", "status": "pass" if rc == 0 else "fail", "detail": f"rc={rc}"}],
    "metrics": {},
    "artifacts": artifacts,
    "error": None if rc == 0 else {"type": "InferFailed", "message": f"exit {rc}"},
}
(job_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {job_dir / 'result.json'}")
PY

exit "$INFER_RC"
