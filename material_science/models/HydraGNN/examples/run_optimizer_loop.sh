#!/usr/bin/env bash
# run_optimizer_loop.sh — entrypoint for the HydraGNN iterative sysopt loop.
#
# Runs pre-flight checks, sets up the loop directory, and invokes the
# orchestrator subagent via Claude Code CLI in the current tmux session.
# Survives ssh disconnect when run inside `tmux new -s hg-loop`.
#
# Usage:
#   export ANTHROPIC_API_KEY=...
#   tmux new -s hg-loop
#   bash material_science/models/HydraGNN/examples/run_optimizer_loop.sh \
#        <loop-uuid> <n_iters_budget>
#   # Ctrl-b d to detach; the loop continues until completion or LOOP_ABORT.
#
# Args:
#   loop-uuid       Unique id for this loop (use `uuidgen`).
#   n_iters_budget  Max iterations (recommend 5).
#
# Abort:
#   Graceful: touch $AI4S_SHARED_DIR/models/HydraGNN/perf-runs/loop-<uuid>/STOP
#   Emergency: scancel <jobid> + tmux kill-session -t hg-loop

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <loop-uuid> <n_iters_budget> [--preflight-only]" >&2
  echo "" >&2
  echo "  --preflight-only  Run pre-flight checks only; do not invoke claude." >&2
  echo "                    Useful for smoke-testing the recipe." >&2
  exit 1
fi

LOOP_UUID="$1"
N_ITERS="$2"
PREFLIGHT_ONLY=0
if [[ "${3:-}" == "--preflight-only" ]]; then
  PREFLIGHT_ONLY=1
fi

# --- Resolve paths ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
RECIPE_DIR="${REPO_ROOT}/material_science/models/HydraGNN/recipes/perf-optimizer-loop"
ORCH_PROMPT="${RECIPE_DIR}/agents/orchestrator.md"

# Loop-dir convention matches recipes/perf-optimizer-loop/README.md
: "${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set (e.g. export AI4S_SHARED_DIR=/shared/\$USER)}"
HG_BASE="${AI4S_SHARED_DIR}/models/HydraGNN"
PERF_RUNS_DIR="${HG_BASE}/perf-runs"
LOOP_DIR="${PERF_RUNS_DIR}/loop-${LOOP_UUID}"
STATUS_FILE="${LOOP_DIR}/STATUS.txt"

mkdir -p "$LOOP_DIR"

# --- Helper: log to STATUS.txt with flock so multiple writers don't clobber ---
_log_status() {
  local msg="$1"
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $msg"
  (
    flock -x 9
    echo "$line" >> "$STATUS_FILE"
  ) 9>>"$STATUS_FILE"
  echo "[STATUS] $line"
}

_log_status "LOOP_START uuid=${LOOP_UUID} n_iters_budget=${N_ITERS} driver=claude-code-cli script=$0"

# --- Pre-flight ---
PREFLIGHT_FAIL_REASON=""
PREFLIGHT_NOTES=()

# 1. cluster up
if command -v sinfo > /dev/null 2>&1; then
  _IDLE_OR_MIX=$(sinfo -h -o '%t' 2>/dev/null | grep -cE '^(idle|mix|alloc)$' || true)
  if [[ "${_IDLE_OR_MIX:-0}" -eq 0 ]]; then
    PREFLIGHT_FAIL_REASON="cluster_down"
    PREFLIGHT_NOTES+=("sinfo reports 0 usable nodes — check partition names in .cluster-config.yaml")
  fi
else
  PREFLIGHT_FAIL_REASON="no_slurm"
  PREFLIGHT_NOTES+=("sinfo not in PATH — must run on a SLURM login node")
fi

# 2. disk free
if [[ -z "$PREFLIGHT_FAIL_REASON" ]]; then
  _USE_PCT=$(df -P "$PERF_RUNS_DIR" | awk 'NR==2 {sub("%","",$5); print $5}')
  if [[ "${_USE_PCT:-0}" -gt 95 ]]; then
    PREFLIGHT_FAIL_REASON="disk_full"
    PREFLIGHT_NOTES+=("/shared at ${_USE_PCT}% — refusing to start a loop that adds ~0.7GB")
  fi
fi

# 3. tools present
for _bin in \
    "/shared/omnihub/tools/omnihub-inspect/bin/omnistat-usermode" \
    "/shared/omnihub/tools/victoriametrics/victoria-metrics-prod"; do
  if [[ -z "$PREFLIGHT_FAIL_REASON" && ! -x "$_bin" ]]; then
    PREFLIGHT_FAIL_REASON="tool_missing"
    PREFLIGHT_NOTES+=("missing: $_bin (run the perf-analysis launcher subagent first to install)")
  fi
done

# 4. SIF + overlay
for _f in \
    "${HG_BASE}/overlays/hydragnn-overlay.img" \
    "${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif"; do  # set AI4S_SHARED_DIR to your cluster's shared storage root
  if [[ -z "$PREFLIGHT_FAIL_REASON" && ! -f "$_f" ]]; then
    PREFLIGHT_FAIL_REASON="image_missing"
    PREFLIGHT_NOTES+=("missing: $_f")
  fi
done

# 5. claude CLI + API key + egress
if [[ -z "$PREFLIGHT_FAIL_REASON" ]]; then
  if ! command -v claude > /dev/null 2>&1; then
    PREFLIGHT_FAIL_REASON="claude_cli_missing"
    PREFLIGHT_NOTES+=("'claude' CLI not in PATH — install per https://docs.anthropic.com/claude/docs/claude-code")
  elif [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    PREFLIGHT_FAIL_REASON="no_api_key"
    PREFLIGHT_NOTES+=("ANTHROPIC_API_KEY env var not set — required for unattended runs")
  else
    # Egress check: we only care that we can REACH api.anthropic.com — not
    # that the bare endpoint accepts our key. A 401/403 response means TLS
    # completed and we got a server reply, i.e. egress works (and the API
    # key, if any, doesn't have read access to /v1/models — that's normal
    # for many key scopes; /v1/messages is what actually matters and is
    # exercised when claude actually runs). Treat 401/403 as success.
    _http_code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 https://api.anthropic.com/v1/models 2>/dev/null || echo "000")
    case "$_http_code" in
      000|5*) PREFLIGHT_FAIL_REASON="api_egress_blocked"
              PREFLIGHT_NOTES+=("cannot reach api.anthropic.com:443 (curl returned ${_http_code})") ;;
      2*|401|403) : ;;  # ok: server reachable
      *) PREFLIGHT_FAIL_REASON="api_egress_unexpected"
         PREFLIGHT_NOTES+=("unexpected http code ${_http_code} from api.anthropic.com/v1/models") ;;
    esac
  fi
fi

# 6. orchestrator prompt present
if [[ -z "$PREFLIGHT_FAIL_REASON" && ! -f "$ORCH_PROMPT" ]]; then
  PREFLIGHT_FAIL_REASON="orch_prompt_missing"
  PREFLIGHT_NOTES+=("orchestrator prompt not found: $ORCH_PROMPT")
fi

if [[ -n "$PREFLIGHT_FAIL_REASON" ]]; then
  _log_status "PREFLIGHT_FAIL reason=${PREFLIGHT_FAIL_REASON} notes='${PREFLIGHT_NOTES[*]}'"
  echo "Pre-flight failed: ${PREFLIGHT_FAIL_REASON}" >&2
  printf '  - %s\n' "${PREFLIGHT_NOTES[@]}" >&2
  exit 2
fi

_DISK_FREE=$(df -h "$PERF_RUNS_DIR" | awk 'NR==2 {print $4}')
_log_status "PREFLIGHT_OK disk_free=${_DISK_FREE} claude_cli=ok api_egress=ok orch_prompt=ok"

if [[ $PREFLIGHT_ONLY -eq 1 ]]; then
  _log_status "PREFLIGHT_ONLY mode: exiting before invoking orchestrator"
  echo ""
  echo "Pre-flight complete; orchestrator NOT invoked (--preflight-only)."
  echo "  STATUS file: $STATUS_FILE"
  exit 0
fi

# --- Trap SIGINT/SIGTERM and log LOOP_ABORT before exiting ---
_on_signal() {
  _log_status "LOOP_ABORT reason=signal sig=$1 (driver exiting; in-flight SLURM job not cancelled)"
  exit 130
}
trap '_on_signal INT' INT
trap '_on_signal TERM' TERM

# --- Invoke orchestrator via claude CLI with retry-with-backoff ---
# We invoke claude ONCE per attempt and let the orchestrator drive the full
# loop. If claude exits non-zero (transient API failure, network hiccup) we
# retry up to 3 times with exp backoff; on the fourth failure we log
# LOOP_ABORT and exit 1.
#
# The orchestrator persists all state to disk (foms.csv, STATUS.txt,
# do_not_retry.json) so a fresh invocation can resume from where the
# previous one left off — see orchestrator.md §0 (Resume detection).

USER_PROMPT="Drive the HydraGNN iterative sysopt loop with these parameters:
- Loop UUID: ${LOOP_UUID}
- Loop directory: ${LOOP_DIR}
- Iterations budget: ${N_ITERS}
- Status file: ${STATUS_FILE}
- Recipe: ${RECIPE_DIR}/README.md

Follow the steps in the orchestrator.md system prompt. Persist all state to
disk so you can resume from where you left off if I have to restart you.
On any LOOP_COMPLETE, LOOP_ABORT, or LOOP_PAUSE, exit with a single line:
  STATUS=<ok|partial|fail>; reason=<short>"

_MAX_ATTEMPTS=3
_attempt=1
while [[ $_attempt -le $_MAX_ATTEMPTS ]]; do
  _log_status "ORCH_INVOKE attempt=${_attempt} max=${_MAX_ATTEMPTS}"
  set +e
  claude \
      --print \
      --dangerously-skip-permissions \
      --max-turns 250 \
      --append-system-prompt "$(cat "$ORCH_PROMPT")" \
      "$USER_PROMPT"
  _rc=$?
  set -e
  if [[ $_rc -eq 0 ]]; then
    _log_status "ORCH_EXIT_OK attempt=${_attempt}"
    break
  fi
  _log_status "ORCH_EXIT_FAIL attempt=${_attempt} rc=${_rc}"
  if [[ $_attempt -eq $_MAX_ATTEMPTS ]]; then
    _log_status "LOOP_ABORT reason=orchestrator_failed rc=${_rc} attempts=${_attempt}"
    exit 1
  fi
  _backoff=$(( 30 * (2 ** (_attempt - 1)) ))  # 30s, 60s, 120s
  _log_status "ORCH_BACKOFF sleep=${_backoff}s"
  sleep "$_backoff"
  _attempt=$(( _attempt + 1 ))
done

_log_status "DRIVER_EXIT clean"

echo ""
echo "=== Loop ${LOOP_UUID} driver exited cleanly ==="
echo "  STATUS file : $STATUS_FILE"
echo "  Loop dir    : $LOOP_DIR"
echo ""
echo "Inspect the final result:"
echo "  cat $LOOP_DIR/story.md"
echo "  open $LOOP_DIR/foms.png"
