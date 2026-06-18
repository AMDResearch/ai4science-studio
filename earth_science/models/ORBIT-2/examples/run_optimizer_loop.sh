#!/usr/bin/env bash
# run_optimizer_loop.sh — ORBIT-2 iterative sysopt loop driver (Bayes-CAST EDM, MI355X).
#
# Usage (tmux on login node):
#   export ANTHROPIC_API_KEY=...   # optional; only for Claude Code CLI driver
#   export AI4S_SHARED_DIR=...
#   export OMNIHUB_TOOLS_DIR=...
#   bash earth_science/models/ORBIT-2/examples/run_optimizer_loop.sh <loop-uuid> <n_iters> [--preflight-only]
#
# Graceful stop: touch $AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/loop-<uuid>/STOP
#
# See ../recipes/perf-optimizer-loop/README.md for artifact layout and agent flow.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <loop-uuid> <n_iters_budget> [--preflight-only]" >&2
  exit 1
fi

LOOP_UUID="$1"
N_ITERS="$2"
PREFLIGHT_ONLY=0
if [[ "${3:-}" == "--preflight-only" ]]; then
  PREFLIGHT_ONLY=1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
RECIPE_DIR="${REPO_ROOT}/earth_science/models/ORBIT-2/recipes/perf-optimizer-loop"
# Loop driver orchestrator (full multi-subagent TraceLens/Omnistat flow). Override with
# ORBIT2_ORCH_PROMPT to point at a different agent prompt.
ORCH_PROMPT="${ORBIT2_ORCH_PROMPT:-${RECIPE_DIR}/agents/orchestrator.md}"

: "${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}"
: "${OMNIHUB_TOOLS_DIR:?OMNIHUB_TOOLS_DIR must be set}"

ORBIT2_BASE="${AI4S_SHARED_DIR}/models/ORBIT-2"
PERF_RUNS_DIR="${ORBIT2_BASE}/perf-runs"
LOOP_DIR="${PERF_RUNS_DIR}/loop-${LOOP_UUID}"
STATUS_FILE="${LOOP_DIR}/STATUS.txt"

mkdir -p "$LOOP_DIR"

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

_log_status "LOOP_START uuid=${LOOP_UUID} n_iters_budget=${N_ITERS} driver=claude-code-cli script=$0 model=ORBIT-2"

PREFLIGHT_FAIL_REASON=""
PREFLIGHT_NOTES=()

if command -v sinfo > /dev/null 2>&1; then
  _IDLE_OR_MIX=$(sinfo -h -o '%t' 2>/dev/null | grep -cE '^(idle|mix|alloc)$' || true)
  if [[ "${_IDLE_OR_MIX:-0}" -eq 0 ]]; then
    PREFLIGHT_FAIL_REASON="cluster_down"
    PREFLIGHT_NOTES+=("sinfo reports 0 usable nodes")
  fi
else
  PREFLIGHT_NOTES+=("sinfo not in PATH — cluster checks skipped")
fi

if [[ -z "$PREFLIGHT_FAIL_REASON" ]]; then
  _USE_PCT=$(df -P "$PERF_RUNS_DIR" 2>/dev/null | awk 'NR==2 {sub("%","",$5); print $5}')
  if [[ -n "${_USE_PCT:-}" && "${_USE_PCT:-0}" -gt 95 ]]; then
    PREFLIGHT_FAIL_REASON="disk_full"
    PREFLIGHT_NOTES+=("perf-runs filesystem at ${_USE_PCT}%")
  fi
fi

for _bin in \
    "${OMNIHUB_TOOLS_DIR}/omnihub-inspect/bin/omnistat-usermode" \
    "${OMNIHUB_TOOLS_DIR}/victoriametrics/victoria-metrics-prod"; do
  if [[ -z "$PREFLIGHT_FAIL_REASON" && ! -x "$_bin" ]]; then
    PREFLIGHT_FAIL_REASON="tool_missing"
    PREFLIGHT_NOTES+=("missing: $_bin")
  fi
done

for _f in \
    "${ORBIT2_BASE}/overlays/orbit2-overlay.img" \
    "${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif"; do
  if [[ -z "$PREFLIGHT_FAIL_REASON" && ! -f "$_f" ]]; then
    PREFLIGHT_FAIL_REASON="image_missing"
    PREFLIGHT_NOTES+=("missing: $_f")
  fi
done

if [[ -z "$PREFLIGHT_FAIL_REASON" && ! -f "$ORCH_PROMPT" ]]; then
  PREFLIGHT_FAIL_REASON="orch_prompt_missing"
  PREFLIGHT_NOTES+=("missing: $ORCH_PROMPT")
fi

if [[ -n "$PREFLIGHT_FAIL_REASON" ]]; then
  _log_status "PREFLIGHT_FAIL reason=${PREFLIGHT_FAIL_REASON} notes='${PREFLIGHT_NOTES[*]}'"
  echo "Pre-flight failed: ${PREFLIGHT_FAIL_REASON}" >&2
  printf '  - %s\n' "${PREFLIGHT_NOTES[@]}" >&2
  exit 2
fi

_log_status "PREFLIGHT_OK orch_prompt=ok"

if [[ $PREFLIGHT_ONLY -eq 1 ]]; then
  _log_status "PREFLIGHT_ONLY exiting"
  echo "Pre-flight OK; orchestrator not invoked."
  exit 0
fi

if ! command -v claude > /dev/null 2>&1; then
  echo "NOTE: 'claude' CLI not found — run the orchestrator manually in Cursor:" >&2
  echo "  Read: $ORCH_PROMPT" >&2
  echo "  Loop dir: $LOOP_DIR" >&2
  _log_status "LOOP_ABORT reason=no_claude_cli manual_orchestrator_required"
  exit 0
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY not set; cannot invoke claude non-interactively." >&2
  _log_status "LOOP_ABORT reason=no_api_key"
  exit 2
fi

_on_signal() {
  _log_status "LOOP_ABORT reason=signal sig=$1"
  exit 130
}
trap '_on_signal INT' INT
trap '_on_signal TERM' TERM

USER_PROMPT="Drive the ORBIT-2 iterative sysopt loop (Bayes-CAST EDM, throughput primary FOM):
- Loop UUID: ${LOOP_UUID}
- Loop directory: ${LOOP_DIR}
- Iterations budget: ${N_ITERS}
- Status file: ${STATUS_FILE}
- Recipe README: ${RECIPE_DIR}/README.md

Follow ${ORCH_PROMPT}. Persist state to disk for resume."

# Model for the orchestrator. The `opus` alias still points to claude-opus-4-7 on
# CLI 2.1.x, so default to the explicit 4.8 slug; override with ORBIT2_CLAUDE_MODEL.
ORBIT2_CLAUDE_MODEL="${ORBIT2_CLAUDE_MODEL:-claude-opus-4-8}"
_log_status "ORCH_MODEL model=${ORBIT2_CLAUDE_MODEL}"

_MAX_ATTEMPTS=3
_attempt=1
while [[ $_attempt -le $_MAX_ATTEMPTS ]]; do
  _log_status "ORCH_INVOKE attempt=${_attempt} model=${ORBIT2_CLAUDE_MODEL}"
  set +e
  claude \
      --print \
      --model "$ORBIT2_CLAUDE_MODEL" \
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
    _log_status "LOOP_ABORT reason=orchestrator_failed rc=${_rc}"
    exit 1
  fi
  _backoff=$(( 30 * (2 ** (_attempt - 1)) ))
  sleep "$_backoff"
  _attempt=$(( _attempt + 1 ))
done

_log_status "DRIVER_EXIT clean"
echo "=== ORBIT-2 loop driver done ==="
echo "  $LOOP_DIR"