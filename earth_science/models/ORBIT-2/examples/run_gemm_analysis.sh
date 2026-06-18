#!/usr/bin/env bash
# run_gemm_analysis.sh — unattended ORBIT-2 GEMM-time bottleneck analysis (1-node + 2-node).
#
# Launches the full TraceLens + Omnistat analyst/verifier flow via the Claude Code CLI to answer
# "where does ORBIT-2's GEMM time actually go?" at 1 and 2 nodes. Set EXCLUDE_NODES to skip any
# known-bad nodes. Runs in tmux; resumable; disk-guarded.
#
# Usage (tmux on login node):
#   export ANTHROPIC_API_KEY=...   export AI4S_SHARED_DIR=<shared-root>
#   export OMNIHUB_TOOLS_DIR=<perf-tools-dir>   # omnihub.tools_dir in .cluster-config.yaml
#   export EXCLUDE_NODES=...                  # optional: comma-separated nodes to skip
#   bash earth_science/models/ORBIT-2/examples/run_gemm_analysis.sh <uuid> [--preflight-only]
#
# Graceful stop: touch <analysis-dir>/STOP
set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: $0 <uuid> [--preflight-only]" >&2; exit 1; }
UUID="$1"; PREFLIGHT_ONLY=0
[[ "${2:-}" == "--preflight-only" ]] && PREFLIGHT_ONLY=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
ORCH_PROMPT="${REPO_ROOT}/earth_science/models/ORBIT-2/recipes/perf-analysis/agents/orchestrator_gemm_analysis.md"

: "${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}"
: "${OMNIHUB_TOOLS_DIR:?OMNIHUB_TOOLS_DIR must be set}"
export AI4S_SHARED_DIR OMNIHUB_TOOLS_DIR

ORBIT2_BASE="${AI4S_SHARED_DIR}/models/ORBIT-2"
PERF_RUNS_DIR="${ORBIT2_BASE}/perf-runs"
ANALYSIS_DIR="${PERF_RUNS_DIR}/gemm-analysis-${UUID}"
STATUS_FILE="${ANALYSIS_DIR}/STATUS.txt"
REPORT="${ANALYSIS_DIR}/GEMM_TIME_REPORT.md"
ORBIT2_CLAUDE_MODEL="${ORBIT2_CLAUDE_MODEL:-claude-opus-4-8}"
DISK_GUARD_PCT="${ORBIT2_DISK_GUARD_PCT:-93}"
mkdir -p "$ANALYSIS_DIR"

_log() {
  local line; line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $1"
  ( flock -x 9; echo "$line" >> "$STATUS_FILE"; ) 9>>"$STATUS_FILE"
  echo "[GEMM] $line"
}
_log "ANALYSIS_START uuid=${UUID} dir=${ANALYSIS_DIR} model=ORBIT-2"

# ---------------------------------------------------------------- preflight ---
_fail=""; _notes=()
if command -v sinfo >/dev/null 2>&1; then
  _n=$(sinfo -h -o '%t' 2>/dev/null | grep -cE '^(idle|mix|alloc)$' || true)
  [[ "${_n:-0}" -eq 0 ]] && { _fail="cluster_down"; _notes+=("0 usable nodes"); }
else _notes+=("sinfo not in PATH"); fi
_pct=$(df -P "$PERF_RUNS_DIR" 2>/dev/null | awk 'NR==2 {sub("%","",$5); print $5}')
[[ -z "$_fail" && "${_pct:-0}" -ge 95 ]] && { _fail="disk_full"; _notes+=("perf-runs at ${_pct}%"); }
VENV="${OMNIHUB_TOOLS_DIR}/omnihub-inspect"
for _f in "$ORCH_PROMPT" \
          "${SCRIPT_DIR}/sbatch_train_perf_amd.sh" \
          "${VENV}/bin/omnistat-usermode" \
          "${OMNIHUB_TOOLS_DIR}/victoriametrics/victoria-metrics-prod" \
          "${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif"; do
  [[ -z "$_fail" && ! -e "$_f" ]] && { _fail="missing_file"; _notes+=("missing: $_f"); }
done
# TraceLens importable?
if [[ -z "$_fail" ]] && ! "${VENV}/bin/python" -c "import TraceLens" >/dev/null 2>&1; then
  _fail="tracelens_missing"; _notes+=("TraceLens not importable in ${VENV}")
fi
for _a in tracelens_analyst tracelens_verifier omnistat_analyst omnistat_verifier synthesizer; do
  _p="${REPO_ROOT}/earth_science/models/ORBIT-2/recipes/perf-analysis/agents/${_a}.md"
  [[ -z "$_fail" && ! -f "$_p" ]] && { _fail="missing_subagent"; _notes+=("missing: $_p"); }
done
if [[ -n "$_fail" ]]; then
  _log "PREFLIGHT_FAIL reason=${_fail} notes='${_notes[*]}'"
  printf 'Pre-flight failed: %s\n' "$_fail" >&2; printf '  - %s\n' "${_notes[@]}" >&2; exit 2
fi
_log "PREFLIGHT_OK notes='${_notes[*]:-none}'"
[[ $PREFLIGHT_ONLY -eq 1 ]] && { _log "PREFLIGHT_ONLY exiting"; echo "Pre-flight OK."; exit 0; }

command -v claude >/dev/null 2>&1 || { _log "ANALYSIS_ABORT reason=no_claude_cli"; echo "claude CLI not found." >&2; exit 2; }
[[ -n "${ANTHROPIC_API_KEY:-}" ]] || { _log "ANALYSIS_ABORT reason=no_api_key"; echo "ANTHROPIC_API_KEY unset." >&2; exit 2; }

# ------------------------------------------------------- disk-guard watchdog --
_disk_guard() {
  while [[ ! -f "${ANALYSIS_DIR}/STOP" && ! -f "${ANALYSIS_DIR}/DONE" ]]; do
    local pct; pct=$(df -P "$PERF_RUNS_DIR" 2>/dev/null | awk 'NR==2 {sub("%","",$5); print $5}')
    # NOTE: never delete traces here — the analysts need them. Only warn.
    [[ -n "${pct:-}" && "${pct}" -ge "$DISK_GUARD_PCT" ]] && _log "DISK_WARN fs=${pct}%"
    sleep 300
  done
}
_disk_guard & _GUARD_PID=$!
trap '_log "ANALYSIS_SIGNAL cleanup"; kill $_GUARD_PID 2>/dev/null || true' EXIT INT TERM

# ------------------------------------------------------- invoke orchestrator --
USER_PROMPT="Run the ORBIT-2 GEMM-time bottleneck analysis (TraceLens + Omnistat analyst/verifier) at 1 and 2 nodes. Follow ${ORCH_PROMPT}.
REPO_ROOT=${REPO_ROOT}
AI4S_SHARED_DIR=${AI4S_SHARED_DIR}
OMNIHUB_TOOLS_DIR=${OMNIHUB_TOOLS_DIR}
ANALYSIS_DIR=${ANALYSIS_DIR}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
Write GEMM_TIME_REPORT.md to ANALYSIS_DIR. Persist state for resume."

_MAX=10; i=1
while [[ ! -f "$REPORT" ]]; do
  [[ -f "${ANALYSIS_DIR}/STOP" ]] && { _log "ANALYSIS_ABORT reason=stop_flag"; break; }
  [[ $i -gt $_MAX ]] && { _log "ANALYSIS_GIVEUP after ${_MAX} invocations"; break; }
  _log "ORCH_INVOKE attempt=${i} model=${ORBIT2_CLAUDE_MODEL}"
  set +e
  claude --print --model "$ORBIT2_CLAUDE_MODEL" --dangerously-skip-permissions \
      --max-turns 300 --append-system-prompt "$(cat "$ORCH_PROMPT")" "$USER_PROMPT"
  rc=$?
  set -e
  _log "ORCH_RETURN attempt=${i} rc=${rc} report=$([[ -f "$REPORT" ]] && echo yes || echo no)"
  [[ -f "$REPORT" ]] && break
  sleep $(( 30 * i )); i=$(( i + 1 ))
done

if [[ -f "$REPORT" ]]; then
  touch "${ANALYSIS_DIR}/DONE"
  _log "ANALYSIS_COMPLETE report=${REPORT}"
  echo "=== GEMM-time analysis complete ==="; echo "  report: ${REPORT}"
else
  _log "ANALYSIS_INCOMPLETE no report after ${_MAX} attempts"; exit 1
fi