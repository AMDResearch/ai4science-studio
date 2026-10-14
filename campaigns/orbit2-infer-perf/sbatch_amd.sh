#!/usr/bin/env bash
# One-allocation campaign: ORBIT-2 inference + perf-report via LangGraph runner.
#
#   export AI4S_SHARED_DIR=/path/to/shared
#   export PERF_TOOLS_DIR=/path/to/perf-tools
#   export ORBIT2_ROOT=/path/to/ORBIT-2-clone
#   # optional external LLM:
#   # export AI4S_LLM_ENDPOINT=https://.../v1
#   # export OPENAI_API_KEY=...
#   sbatch campaigns/orbit2-infer-perf/sbatch_amd.sh
#
# The campaign runner uses srun inside this allocation; it never nests sbatch.

#SBATCH -J ai4s-orbit2-infer-perf
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH -t 01:30:00
#SBATCH -o ai4s-orbit2-infer-perf-%j.out
#SBATCH -e ai4s-orbit2-infer-perf-%j.out

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _ORIG_CMD=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\(\S\+\).*/\1/p')
  SCRIPT_DIR=$(cd "$(dirname "$_ORIG_CMD")" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/tools${PYTHONPATH:+:$PYTHONPATH}"

: "${AI4S_SHARED_DIR:?AI4S_SHARED_DIR must be set}"
: "${PERF_TOOLS_DIR:?PERF_TOOLS_DIR must be set}"
: "${ORBIT2_ROOT:?ORBIT2_ROOT must be set}"

# Optional cluster registry (reusable agents only).
export AI4S_REGISTRY_DIR="${AI4S_REGISTRY_DIR:-${AI4S_SHARED_DIR}/registry}"

# Prefer an explicit venv (shared-fs install); fall back to python3 on PATH.
AI4S_PYTHON="${AI4S_PYTHON:-${AI4S_SHARED_DIR}/venvs/ai4s-campaign/bin/python}"
if [[ ! -x "$AI4S_PYTHON" ]]; then
  AI4S_PYTHON="${AI4S_PYTHON_FALLBACK:-python3}"
fi

RUN_DIR="${AI4S_CAMPAIGN_RUN_DIR:-${AI4S_SHARED_DIR}/campaign-runs/orbit2-infer-perf-${SLURM_JOB_ID:-$$}}"
mkdir -p "$RUN_DIR"

echo "=== AI4S campaign orbit2-infer-perf ==="
echo "  REPO_ROOT   : $REPO_ROOT"
echo "  RUN_DIR     : $RUN_DIR"
echo "  AI4S_PYTHON : $AI4S_PYTHON"
echo "  LLM endpoint: ${AI4S_LLM_ENDPOINT:-'(auto/local)'}"
echo "  OPENAI_API_KEY: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo set || echo unset)"
echo ""

"$AI4S_PYTHON" -m ai4s run "$SCRIPT_DIR/campaign.yaml" \
  --run-dir "$RUN_DIR" \
  --set "AI4S_SHARED_DIR=${AI4S_SHARED_DIR}" \
  --set "PERF_TOOLS_DIR=${PERF_TOOLS_DIR}" \
  --set "ORBIT2_ROOT=${ORBIT2_ROOT}"
