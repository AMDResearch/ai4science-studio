#!/usr/bin/env bash
# One-allocation campaign: ORBIT-2 2-node training + perf-report via LangGraph runner.
#
#   export AI4S_SHARED_DIR=/path/to/shared
#   export PERF_TOOLS_DIR=/path/to/perf-tools
#   export ORBIT2_ROOT=/path/to/ORBIT-2-clone        # optional, auto-detected otherwise
#   export ORBIT2_DATA_ROOT=/path/to/data/synthetic_scale
#   export ORBIT2_CONFIG_TEMPLATE=interm_8m_synthetic_train.yaml
#   # optional external LLM:
#   # export AI4S_LLM_ENDPOINT=https://.../v1
#   # export OPENAI_API_KEY=...
#   # or, relay-backed (see launch_lux_gateway.sh): AI4S_RELAY_CLIENT_ENV=~/.config/ai4s-relay/client.env
#   sbatch campaigns/orbit2-train-perf/sbatch_amd.sh
#
# The campaign runner uses srun inside this allocation; it never nests sbatch.
# Modeled directly on campaigns/orbit2-infer-perf/sbatch_amd.sh.

#SBATCH -J ai4s-orbit2-train-perf
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 01:30:00
#SBATCH -o ai4s-orbit2-train-perf-%j.out
#SBATCH -e ai4s-orbit2-train-perf-%j.out

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

# Smoke-test defaults (proven smoke-test defaults): the synthetic_scale NPZ data
# + tiny synthetic training template, so the campaign exercises the full
# Omnistat+FOM+perf-report path quickly. Override at submit time for a real
# dataset.
ORBIT2_DATA_ROOT="${ORBIT2_DATA_ROOT:-${AI4S_SHARED_DIR}/models/ORBIT-2/data/synthetic_scale}"
ORBIT2_CONFIG_TEMPLATE="${ORBIT2_CONFIG_TEMPLATE:-interm_8m_synthetic_train.yaml}"
ORBIT2_MAX_EPOCH="${ORBIT2_MAX_EPOCH:-4}"
ORBIT2_MAX_BATCHES="${ORBIT2_MAX_BATCHES:-5}"
ORBIT2_BATCH_SIZE="${ORBIT2_BATCH_SIZE:-2}"
# HSDP shape (proven defaults): shard within a node, replicate across nodes
# ("studio" entrypoint run_orbit2_train.py -> upstream intermediate_downscaling.py,
# "enter hybrid FSDP"). A prior version of this script defaulted to full-shard
# (fsdp=NODES*8, simple_ddp=1) based on a mistaken "hybrid-FSDP never completes"
# diagnosis — see the corrected entry in .cursor/skills/ai4science-studio/reference/rocm.md
# ("RCCL data plane"). Cross-node NCCL comm init can be intermittently slow
# independent of this fsdp/simple_ddp choice — if a run stalls, suspect transient
# network contention or Omnistat collector overhead before re-litigating the parallelism shape.
ORBIT2_FSDP="${ORBIT2_FSDP:-8}"
ORBIT2_SIMPLE_DDP="${ORBIT2_SIMPLE_DDP:-${SLURM_JOB_NUM_NODES:-2}}"
export ORBIT2_DATA_ROOT ORBIT2_CONFIG_TEMPLATE ORBIT2_MAX_EPOCH ORBIT2_MAX_BATCHES ORBIT2_BATCH_SIZE ORBIT2_FSDP ORBIT2_SIMPLE_DDP

# Optional site module — pass OMNISTAT_MODULEPATH and OMNISTAT_MODULE at submit time if your site provides Omnistat as a module.
# Batch shells launched from non-login sessions may lack the `module` function.
if [[ -n "${OMNISTAT_MODULE:-}" ]] && ! type module >/dev/null 2>&1; then
  set +eu; source /etc/profile >/dev/null 2>&1; set -eu  # /etc/profile is not set -eu clean
fi
if [[ -n "${OMNISTAT_MODULEPATH:-}" ]]; then
  module use "${OMNISTAT_MODULEPATH}"
fi
if [[ -n "${OMNISTAT_MODULE:-}" ]]; then
  module load "${OMNISTAT_MODULE}"
  echo "  Omnistat module: ${OMNISTAT_MODULE} (OMNISTAT_DIR=${OMNISTAT_DIR:-unset})"
  # The module prepends its python-libs onto PYTHONPATH. Those wheels do not
  # match the campaign venv (orjson import breaks LangGraph). Keep them only
  # for omnistat-usermode via OMNISTAT_PYTHONPATH.
  if [[ -n "${OMNISTAT_DIR:-}" ]]; then
    OMNISTAT_PYTHONPATH="$(dirname "$OMNISTAT_DIR")/python-libs"
    export OMNISTAT_PYTHONPATH
    if [[ -n "${PYTHONPATH:-}" && -d "$OMNISTAT_PYTHONPATH" ]]; then
      PYTHONPATH="$(echo ":${PYTHONPATH}:" | sed "s|:${OMNISTAT_PYTHONPATH}:|:|g; s|^:||; s|:$||")"
      export PYTHONPATH
    fi
  fi
fi

# Relay-backed LLM: read the relay token from a user-only file on the shared fs so the secret
# never travels through sbatch arguments or SLURM job environment.
if [[ -n "${AI4S_RELAY_CLIENT_ENV:-}" ]]; then
  set -a; source "$AI4S_RELAY_CLIENT_ENV"; set +a
  export OPENAI_API_KEY="${AI4S_RELAY_TOKEN:?AI4S_RELAY_TOKEN missing in $AI4S_RELAY_CLIENT_ENV}"
  unset AI4S_RELAY_TOKEN
fi

# Optional cluster registry (reusable agents only).
export AI4S_REGISTRY_DIR="${AI4S_REGISTRY_DIR:-${AI4S_SHARED_DIR}/registry}"

# Prefer an explicit venv (shared-fs install); fall back to python3 on PATH.
AI4S_PYTHON="${AI4S_PYTHON:-${AI4S_SHARED_DIR}/venvs/ai4s-campaign/bin/python}"
if [[ ! -x "$AI4S_PYTHON" ]]; then
  AI4S_PYTHON="${AI4S_PYTHON_FALLBACK:-python3}"
fi

RUN_DIR="${AI4S_CAMPAIGN_RUN_DIR:-${AI4S_SHARED_DIR}/campaign-runs/orbit2-train-perf-${SLURM_JOB_ID:-$$}}"
mkdir -p "$RUN_DIR"

echo "=== AI4S campaign orbit2-train-perf ==="
echo "  REPO_ROOT    : $REPO_ROOT"
echo "  RUN_DIR      : $RUN_DIR"
echo "  AI4S_PYTHON  : $AI4S_PYTHON"
echo "  Data root    : $ORBIT2_DATA_ROOT"
echo "  Config tmpl  : $ORBIT2_CONFIG_TEMPLATE"
echo "  Max epoch    : $ORBIT2_MAX_EPOCH  Max batches: $ORBIT2_MAX_BATCHES  Batch size: $ORBIT2_BATCH_SIZE"
echo "  FSDP shape   : fsdp=$ORBIT2_FSDP simple_ddp=$ORBIT2_SIMPLE_DDP (HSDP: shard within node, replicate across nodes)"
echo "  LLM endpoint : ${AI4S_LLM_ENDPOINT:-'(auto/local)'}"
echo "  OPENAI_API_KEY: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo set || echo unset)"
echo ""

SET_ARGS=(
  --set "AI4S_SHARED_DIR=${AI4S_SHARED_DIR}"
  --set "PERF_TOOLS_DIR=${PERF_TOOLS_DIR}"
  --set "ORBIT2_DATA_ROOT=${ORBIT2_DATA_ROOT}"
  --set "ORBIT2_CONFIG_TEMPLATE=${ORBIT2_CONFIG_TEMPLATE}"
  --set "ORBIT2_MAX_EPOCH=${ORBIT2_MAX_EPOCH}"
  --set "ORBIT2_MAX_BATCHES=${ORBIT2_MAX_BATCHES}"
  --set "ORBIT2_BATCH_SIZE=${ORBIT2_BATCH_SIZE}"
)
[[ -n "${ORBIT2_ROOT:-}" ]] && SET_ARGS+=(--set "ORBIT2_ROOT=${ORBIT2_ROOT}")
[[ -n "${ORBIT2_SIF:-}" ]] && SET_ARGS+=(--set "ORBIT2_SIF=${ORBIT2_SIF}")
[[ -n "${ORBIT2_OVERLAY:-}" ]] && SET_ARGS+=(--set "ORBIT2_OVERLAY=${ORBIT2_OVERLAY}")

"$AI4S_PYTHON" -m ai4s run "$SCRIPT_DIR/campaign.yaml" \
  --run-dir "$RUN_DIR" \
  "${SET_ARGS[@]}"
