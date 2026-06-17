#!/usr/bin/env bash
# validate_orbit2_optimizer_loop_recipe.sh — CI/repo smoke: required files exist.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
BASE="${REPO_ROOT}/earth_science/models/ORBIT-2"
REQ=(
  "${BASE}/recipes/perf-optimizer-loop/README.md"
  "${BASE}/recipes/perf-optimizer-loop/lever_catalog.yaml"
  "${BASE}/recipes/perf-optimizer-loop/pitfall-diagnosis.md"
  "${BASE}/recipes/perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md"
  "${BASE}/recipes/perf-optimizer-loop/agents/orchestrator.md"
  "${BASE}/recipes/perf-optimizer-loop/agents/lever_picker.md"
  "${BASE}/recipes/perf-optimizer-loop/agents/fom_extractor.md"
  "${BASE}/recipes/perf-optimizer-loop/agents/story_writer.md"
  "${BASE}/examples/run_optimizer_loop.sh"
  "${BASE}/examples/sweep_orbit2_batch_bf16_amd.sh"
  "${BASE}/examples/run_fom_extractor.py"
)
for f in "${REQ[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f" >&2
    exit 1
  fi
done
echo "ORBIT-2 perf-optimizer-loop recipe files: OK"
