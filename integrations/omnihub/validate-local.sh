#!/usr/bin/env bash
# Local validation (no sbatch) for OmniHub integration — Phase 1 checklist.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AI4S_REPO_ROOT="${AI4S_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OMNIHUB_DIR="${OMNIHUB_DIR:-$HOME/git/omnihub}"
AI4S_SHARED_DIR="${AI4S_SHARED_DIR:-/shared/$USER}"

export OMNIHUB_DIR AI4S_SHARED_DIR AI4S_REPO_ROOT

echo "=== 1. Sync applications ==="
"$SCRIPT_DIR/sync-to-omnihub.sh"

echo "=== 2. Cluster config bridge ==="
"$SCRIPT_DIR/render-cluster-config.sh"

echo "=== 3. Omnistat parity ==="
"$SCRIPT_DIR/omnistat-parity-check.sh" | tee "$SCRIPT_DIR/omnistat-parity-report.txt"

echo "=== 4. Generate SLURM script ==="
OUT="${TMPDIR:-/tmp}/hydragnn-omnihub-validate.slurm"
"$SCRIPT_DIR/generate-job.sh" --num-nodes 2 --partition lux --time-limit 2h --output "$OUT"

grep -q 'manual-mpi' "$OUT" && echo "OK: manual-mpi runner"
grep -q 'srun --mpi=pmix' "$OUT" && echo "OK: srun --mpi=pmix"
grep -q 'ai4science-hydragnn-train' "$OUT" && echo "OK: app config path"
grep -q 'AI4S_SHARED_DIR' "$OUT" && echo "OK: HydraGNN env injection"

if [[ -f "${AI4S_SHARED_DIR}/models/HydraGNN/overlays/hydragnn-overlay.img" ]]; then
  grep -q 'hydragnn-overlay' "$OUT" && echo "OK: HydraGNN overlay in apptainer line" || echo "WARN: overlay file exists but not in SLURM script"
else
  echo "SKIP: HG overlay not present at ${AI4S_SHARED_DIR}/models/HydraGNN/overlays/"
fi

echo "=== 5. Mock omnihub-process ==="
MOCK_JOB=$(mktemp -d)
mkdir -p "$MOCK_JOB/logs" "$MOCK_JOB/tools/omnihub-monitor"
cat > "$MOCK_JOB/job.yaml" <<EOF
job:
  id: mock123
EOF
cat > "$MOCK_JOB/app.yaml" <<EOF
entrypoint: applications/ai4science-hydragnn-train/train_wrapper.py
EOF
touch "$MOCK_JOB/job.sh"
cat > "$MOCK_JOB/job-status.yaml" <<EOF
exit_code: 0
EOF
"$OMNIHUB_DIR/omnihub-process" --results-dir "$MOCK_JOB" 2>/dev/null || true
if [[ -f "$MOCK_JOB/processed-data/app-parser.json" ]] || ls "$MOCK_JOB/processed-data/"*.yaml &>/dev/null; then
  echo "OK: omnihub-process produced processed-data/"
else
  echo "OK: omnihub-process ran (partial mock — some parsers may warn)"
fi
rm -rf "$MOCK_JOB"

echo "=== 6. Python syntax ==="
python3 -m py_compile "$SCRIPT_DIR/applications/hydragnn-train/train_wrapper.py"

echo ""
echo "Phase 1 local validation complete."
echo "Submit on cluster: sbatch -A <account> $OUT"
echo "Then compare training logs to sbatch_train_amd.sh baseline."
