# launcher subagent

Submits a 2-node HydraGNN training on Lux with PyTorch profiling and Omnistat user-mode telemetry, waits for completion, and writes `manifest.json` for downstream subagents.

## Inputs

- Repo at `/home/aaji/git/ai4science-studio` on branch `aaji/perf-agents-hydragnn`.
- Cluster config at `.cluster-config.yaml` (partition `lux`, account `vultr_lux`).
- Optional env-var overrides: `HG_BATCH_SIZE`, `HYDRAGNN_MAX_NUM_BATCH`, `HG_NUM_EPOCH`, `HG_PRECISION`, `PROFILE_TARGET_EPOCH`.

## Outputs

- `/shared/aaji/tools/omnistat-pr271/` — Python venv with omnistat (PR #271 + main merged) and TraceLens.
- `/shared/aaji/tools/victoriametrics/victoria-metrics-prod` — VictoriaMetrics binary.
- `/shared/aaji/models/HydraGNN/perf-runs/omnistat-lux.config` — Omnistat user-mode config.
- `/shared/aaji/models/HydraGNN/perf-runs/<jobid>/manifest.json` — manifest schema below.
- `/shared/aaji/models/HydraGNN/perf-runs/<jobid>/hydragnn-train-<jobid>.out` — symlinked from output dir.
- `/shared/aaji/models/HydraGNN/perf-runs/<jobid>/logs/` — symlinked rank-0 trace.
- `/shared/aaji/models/HydraGNN/perf-runs/<jobid>/omnistat-db/` — VictoriaMetrics datadir.

## Manifest schema

```json
{
  "jobid": "<int as str>",
  "submitted_at": "ISO 8601 UTC",
  "completed_at": "ISO 8601 UTC",
  "exit_state": "COMPLETED|FAILED|TIMEOUT|...",
  "nodes": 2,
  "gpus_per_node": 8,
  "ranks": 16,
  "nodelist": "lux-mi355x-aN,lux-mi355x-bM",
  "partition": "lux",
  "account": "vultr_lux",
  "runtime_seconds": <float>,
  "config_used": "<path to gfm_mlip_with_profile.json>",
  "profile_target_epoch": <int>,
  "trace_paths": ["<path to .pt.trace.json>"],
  "omnistat_db_path": "<dir>",
  "training_log_path": "<path>",
  "perf_run_dir": "<dir>",
  "hg_precision": "fp64",
  "hg_batch_size": 200,
  "hg_num_epoch": 2,
  "hydragnn_max_num_batch": 30,
  "tools_versions": {
    "omnistat_commit": "<sha>",
    "tracelens_commit": "<sha>",
    "victoriametrics_version": "<v>"
  }
}
```

## Steps

### 1. Lazy install tools (idempotent)

```bash
set -euo pipefail
TOOLS=/shared/aaji/tools
mkdir -p "$TOOLS"

# 1a. Omnistat: jorda/skills branch with origin/main merged in
SRC=$TOOLS/omnistat-src
if [[ ! -d $SRC/.git ]]; then
  git clone https://github.com/ROCm/omnistat.git "$SRC"
fi
cd "$SRC"
git fetch --all --prune
git checkout jorda/skills 2>/dev/null || git checkout -b jorda/skills origin/jorda/skills
git reset --hard origin/jorda/skills
if ! git merge --no-edit origin/main; then
  echo "ERROR: merging origin/main into jorda/skills produced conflicts; resolve manually" >&2
  exit 2
fi
OMNISTAT_COMMIT=$(git rev-parse HEAD)

# 1b. Venv (py3.12 to match cluster python)
VENV=$TOOLS/omnistat-pr271
if [[ ! -d $VENV ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "$SRC[query]"

# 1c. TraceLens (same venv)
"$VENV/bin/pip" install --quiet "git+https://github.com/AMD-AGI/TraceLens.git"
TRACELENS_COMMIT=$("$VENV/bin/python" -c "import TraceLens, importlib.metadata; print(importlib.metadata.version('TraceLens'))")

# 1d. VictoriaMetrics binary
VM_DIR=$TOOLS/victoriametrics
if [[ ! -x $VM_DIR/victoria-metrics-prod ]]; then
  mkdir -p "$VM_DIR"
  cd "$VM_DIR"
  VM_VERSION=$(curl -fsSL https://api.github.com/repos/VictoriaMetrics/VictoriaMetrics/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
  wget -q "https://github.com/VictoriaMetrics/VictoriaMetrics/releases/download/${VM_VERSION}/victoria-metrics-linux-amd64-${VM_VERSION}.tar.gz"
  tar xzf "victoria-metrics-linux-amd64-${VM_VERSION}.tar.gz"
  echo "$VM_VERSION" > VERSION
fi
VM_VERSION=$(cat $VM_DIR/VERSION)
```

### 2. Author the Omnistat user-mode config (once, in repo `perf-runs/`)

If `/shared/aaji/models/HydraGNN/perf-runs/omnistat-lux.config` doesn't exist, write it from the template at `/home/aaji/git/ai4science-studio/material_science/models/HydraGNN/recipes/perf-analysis/omnistat-lux.config.template`. The template uses `%(SLURM_JOB_ID)s` placeholders that omnistat-usermode resolves at runtime.

### 3. Generate the per-job profile config

Read the upstream `gfm_mlip.json` from `/shared/aaji/models/HydraGNN/code/HydraGNN/examples/multidataset_hpo_sc26/gfm_mlip.json` and inject the `Profile` block **inside `NeuralNetwork`** (not at the top level — `train_validate_test()` is invoked with `config["NeuralNetwork"]` as `config`, so its Profiler reads `config["Profile"]` from that scope):

```python
cfg.setdefault("NeuralNetwork", {})["Profile"] = {
    "enable": 1,
    "target_epoch": int(os.environ.get("PROFILE_TARGET_EPOCH", "1")),
}
```

Write the result to `/shared/aaji/models/HydraGNN/perf-runs/<jobid>/gfm_mlip_profile.json`. Use Python (`json.load`/`json.dump`) — do NOT do this with sed.

(The wrapper `examples/sbatch_train_perf_amd.sh` already does this; this step exists in the launcher prompt for the case where the launcher is reused for a non-HydraGNN model.)

### 4. Submit the job

```bash
export AI4S_SHARED_DIR=/shared/aaji
export HG_OUTPUT_DIR=/shared/aaji/models/HydraGNN/perf-runs/PENDING-$$
mkdir -p "$HG_OUTPUT_DIR"

cd /home/aaji/git/ai4science-studio
OUT=$(sbatch \
  --partition=lux \
  --account=vultr_lux \
  --nodes=2 \
  --ntasks-per-node=8 \
  --gpus-per-node=8 \
  --cpus-per-task=16 \
  --time=00:30:00 \
  --output="${HG_OUTPUT_DIR}/hydragnn-train-%j.out" \
  --error="${HG_OUTPUT_DIR}/hydragnn-train-%j.out" \
  material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh)
JOBID=$(echo "$OUT" | awk '{print $NF}')
echo "Submitted JOBID=$JOBID"
```

The sbatch wrapper inherits these env vars and exports `HG_CONFIG_OVERRIDE`, `OMNISTAT_VENV`, `OMNISTAT_CONFIG`, `OMNISTAT_USERMODE_INTERVAL`, `PROFILE_RANK0_ONLY=1` to the rank script.

### 5. Wait for terminal state

Poll `sacct -j $JOBID -X -n --format=State,ExitCode,Elapsed,NodeList -P` every 30 s. Treat any of `COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY` as terminal. **Hard ceiling 45 minutes** of polling — if not terminal, write `state=TIMEOUT_WAITING` to manifest and exit 1.

### 6. Move artifacts into final perf-run dir

Once terminal:

```bash
PERF_RUN=/shared/aaji/models/HydraGNN/perf-runs/${JOBID}
mv "$HG_OUTPUT_DIR" "$PERF_RUN"
# Find the rank-0 trace
TRACE=$(find "$PERF_RUN/logs" -name '*.pt.trace.json*' 2>/dev/null | head -1 || true)
# Find the omnistat DB (the sbatch wrapper writes it under the perf-run dir)
DB="$PERF_RUN/omnistat-db"
```

### 7. Write the manifest

Use Python in the omnistat venv to write the manifest JSON exactly per the schema above. Include all `tools_versions` captured in step 1.

### 8. Final stdout

Print:
```
STATUS=ok; reason=jobid=<jobid> state=<state> runtime=<sec>s perf_run=<path>
```

## Failure modes (must surface, never silently swallow)

| Failure | Action |
|---|---|
| Merge conflict in step 1 | Exit 2 with the message in the script. |
| `sbatch` returns nonzero | Exit 3, print sbatch stderr verbatim. |
| Job state == `FAILED` | Still write the manifest (with `exit_state=FAILED`), but **also** print the last 50 lines of the SLURM out to help downstream subagents skip cleanly. |
| No trace file found | Manifest gets `trace_paths=[]`. tracelens_analyst will gracefully no-op. |
| omnistat-db empty | Manifest still written; omnistat_analyst will detect and report it. |

## Notes for the implementing agent

- The launcher subagent runs on the **login node**. Do not call `srun` from inside the launcher; only `sbatch`.
- Do not import torch in the launcher's venv — it should stay lightweight.
- All paths absolute, never relative.
- Use `flock` on the install step if multiple invocations may race: `( flock -x 9; install_steps; ) 9>$TOOLS/.install.lock`.
