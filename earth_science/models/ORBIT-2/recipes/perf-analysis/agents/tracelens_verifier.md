# tracelens_verifier subagent

Independently re-derive the top 2-3 claims from `tracelens/claims.json` against the raw `.pt.trace.json` and (optionally) probe **one** cheap remedy on a 1-node interactive `srun`.

## Inputs

- `<perf_run_dir>/manifest.json`
- `<perf_run_dir>/tracelens/claims.json`
- The raw trace: the single `*.pt.trace.json` under `manifest.trace_dir` (find with `find "$(jq -r .trace_dir <manifest>)" -name '*.pt.trace.json' 2>/dev/null | sort | head -1`).

## Outputs

- `<perf_run_dir>/tracelens/verified_claims.json` — same schema as `claims.json` plus a `verdict` field per entry.

## Verdict schema

Each input claim is copied through with one new field:

```json
{
  "...all fields from claims.json...": "...",
  "verdict": "verified|refuted|inconclusive",
  "verifier_evidence": "<200 chars: how you re-derived it>",
  "verifier_value": <re-derived number>,
  "remedy_probe": {
    "ran": true|false,
    "command": "<srun command or null>",
    "baseline_seconds_per_step": <float|null>,
    "remedy_seconds_per_step": <float|null>,
    "delta_pct": <float|null>,
    "notes": "<short>"
  }
}
```

## Steps

### 1. Re-derive top 3 claims from the raw JSON

```python
import json, gzip
from pathlib import Path
def open_trace(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p, 'r')
with open_trace(trace) as f:
    data = json.load(f)
events = data["traceEvents"]
```

For each top claim, do a parallel re-derivation:

| Claim class | Re-derivation |
|---|---|
| `comm_xgmi` / `comm_scaleout` | sum `dur` for events whose `name` matches `^(nccl|rccl|ncclKernel|AllReduce|AllGather|ReduceScatter)`; divide by total span (`max(ts+dur)-min(ts)` of GPU events). Compare to claim's `magnitude.value`. |
| `gpu_compute` | filter `cat=='kernel'`, find top-N by `sum(dur)`; check the top kernel name + percent matches the claim. |
| `cpu_dispatch` | `(total_span - sum(GPU kernel dur)) / total_span` should match "idle %". |
| `dataloader` | sum `dur` of events with `name` containing `enumerate(DataLoader)`. |

If verifier_value within ±20% of claim's value → `verdict=verified`. If outside → `verdict=refuted` with the actual number. If we can't isolate the events → `verdict=inconclusive`.

### 2. Pick at most ONE remedy probe

Choose the highest-impact `verified` claim with a non-null `remedy_test_command_or_null`. Constraints:

- 1 node only (`-N1`).
- ≤5 minutes wall-time.
- Must produce a comparable number (steps/sec or seconds/step) to the original 2-node baseline. If the remedy can only be tested at scale (e.g. RCCL inter-node tweak), set `remedy_probe.ran=false; notes="remedy requires multi-node; deferred to next iteration"`.

Example probe for the bf16 vs float32 discriminator claim — run via the existing ORBIT-2 sbatch script in interactive mode:

```bash
srun -p <partition> -A <account> -N1 --ntasks=8 --gpus-per-node=8 --cpus-per-task=16 \
    --time=00:05:00 --pty bash -c '
export AI4S_SHARED_DIR=/path/to/shared
export ORBIT2_OUTPUT_DIR=/tmp/perf-probe-$$
mkdir -p "$ORBIT2_OUTPUT_DIR"
# Prefer submitting via sbatch_train_amd.sh from the repo; ad-hoc srun probes are site-specific.
bash "$REPO_ROOT/earth_science/models/ORBIT-2/examples/sbatch_train_amd.sh" \
    2>&1 | tee /tmp/probe-$SLURM_JOB_ID.out
grep -oE "[0-9.]+s/it" /tmp/probe-$SLURM_JOB_ID.out | tail -5
'
```

Note: `sbatch_train_amd.sh` is designed for `sbatch`, not `srun --pty`. For probes prefer using its rank-script logic directly via a small ad-hoc wrapper rather than running the whole sbatch under srun. The verifier may inline a minimal Python invocation that bypasses sbatch and just runs `intermediate_downscaling.py` (or `examples/run_orbit2_train.py`) for ~10 batches inside the container with the alternate dtype.

If the verifier judges that no cheap probe is feasible, set `remedy_probe.ran=false` with a clear note and proceed.

### 3. Emit `verified_claims.json`

Same length and order as `claims.json`. Even refuted claims stay in the file — the synthesizer will drop them, but they should remain visible.

### 4. Final stdout

```
STATUS=ok; reason=<n_verified>v/<n_refuted>r/<n_inconclusive>i; probe=<class|none>
```

## Constraints

- Never modify `claims.json` — only write `verified_claims.json`.
- Never run a 2-node `srun` or `sbatch` from this subagent.
- If the trace file is missing, write `verified_claims.json` as `[]` and `STATUS=partial; reason=no_trace`.

## Why two phases (analyst + verifier)?

Hallucination defense. The analyst can produce a number from a tool's report; the verifier re-derives it from raw events. Disagreement => the user gets to see both numbers and the synthesizer flags the conflict in the final report rather than silently picking one. This is the same approach the omnistat side uses with PromQL-from-curl re-derivation.
