# fom_extractor subagent

Computes the six Figures of Merit for an iteration and writes the
TraceLens↔Omnistat 1-second-window correlation that subsequent analysts and
lever_picker use as evidence for per-kernel attribution (cheaper than the
omnistat kernel-trace collector).

## Inputs

- `<perf_run>/manifest.json` (from the existing launcher subagent)
- `<perf_run>/hydragnn-train-<jobid>.out` (training log; epoch summaries + tqdm)
- `<perf_run>/logs/*.pt.trace.json` (rank-0 PyTorch trace; may be a single file)
- `<perf_run>/omnistat-db/` (VictoriaMetrics datadir)
- `<perf_run>/tracelens/csvs/` (already produced by tracelens_analyst; we re-use `ops_summary.csv` etc., but we MUST NOT depend on tracelens_analyst running successfully — fall back to raw trace if csvs missing)
- A live VictoriaMetrics endpoint started by omnistat_analyst on this perf_run, OR start our own on a free port if not running

## Outputs

- `<perf_run>/foms.json` — schema below
- `<perf_run>/kernel_correlation.csv` — per-1s-window, per-card, with columns:
  `window_start_iso, window_start_ts, instance, gpu_id, top_kernel_name, top_kernel_busy_frac, second_kernel_name, second_kernel_busy_frac, mfma_tflops, hbm_read_GBps, mean_power_W`

## FOM schema (`foms.json`)

```json
{
  "iter": 1,
  "jobid": "7050",
  "lever": {"id": "batch_800", "env_diff": {"HG_BATCH_SIZE": "800"}},
  "runtime_s": 487.3,
  "num_epochs_completed": 3,
  "samples_processed": 384000,
  "foms": {
    "epoch_time_s": 145.3,
    "throughput_samples_per_s": 11020.0,
    "mfma_tflops": 0.687,
    "energy_J": 1.12e6,
    "mean_power_W": 4800.0,
    "energy_per_sample_J": 12.4,
    "final_loss": 5.21
  },
  "fom_sources": {
    "epoch_time_s": "parse_convergence.py over hydragnn-train-7050.out, mean(wall_time_s) for epoch>=1",
    "throughput_samples_per_s": "max_num_batch * batch_size * ranks / epoch_time_s",
    "mfma_tflops": "PromQL: avg(rate(SQ_INSTS_VALU_MFMA_MOPS_F64[10s]) * on(instance) group_left() max by(instance)(rmsjob_info{jobid='7050'})) * 4",
    "energy_J": "integral of rocm_avg_pwr over runtime_s, summed across 16 cards",
    "mean_power_W": "energy_J / runtime_s",
    "energy_per_sample_J": "energy_J / samples_processed",
    "final_loss": "last epoch_summary train_loss from convergence.csv"
  },
  "kernel_correlation_summary": {
    "windows_examined": 280,
    "windows_with_top_kernel_busy_frac_gt_0p5": 165,
    "top_kernel_dominants": [["aten::mm", 78], ["aten::bmm", 52], ["triton_red_fused_*", 35]],
    "median_mfma_tflops_when_aten_mm_dominant": 3.15,
    "median_mfma_tflops_when_aten_bmm_dominant": 0.019,
    "median_hbm_read_GBps_when_mul_dominant": 4.5,
    "attribution_quality": "good"
  }
}
```

`attribution_quality`: "good" if > 50% of windows have `top_kernel_busy_frac > 0.5`; "fair" if 25-50%; "poor" if < 25%. lever_picker uses "poor" as one of the gates for picking `kernel_trace_diag_only`.

## Steps

### 1. Sanity check + env

```bash
set -euo pipefail
. "${PERF_TOOLS_DIR}/perf-inspect/bin/activate"
PERF_RUN=$(jq -r .perf_run_dir <manifest>)
JOBID=$(jq -r .jobid <manifest>)
RUNTIME=$(jq -r .runtime_seconds <manifest>)
TRACE=$(jq -r '.trace_paths[0] // empty' <manifest>)
[[ -z "$TRACE" ]] && { echo "STATUS=partial; reason=no_trace; FOMs computed without correlation" >&2; }
```

### 2. Time-based FOMs from training log

Use the existing parser, extended with a `--json-foms` flag (see "parse_convergence.py extension" below):

```bash
python3 "$REPO_ROOT/material_science/models/HydraGNN/examples/parse_convergence.py" \
    --log "$PERF_RUN/hydragnn-train-$JOBID.out" \
    --output "$PERF_RUN/convergence.csv" \
    --json-foms "$PERF_RUN/_convergence_foms.json"
```

`_convergence_foms.json` is consumed below; it provides `mean_epoch_time_excluding_epoch_0`, `num_epochs_completed`, `final_train_loss`, `total_batches_observed`.

Compute:
- `epoch_time_s = mean_epoch_time_excluding_epoch_0`
- `samples_processed = num_epochs_completed * (max_num_batch * batch_size * ranks)` where the three multiplicand values come from `manifest.json`'s `hg_batch_size`, `hydragnn_max_num_batch`, `ranks`.
- `throughput_samples_per_s = samples_processed / (runtime_s)`. (Use runtime_s rather than mean_epoch_time × n_epochs because runtime includes warm-up, JIT, and finalization — the user wants real wall-time efficiency.)

### 3. Counter-based FOMs via PromQL

Start a local VictoriaMetrics if not already running:

```bash
PORT=8428
ss -lnt "sport = :$PORT" 2>/dev/null | grep -q LISTEN && PORT=8429
if ! curl -sf "http://127.0.0.1:$PORT/api/v1/status/tsdb" > /dev/null 2>&1; then
  nohup "${PERF_TOOLS_DIR}/victoriametrics/victoria-metrics-prod" \
      -storageDataPath="$PERF_RUN/omnistat-db" \
      -httpListenAddr=127.0.0.1:$PORT \
      -retentionPeriod=100y -fs.disableMmap \
      > "$PERF_RUN/vm_for_foms.log" 2>&1 &
  echo $! > "$PERF_RUN/vm_for_foms.pid"
  for i in {1..20}; do sleep 1; curl -sf "http://127.0.0.1:$PORT/api/v1/status/tsdb" > /dev/null && break; done
fi
TSDB_URL="http://127.0.0.1:$PORT"
```

For each FOM, run the indicated PromQL via `curl ${TSDB_URL}/api/v1/query` and parse the scalar.

| FOM | PromQL (substitute $JOBID) |
|---|---|
| `mfma_tflops` (per-card avg) | `avg(rate(SQ_INSTS_VALU_MFMA_MOPS_F64[10s]) * on(instance) group_left() max by(instance)(rmsjob_info{jobid="$JOBID"})) * 4 / 1e12` |
| `mean_power_W` (per-card mean) | `avg_over_time((rocm_avg_pwr * on(instance) group_left() max by(instance)(rmsjob_info{jobid="$JOBID"}))[${RUNTIME}s:])` |
| `energy_J` (total, all 16 cards) | `sum(avg_over_time((rocm_avg_pwr * on(instance) group_left() max by(instance)(rmsjob_info{jobid="$JOBID"}))[${RUNTIME}s:])) * ${RUNTIME}` |

The MFMA→TFLOP/s conversion uses the standard MFMA MOPS-to-FLOPS factor of 4 (each MFMA op is 4 FMAs). This matches the formula used by [omnistat_verifier.md](../perf-analysis/agents/omnistat_verifier.md) and the existing 6985 combined_report.

If a query returns `null` or empty data (rocprofiler counters were off, OR no rmsjob_info records exist), write `null` for that FOM and add an entry to `foms.json`'s `notes` field: `"mfma_tflops=null because rocprofiler appears disabled (check sbatch banner for enable_rocprofiler state)"`.

### 4. TraceLens↔Omnistat 1-second-window correlation

This is the headline "cheaper than kernel-trace" capability. Build `kernel_correlation.csv` as follows:

#### 4a. Parse the rank-0 trace into a flat kernel-event table

```python
import gzip, json, os, pathlib
def _open(p):
    return gzip.open(p, 'rt') if p.endswith('.gz') else open(p)

with _open(TRACE) as f:
    trace = json.load(f)

events = trace.get("traceEvents", [])
# Keep only GPU kernel dispatches; PyTorch kineto marks them ph='X' with cat=='kernel'
kernels = [(e['ts'], e['dur'], e['name'])
           for e in events
           if e.get('ph') == 'X' and e.get('cat') == 'kernel' and 'ts' in e and 'dur' in e]
# kineto ts is in microseconds from trace start; convert to UTC by adding trace's reference epoch
ref_us = int(trace.get('baseTimeNanoseconds', 0)) // 1000  # may be missing on older kineto
# Fallback: pull job_info.json from omnistat_analyst output; its start_time is UTC ISO
```

If `ref_us` is missing or zero, read `omnistat/inspect_outputs/job_info.json` (written by omnistat_analyst) and use its `start_time` as the trace reference. **Important:** that timestamp is naive UTC; convert with `calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))`, NOT `time.mktime` (see SKILL §11.7). The PyTorch trace's `displayTimeUnit` and per-event `ts` are microseconds since the kineto "profile start" anchor, which is typically the wallclock time the profiler armed — this is close-enough to the job's epoch start that 1-s windowing absorbs any drift.

#### 4b. Bucket kernels into 1-second windows (rank-0 view of GPU 0; we approximate "card" with rank-0's device)

```python
from collections import defaultdict, Counter
window_kernels = defaultdict(Counter)   # window_ts -> {kernel_name: total_dur_us}
for ts, dur, name in kernels:
    win = (ref_us + ts) // 1_000_000  # 1-s windows, unix epoch seconds
    window_kernels[win][name] += dur
```

For each window, compute `top_kernel_name`, `top_kernel_busy_frac = top_dur / 1_000_000`, `second_kernel_name`, `second_kernel_busy_frac` (cap fraction at 1.0).

#### 4c. Query Omnistat for per-window counter rates

For each window timestamp `win_ts` and each of the 16 cards:

```python
import requests
def q(promql, t):
    r = requests.get(f"{TSDB_URL}/api/v1/query",
                     params={"query": promql, "time": t}, timeout=10)
    r.raise_for_status()
    res = r.json()["data"]["result"]
    return {row["metric"].get("instance"): float(row["value"][1]) for row in res}

# All 16 instances at once via the join
mfma_q = (f'rate(SQ_INSTS_VALU_MFMA_MOPS_F64[10s]) * on(instance) group_left() '
          f'max by(instance)(rmsjob_info{{jobid="{JOBID}"}}) * 4 / 1e12')
hbm_q  = (f'rate(FETCH_SIZE[10s]) * on(instance) group_left() '
          f'max by(instance)(rmsjob_info{{jobid="{JOBID}"}}) / 1024 / 1024 / 1024 * 1024')
pwr_q  = (f'rocm_avg_pwr * on(instance) group_left() '
          f'max by(instance)(rmsjob_info{{jobid="{JOBID}"}})')

# Note: FETCH_SIZE units are KB (lesson R1, see SKILL §12 / R1 report); convert to GB/s by
# multiplying KB-per-second by 1024 then dividing by 1e9 -> simplifies to "* 1024 / 1e9".
```

For each `win_ts` in `sorted(window_kernels)`, query the 3 PromQL above. For each `(instance, gpu_id)` pair returned (omnistat tags each per-card metric with both `instance` (hostname) and `card` index), emit one row to `kernel_correlation.csv`.

**Performance:** even at 5-min runs that's ~300 windows × 3 queries = 900 PromQL hits. With `requests` keep-alive + `-search.disableCache` already off (we want cached scans here), this completes in 1-2 minutes. Cache responses with `functools.lru_cache` keyed on (promql, win_ts) to avoid duplicate queries within the same window.

#### 4d. Compute correlation summary

Aggregate to populate `kernel_correlation_summary` in `foms.json`:
- `windows_examined = len(window_kernels)`
- `windows_with_top_kernel_busy_frac_gt_0p5 = sum(1 for w in windows if top_busy_frac > 0.5)`
- `top_kernel_dominants` = `Counter(top_kernel_name for win in windows).most_common(5)` (paired with count)
- `median_mfma_tflops_when_aten_mm_dominant` = median MFMA TFLOP/s across rows where `top_kernel_name.startswith('aten::mm')` (or contains `gemm` for hipBLASLt internal names)
- `median_mfma_tflops_when_aten_bmm_dominant` = same for `aten::bmm` and Triton bmm names
- `median_hbm_read_GBps_when_mul_dominant` = same for `aten::mul`/elementwise patterns
- `attribution_quality` per the rule above.

### 5. Emit `foms.json`

Write `foms.json` and `kernel_correlation.csv`. Both atomically via tmpfile + rename.

### 6. Final stdout

```
STATUS=ok; reason=foms epoch_time=<x>s throughput=<y>/s mfma=<z>TFLOP/s correlation=<quality>
```

If trace is missing OR counter queries returned all nulls:

```
STATUS=partial; reason=<which_part_missing>
```

The orchestrator handles `partial` by recording the FOM row with `null`s and a note; the loop continues.

## Failure modes

| Failure | Action |
|---|---|
| `_convergence_foms.json` empty (job died before any epoch) | `epoch_time_s = null, final_loss = null`; STATUS=partial |
| `manifest.json` missing | STATUS=fail; orchestrator handles as analysis_fail |
| VictoriaMetrics won't start | log to vm_for_foms.log; emit STATUS=partial with counter FOMs null; correlation skipped |
| `rocm_avg_pwr` empty (power telemetry collector dropped data) | `energy_J = null, mean_power_W = null, energy_per_sample_J = null`; STATUS=partial |
| Trace too large to load (>4 GB JSON) | skip correlation, `attribution_quality = "skipped:trace_too_large"`; STATUS=partial |
| PyTorch kineto trace base epoch missing AND job_info.json missing | skip correlation (cannot align timestamps); STATUS=partial |

## Notes for the implementing agent

- This subagent runs on the login node, after omnistat_analyst has started VictoriaMetrics on this perf-run. If VM isn't already up, start a second instance on a different port — the omnistat datadir is read-only friendly under VM. Requires `AI4S_SHARED_DIR` and `REPO_ROOT` to be set.
- Do NOT call `omnistat-inspect`; that's the analyst's job. We go straight to PromQL here because we need finer-grained queries than what `omnistat-inspect` exposes.
- Read at most: manifest.json, hydragnn-train-<jobid>.out, the rank-0 trace, omnistat-db (via VM). Cap trace read at 4 GB.
- Write at most: foms.json, kernel_correlation.csv, vm_for_foms.{log,pid}.

## parse_convergence.py extension

The existing `examples/parse_convergence.py` is extended with `--json-foms <path>` that emits this aggregate:

```json
{
  "num_epochs_completed": 3,
  "mean_epoch_time_excluding_epoch_0": 145.3,
  "epoch_times_s": [156.7, 144.1, 146.5],
  "final_train_loss": 5.21,
  "final_val_loss": 5.08
}
```

`mean_epoch_time_excluding_epoch_0` is `mean(epoch_times_s[1:])` to drop the warm-up. If only 1 epoch ran, fall back to using it (and add `"warning": "single epoch, epoch_0 not dropped"` to the JSON).

`epoch_times_s` comes from the `epoch_summary` records' `wall_time_s` when at least 2 are populated (rare — HydraGNN typically attaches the timer only to the final epoch). Otherwise computed per-epoch from `tqdm_final` records as `rate_s × batches_per_epoch`. `samples_processed` is computed by the fom_extractor from the manifest (`num_epochs × max_num_batch × batch_size × ranks`), NOT from the log, because HydraGNN's rank-0 only emits one "Max memory allocated" line per job.
