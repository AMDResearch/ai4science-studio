# omnistat_verifier subagent

Independently re-derive the top 2-3 claims from `omnistat/claims.json` by issuing raw PromQL via `curl` against the same VictoriaMetrics endpoint, at the **finest sampling step**. Optionally probe one cheap remedy on a 1-node interactive `srun`.

## Inputs

- `<perf_run_dir>/manifest.json`
- `<perf_run_dir>/omnistat/claims.json`
- VictoriaMetrics already running (PID at `<perf_run_dir>/omnistat/vm.pid`, port from analyst output).
- `<perf_run_dir>/omnistat/inspect_outputs/job_info.json` (for time range and sampling interval).

## Outputs

- `<perf_run_dir>/omnistat/verified_claims.json` (same schema as `tracelens/verified_claims.json`).

## Steps

### 1. Re-derive top claims via raw PromQL

Read the time range from `job_info.json` (`start_time`, `end_time`, `sampling_interval`). For each top claim, issue a `query_range` at the finest step (`max(sampling_interval, runtime/90000)`).

Examples:

**Important:** `rocm_*` and `omnistat_host_*` metrics do **not** carry a `jobid` label directly. Filter by joining with `rmsjob_info` per-instance:

```promql
metric_name * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="$JOBID"}))
```

A naive `rocm_utilization_percentage{jobid="..."}` will return 0 series even though the data is there. Use the join in every query.

Also: `start_time`/`end_time` from `job_info.json` are **UTC**. Convert with `calendar.timegm(time.strptime(...))` (NOT `time.mktime`, which assumes local time) before passing to PromQL.

| Claim | PromQL |
|---|---|
| "Mean GPU util = X%" | `avg(rocm_utilization_percentage * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="$JOBID"})))` |
| "VRAM near 100% on N nodes" | `max(rocm_vram_used_percentage * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="$JOBID"})))` per `instance` |
| "Network tx rate plateau" | `rate(omnistat_network_tx_bytes * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="$JOBID"}))[60s])` peak |
| "XGMI imbalance" | per-card stddev of `rocm_xgmi_*_bytes` rate, joined as above |
| "Power throttling events" | sum increase of throttle counters over the job span, joined as above |

```bash
PORT=$(jq -r '.[0].port' <vm_info>)  # or pull from analyst stdout
curl -sG "http://127.0.0.1:$PORT/api/v1/query_range" \
    --data-urlencode "query=avg_over_time(rocm_utilization_percentage{jobid=\"$JOBID\"}[1m])" \
    --data-urlencode "start=$START" \
    --data-urlencode "end=$END" \
    --data-urlencode "step=$STEP" \
    | jq '.data.result | map(.values[][1] | tonumber) | add/length'
```

The verifier should re-derive the actual number from the PromQL result and compare to `claim.magnitude.value`. ±20% → `verdict=verified`. Outside → `refuted`. Empty result series → `inconclusive`.

### 2. Optional remedy probe

Same constraints as the tracelens verifier — at most one 1-node `srun -p <partition> -A <account> -N1 --time=00:05:00`. Telemetry-side probes are typically configuration changes (e.g. enable rocprofiler `hbm` counter set in the omnistat config and re-run for 1 minute on 1 node to see if HBM bandwidth is actually saturating).

If a probe needs the omnistat config to change, write a temp config file based on `omnistat.config` with the rocprofiler section uncommented, point `OMNISTAT_CONFIG` at it, and submit via the existing sbatch script as a tiny job with `ORBIT2_MAX_EPOCH=1 ORBIT2_MAX_BATCHES=10 --nodes=1`. Time-budget that branch generously (10 min) since it does include the omnistat collector startup.

If multi-node is required (e.g. ANP plugin retest), set `remedy_probe.ran=false; notes="multi-node probe deferred"`.

### 3. Emit `verified_claims.json`

Same shape as the tracelens version. Each claim copied through with `verdict`, `verifier_evidence`, `verifier_value`, `remedy_probe`.

### 4. Stop VictoriaMetrics

After all queries are done:

```bash
kill "$(cat <perf_run>/omnistat/vm.pid)" 2>/dev/null || true
```

The synthesizer doesn't need VM running.

### 5. Final stdout

```
STATUS=ok; reason=<n_verified>v/<n_refuted>r/<n_inconclusive>i; probe=<class|none>
```

## Failure modes

- VM not running: try to start it the same way the analyst did before giving up; if still down, STATUS=fail.
- Query returns no data for a key metric the claim relies on: that's an `inconclusive` verdict, not a fail.
- Network blip mid-query: retry once with 5s sleep.

## Notes

- Use **only** `curl` for re-derivation, **not** `omnistat-inspect`. The whole point is an independent path.
- Use the finest step that VM allows (`runtime / 90000` floor); document the actual step used in `verifier_evidence`.
- Never delete the DB or any analyst output.
