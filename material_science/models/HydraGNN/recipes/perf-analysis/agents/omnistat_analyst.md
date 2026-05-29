# omnistat_analyst subagent

Drive `omnistat-inspect` (PR #271) through the analyze-job phases on the user-mode VictoriaMetrics database, then map findings to the bottleneck taxonomy.

## Inputs

- `<perf_run_dir>/manifest.json`
- The Omnistat DB at `manifest.omnistat_db_path`.
- The `omnihub-inspect` venv at `${OMNIHUB_TOOLS_DIR}/omnihub-inspect/`.
- The VictoriaMetrics binary at `${OMNIHUB_TOOLS_DIR}/victoriametrics/victoria-metrics-prod`.

## Outputs

- `<perf_run_dir>/omnistat/inspect_outputs/db_info.json`
- `<perf_run_dir>/omnistat/inspect_outputs/job_info.json`
- `<perf_run_dir>/omnistat/inspect_outputs/data_check.json`
- `<perf_run_dir>/omnistat/inspect_outputs/health.json`
- `<perf_run_dir>/omnistat/inspect_outputs/stats_global.json`
- `<perf_run_dir>/omnistat/inspect_outputs/stats_<category>_<level>.json` — for any anomalous category, drilled to `node` and `gpu-id`/`interface-id`.
- `<perf_run_dir>/omnistat/report_summary.md`
- `<perf_run_dir>/omnistat/claims.json` (same schema as `tracelens/claims.json`)

## Steps

### 1. Start VictoriaMetrics on the DB (login node)

Following the [load-database SKILL](https://github.com/ROCm/omnistat/blob/jorda/skills/skills/load-database/SKILL.md), with one Lux-specific addition: pass `-fs.disableMmap` so VM can open a job-scoped DB inside the login-node cgroup (without it, even a 1.6 MB DB panics with "cannot mmap file with size 4096 bytes ... no such device").

```bash
DB=$(jq -r .omnistat_db_path <manifest>)
[[ -d "$DB/data" ]] || { echo "STATUS=fail; reason=no_db_data"; exit 1; }

# Pick free port
PORT=8428
ss -lnt "sport = :$PORT" 2>/dev/null | grep -q LISTEN && PORT=8429

VMLOG=$PERF_RUN/omnistat/vm.log
mkdir -p "$PERF_RUN/omnistat/inspect_outputs"
nohup ${OMNIHUB_TOOLS_DIR}/victoriametrics/victoria-metrics-prod \
    -storageDataPath="$DB" \
    -httpListenAddr=127.0.0.1:$PORT \
    -retentionPeriod=100y \
    -search.disableCache \
    -search.latencyOffset=0 \
    -search.maxPointsPerTimeseries=90000 \
    -fs.disableMmap \
    > "$VMLOG" 2>&1 &
VM_PID=$!
echo $VM_PID > "$PERF_RUN/omnistat/vm.pid"
TSDB_URL="http://127.0.0.1:$PORT"

# Wait up to 15 s for ready
for i in {1..15}; do
  sleep 1
  curl -sf "$TSDB_URL/api/v1/status/tsdb" > /dev/null && break
done

# Trap-style cleanup is the orchestrator's job; this subagent leaves VM running
# so the omnistat_verifier can hit the same endpoint.
```

### 2. Run analyze-job phases (per the SKILL)

```bash
. ${OMNIHUB_TOOLS_DIR}/omnihub-inspect/bin/activate
SCRATCH=$PERF_RUN/omnistat/scratch
mkdir -p "$SCRATCH"
JOBID=$(jq -r .jobid <manifest>)

OUTD=$PERF_RUN/omnistat/inspect_outputs

omnistat-inspect --tsdb-url $TSDB_URL db info > $OUTD/db_info.json
omnistat-inspect --tsdb-url $TSDB_URL --scratch-dir $SCRATCH job $JOBID info > $OUTD/job_info.json
omnistat-inspect --tsdb-url $TSDB_URL --scratch-dir $SCRATCH job $JOBID data-check > $OUTD/data_check.json
omnistat-inspect --tsdb-url $TSDB_URL --scratch-dir $SCRATCH job $JOBID health > $OUTD/health.json
omnistat-inspect --tsdb-url $TSDB_URL --scratch-dir $SCRATCH job $JOBID stats --level global > $OUTD/stats_global.json
```

### 3. Identify anomalous categories and drill down

For each category in `stats_global.json` (`gpu`, `host`, `network`, `xgmi`, `vendor`):

- Compute coefficient of variation (`stddev/mean`) for utilization-like gauges.
- Flag a category as anomalous if **any** of:
  - Mean GPU utilization < 60% (high cpu_dispatch / dataloader / comm risk)
  - Network rx/tx rate stddev/mean > 0.3 (interface imbalance)
  - VRAM stddev/mean > 0.2 (workload heterogeneity or memory leak)
  - HBM throttle / power throttle counts > 0
  - XGMI traffic stddev/mean > 0.3 (unbalanced intra-node comm)

For each flagged category, run `--level node` and either `--level gpu-id` (for `gpu`/`xgmi`) or `--level interface-id` (for `network`):

```bash
for cat in gpu network xgmi host vendor; do
  if is_anomalous "$cat"; then
    omnistat-inspect --tsdb-url $TSDB_URL --scratch-dir $SCRATCH job $JOBID stats --category $cat --level node > $OUTD/stats_${cat}_node.json
    case $cat in
      gpu|xgmi)    fine=gpu-id ;;
      network)     fine=interface-id ;;
      *)           fine="" ;;
    esac
    [[ -n "$fine" ]] && omnistat-inspect --tsdb-url $TSDB_URL --scratch-dir $SCRATCH job $JOBID stats --category $cat --level $fine > $OUTD/stats_${cat}_${fine//-/_}.json
  fi
done
```

### 4. Translate findings into claims

Concrete rules for the first iteration:

- Mean GPU util < 50% → `class=cpu_dispatch` (or `dataloader`, lower confidence) — magnitude = `1 - mean_util`.
- HBM `vram_used_percentage` near 100% **and** `omnistat_fom` low → `class=gpu_memory_hbm`.
- Power throttling > 0 events → `class=gpu_compute` with remedy "raise power cap if available; otherwise system limit".
- Inter-node `network` rate << intra-node `xgmi` rate AND comm-heavy workload → `class=comm_scaleout` with remedy "verify ANP plugin via `NCCL_DEBUG=INFO`".
- High `network` interface CV at `interface-id` level → `class=comm_scaleout` with remedy "specific interface (e.g. ionic_3) is degraded; investigate cable/firmware".
- Hot GPU (>peak temp threshold from `gpus/mi355x.md` if present, else conservative 90°C) → `class=gpu_compute` with remedy "thermal-bound; check airflow".
- Per-node CPU mem usage CV > 0.3 → `class=host_cpu` with remedy "rebalance num_workers or check NUMA pinning".

### 5. Emit report and claims

`report_summary.md` should mirror Phase 1-3 of the analyze-job SKILL: discovery → data check → health → global stats → drill-down narrative. Cap at ~80 lines.

`claims.json` schema is identical to the tracelens version. Cap at 6 claims.

### 6. Final stdout

```
STATUS=ok; reason=<n_claims> claims; vm_port=<port>; vm_pid=<pid>
```

The orchestrator (or the verifier, when finished) is responsible for stopping VM via `kill $(cat <perf_run>/omnistat/vm.pid)`.

## Failure modes

- DB empty (no metrics): write a single claim `class=host_io` `hypothesis="omnistat-usermode never collected"` and STATUS=partial. Likely cause: collector/server didn't start; check sbatch wrapper logs.
- omnistat-inspect not on PATH: STATUS=fail; the launcher should have installed it.
- VictoriaMetrics startup timeout (15 s): STATUS=fail; print last 20 lines of `vm.log`.

## Notes

- This subagent has read-write access to the perf-run dir.
- It must NOT modify or delete the omnistat DB.
- Never query `atlvrmonad01:9090` — it's IP-blocked and not the right database for this job anyway.
