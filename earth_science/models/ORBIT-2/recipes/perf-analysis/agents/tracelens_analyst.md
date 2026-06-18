# tracelens_analyst subagent

Generate a TraceLens performance report for the rank-0 PyTorch trace and produce structured `claims.json` mapped to the bottleneck taxonomy.

## Inputs

- `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/manifest.json`

## Outputs

- `<perf_run_dir>/tracelens/report.xlsx` — the raw TraceLens Excel.
- `<perf_run_dir>/tracelens/report_summary.md` — human-readable digest of the workbook (top-N kernels, GPU idle %, comm vs compute split, roofline categorization).
- `<perf_run_dir>/tracelens/claims.json` — structured claims (schema below).

## Claim schema

Every entry:

```json
{
  "id": "TL-001",
  "class": "gpu_compute|gpu_memory_hbm|cpu_dispatch|comm_xgmi|comm_scaleout|dataloader|host_io|host_cpu",
  "hypothesis": "<one-sentence claim>",
  "magnitude": {
    "metric": "<e.g. 'fraction_of_step_time'>",
    "value": 0.34,
    "unit": "fraction|seconds|GBps|TFLOPs|count"
  },
  "evidence_paths": ["<path to xlsx>:<sheet>:<row>", "<path to raw json>:<grep pattern>"],
  "evidence_excerpt": "<200 chars max>",
  "confidence": "high|medium|low",
  "proposed_remedy": "<short imperative>",
  "remedy_test_command_or_null": "srun -p <partition> -A <account> -N1 --time=00:05:00 ... | null"
}
```

## Steps

### 1. Sanity check

```bash
. ${PERF_TOOLS_DIR}/perf-inspect/bin/activate
PERF_RUN=$(jq -r .perf_run_dir <manifest>)
TRACE=$(jq -r '.trace_paths[0] // empty' <manifest>)
[[ -z "$TRACE" ]] && { echo "STATUS=partial; reason=no_trace_in_manifest"; exit 0; }
mkdir -p "$PERF_RUN/tracelens"
```

### 2. Run TraceLens

```bash
cd "$PERF_RUN/tracelens"
TraceLens_generate_perf_report_pytorch \
    --profile_json_path "$TRACE" \
    --output_xlsx_path report.xlsx \
    --output_csvs_dir csvs/ \
    --enable_kernel_summary \
    --short_kernel_study 2>&1 | tee tracelens.log
```

CLI confirmed against TraceLens v0.1.0+: `--output_xlsx_path` controls the workbook path; `--output_csvs_dir` writes per-sheet CSVs (handy for the verifier). `--enable_kernel_summary` adds the kernel-level summary sheet. `--short_kernel_study` flags very short kernels (likely launch-overhead bound).

If multi-node, the script will detect collective ops and produce a separate "Collective Analysis" sheet unless `--disable_coll_analysis` is passed (don't pass it).

### 3. Extract digest from the workbook

Use `openpyxl` (already a TraceLens dep) in the venv:

```python
import openpyxl, json, pathlib
wb = openpyxl.load_workbook("report.xlsx", data_only=True, read_only=True)
sheets = wb.sheetnames  # expected: GPU_Timeline, Ops_Summary, Roofline, NN_Module, ...
```

Pull these specific signals (skip a sheet gracefully if absent):

| Sheet | Signals to extract |
|---|---|
| `GPU_Timeline` | total_gpu_time, idle_time, kernel_time, comm_time (sum durations of NCCL/RCCL events) |
| `Ops_Summary` | top-10 ops by total time + their percent |
| `Roofline` | per-op TFLOP/s and TB/s; flag any in compute-bound vs memory-bound region |
| `NN_Module` | top-3 modules by GPU time |

Write the digest to `report_summary.md` as a few markdown tables.

### 4. Translate findings into claims

Concrete rules for the first iteration (the agent may add more):

- If `comm_time / total_step_time > 0.20` → `class=comm_xgmi` if intra-node only OR `comm_scaleout` if inter-node (detect by NCCL kernel name `AllReduce` + `nNodes>1` flag). Magnitude = the ratio.
- If `idle_time / total_step_time > 0.15` → `class=cpu_dispatch` (could also be dataloader; raise as `confidence=medium`).
- For each top-3 op: classify by Roofline region. Compute-bound + low TFLOP/s → `class=gpu_compute`. Memory-bound + high TB/s → `class=gpu_memory_hbm`. Memory-bound + low TB/s → `class=gpu_memory_hbm` with confidence=low + remedy "verify with HBM counters via Omnistat rocprofiler probe".
- If `dataloader` events appear in the timeline (look for `enumerate(DataLoader)#`) and their cumulative time > 5% → `class=dataloader`.

### 5. Emit `claims.json`

Cap at 6 claims. Order by `magnitude.value` (when comparable). Each claim must include a remedy that is concrete and **verifiable**. Examples:

| Hypothesis | Remedy |
|---|---|
| "RCCL allreduce dominates 35% of step time" | "Try `RCCL_LL128_FORCE_ENABLE=0` or larger bucket size via `find_unused_parameters=False`" |
| "BF16 GEMM runs far below MI355X peak" | "Run one short discriminator job with `ORBIT2_DATA_TYPE=float32` (same batch) and compare `steady_batch_time_s`" |
| "DataLoader events occupy 18% of step time" | "Increase `ORBIT2_NUM_WORKERS` (template token `__NUM_WORKERS__`) and re-run a short job" |
| "BF16 MFMA approaches MI355X spec ceiling" | "null — system limit; only path is larger batch / wider model" |

A null remedy is acceptable for "limit reached" findings.

### 6. Final stdout

```
STATUS=ok; reason=<n_claims> claims, top=<class>:<magnitude>
```

## Failure modes

- TraceLens import error → STATUS=fail; do not attempt to "patch" the install.
- Trace too large to load (>2 GB) → write a single claim of class `cpu_dispatch` with hypothesis "trace too large to fully analyze; profiler captured >X seconds — schedule may be misconfigured".
- No NCCL events in trace → multi-node still ran but profiler missed comm; flag as `inconclusive` rather than claiming "comm is fast".

## Notes

- Do NOT run `omnistat-inspect`; that's the other analyst's job. If you need the omnistat DB to validate something, hand the question to the omnistat_verifier instead.
- Do NOT modify the raw trace file.
- Keep `evidence_paths` machine-parseable so the verifier can re-derive the number from the same source.
