# Dispatch attribution — job 7187 (current best, 75 s/epoch)

**Perf run:** `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/7187/`  
**Lever:** `rccl_high_priority` (`TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`)  
**Nodes:** 2 × MI355X (8 GPUs each)

## Artifacts produced (2026-05-27)

| Artifact | Path |
|---|---|
| Manifest (backfilled) | `perf-runs/7187/manifest.json` |
| TraceLens workbook | `perf-runs/7187/tracelens/report.xlsx` |
| TraceLens digest | `perf-runs/7187/tracelens/report_summary.md` |
| FOMs + correlation summary | `perf-runs/7187/foms.json` |
| 1s window join | `perf-runs/7187/kernel_correlation.csv` |

## `kernel_correlation_summary.attribution_quality`

**Value: `poor`** (automated rule in `run_fom_extractor.py`)

| Metric | Value |
|---|---|
| `windows_examined` | 4 |
| `windows_with_top_kernel_busy_frac_gt_0p5` | 0 |
| `top_kernel_dominants` | `aten::mm` (3 windows), elementwise (1) |

The 1s-window heuristic is **misleading for this run**: rank-0 kineto captures only **~6 s of epoch-2 profiling** (~0.88 s summed kernel time), so `top_kernel_busy_frac` never exceeds ~0.18 in any 1 s wall-clock bucket. That does **not** mean dispatch offenders are unknown.

## TraceLens attribution (authoritative for this run)

From the profile-epoch trace (`tracelens/csvs/ops_summary.csv`):

| Parent op | GPU kernel time | Share |
|---|---:|---:|
| `aten::mm` | 371.7 ms | **47.1%** |
| `aten::bmm` | 259.7 ms | **32.9%** |
| `aten::mul` + elementwise + reduce | remainder | ~20% |

**NCCL** (`record_param_comms` → `ncclDevKernel_Generic_2`) is **~10.6%** of kernel time in the profile slice — non-trivial but secondary to GEMM/BMM launch volume.

Short-kernel study (`short_kernels_summary.csv`): thousands of `aten::fill_`, `aten::copy_`, `aten::mm`/`aten::bmm` dispatches with mean duration **&lt;10 µs** — classic **dispatch-bound** signature, aligned with lesson #13 (fp32≈fp64 epoch time).

## Omnistat join (profile windows)

`kernel_correlation.csv` joins rank-0 trace windows to per-node MFMA/HBM/power. In profile windows MFMA ≈ **0.012–0.014 TFLOP/s** per GPU — near idle vs the job-average **~17 TFLOP/s** in `foms.json` (steady-state epochs 0–1). Low MFMA during the profile slice is expected (profiler overhead + short window).

## Decision (per perf plan)

| Gate | Result |
|---|---|
| TraceLens names top dispatch families? | **Yes** — `aten::mm`, `aten::bmm`, many short kernels |
| `attribution_quality` automated | **poor** (short trace window artifact) |
| Escalate to `OMNISTAT_KERNEL_TRACE=1`? | **No** — TraceLens/kineto already identifies the culprit families; kernel trace would not change the no-upstream playbook |

**Next steps (no upstream HydraGNN):** 8/16-node scaling sweep; do **not** re-run compile/TunableOp/batch levers.

## Commands to reproduce

```bash
export AI4S_SHARED_DIR=/shared/$USER
export REPO_ROOT=$HOME/git/ai4science-studio
source ${OMNIHUB_TOOLS_DIR}/omnihub-inspect/bin/activate

# VictoriaMetrics on the perf-run DB (login node)
${OMNIHUB_TOOLS_DIR}/victoriametrics/victoria-metrics-prod \
  -storageDataPath=$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/7187/omnistat-db \
  -httpListenAddr=127.0.0.1:8432 -retentionPeriod=100y -fs.disableMmap &

cd $AI4S_SHARED_DIR/models/HydraGNN/perf-runs/7187/tracelens
TraceLens_generate_perf_report_pytorch \
  --profile_json_path ../logs/hydragnn-train-7187-N2/<node>_<rank>.pt.trace.json \
  --output_xlsx_path report.xlsx --output_csvs_dir csvs/ \
  --enable_kernel_summary --short_kernel_study

python3 $REPO_ROOT/material_science/models/HydraGNN/examples/run_fom_extractor.py \
  --manifest $AI4S_SHARED_DIR/models/HydraGNN/perf-runs/7187/manifest.json \
  --tsdb-url http://127.0.0.1:8432 --no-start-vm
```