# ORBIT-2 one-node pitfall diagnosis (template)

Fill this file **after** the locked baseline perf job completes and the perf-analysis subagents run (`tracelens_*`, `omnistat_*`, `synthesizer` → `combined_report.md`).

## Baseline job

| Field | Value |
|-------|--------|
| Job ID | `<jobid>` |
| `manifest.json` | `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/manifest.json` |
| `ORBIT2_BATCH_SIZE` / global batch | from manifest |
| dtype | `bfloat16` + `ORBIT2_FUSED_ATTN=DEFAULT` |
| Data | `ORBIT2_DATA_ROOT` + `ORBIT2_ERA5_SPATIAL_RES` |

## Primary metrics (from `run_fom_extractor.py`)

| Metric | Value | Notes |
|--------|-------|--------|
| `throughput_samples_per_s` | | Primary FOM for optimizer loop |
| `steady_batch_time_s` | | Latency control |
| `loss_sanity_pass` | | Crash detector (same-dir ERA5) |

## Omnistat (HBM / BF16 MFMA / XGMI)

- **Profile:** default `hbm_flops_bf16` in [`../perf-analysis/omnistat.config.template`](../perf-analysis/omnistat.config.template) — `FETCH_SIZE` + `SQ_INSTS_VALU_MFMA_MOPS_BF16` ([AMD counter tables](https://rocm.docs.amd.com/en/develop/conceptual/gpu-arch/mi300-mi200-performance-counters.html)).
- **XGMI:** category `xgmi` in `omnistat-inspect` / `stats_global.json` — intra-node FSDP traffic.
- **Network:** inter-node only relevant at N>1; see [README.md](README.md) §Two-node gate.

Paste 2–3 bullets from `omnistat/report_summary.md`.

## TraceLens (overlap / kernels)

Paste top kernels / NCCL overlap narrative from `tracelens/report_summary.md`.

## Bottleneck class (pick one dominant)

- [ ] `gpu_compute` (MFMA / HBM read high vs peak)
- [ ] `gpu_memory_hbm` (VRAM saturated, step time grows with batch)
- [ ] `cpu_dispatch` / `dataloader` (low MFMA, host I/O or DataLoader gaps in trace)
- [ ] `comm_xgmi` (FSDP all-gather / reduce-scatter dominates on 1×8)
- [ ] `comm_scaleout` (only N>1)

## bf16 vs fp32 discriminator (optional one-shot)

Run one short job at **`ORBIT2_DATA_TYPE=float32`** with the **same** `ORBIT2_BATCH_SIZE`. If throughput barely changes → dispatch- or attention-path limited; if fp32 ≫ bf16 → memory-bandwidth sensitive.

| dtype | throughput_samples_per_s |
|-------|--------------------------|
| bf16 | |
| fp32 | |

## Levers justified (maps to `lever_catalog.yaml`)

List 3–5 catalog `id`s with one-line rationale each, grounded in sections above + citations in the catalog.

---

_Replace `<jobid>` and tables after the first diagnosis run; keep this file under git for loop traceability._
