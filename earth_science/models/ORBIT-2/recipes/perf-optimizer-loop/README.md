# ORBIT-2 — iterative systems optimizer loop (1 node × MI355X, Bayes-CAST EDM)

Agent-driven optimization loop for **ORBIT-2** training with **throughput (`throughput_samples_per_s`)** as the **primary** figure of merit, **`steady_batch_time_s`** + **`loss_sanity_pass`** as controls, and the same **TraceLens + Omnistat** analyst stack as [`../perf-analysis/`](../perf-analysis/).

> **Audience:** AMD performance engineering. Not a scientific downscaling claim — same-dir ERA5 configs are timing-friendly.

## Prerequisites

- **Hardware:** 1 node × 8× MI355X (gfx950) for phase-1; see §Two-node gate below.
- **Container:** `pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` + ORBIT-2 overlay (`build_overlay_amd.sh`).
- **Data:** staged ERA5 NPZ tree; for HBM saturation follow [`STAGING_ERA5_FOR_HBM.md`](STAGING_ERA5_FOR_HBM.md).
- **Code:** `ORBIT2_ROOT` → Bayes-CAST with `launch/train_edm.py` (see [perf-analysis/README.md](../perf-analysis/README.md)).
- **Tools:** `PERF_TOOLS_DIR` (omnistat-usermode + VictoriaMetrics), `AI4S_SHARED_DIR`.

## Environment variables (loop driver)

| Variable | Required | Description |
|----------|----------|-------------|
| `AI4S_SHARED_DIR` | Yes | Shared storage root |
| `PERF_TOOLS_DIR` | Yes | Omnistat + VM binaries |
| `ORBIT2_ROOT` | Recommended | Bayes-CAST checkout |
| `ORBIT2_DATA_ROOT` | Yes | ERA5 staging path |
| `ORBIT2_BATCH_SIZE` | Yes (baseline) | Locked after VRAM sweep |
| `ORBIT2_RANK_PRE_TRAIN_HOOK` | Optional | Absolute path to rank hook **before** `sbatch` (compile / patch) |
| `ORBIT2_TSDB_URL` | Optional | VictoriaMetrics URL for `run_fom_extractor.py` PromQL enrichment |
| `OMNISTAT_ROCPROF_PROFILE` | Optional | `hbm_flops_f64` to force fp64 rocprofiler profile instead of default bf16 |

## Quick start

```bash
export AI4S_SHARED_DIR=...
export PERF_TOOLS_DIR=...
bash earth_science/models/ORBIT-2/examples/validate_orbit2_optimizer_loop_recipe.sh

tmux new -s orbit2-loop
bash earth_science/models/ORBIT-2/examples/run_optimizer_loop.sh "$(uuidgen)" 5
# detach: Ctrl-b d
```

Pre-flight only:

```bash
bash earth_science/models/ORBIT-2/examples/run_optimizer_loop.sh "$(uuidgen)" 5 --preflight-only
```

## HBM saturation before iter-0

1. Stage data per [`STAGING_ERA5_FOR_HBM.md`](STAGING_ERA5_FOR_HBM.md).
2. Batch sweep: [`../examples/sweep_orbit2_batch_bf16_amd.sh`](../examples/sweep_orbit2_batch_bf16_amd.sh).
3. Lock batch in [`../perf-analysis/BASELINE_LOCKIN.md`](../perf-analysis/BASELINE_LOCKIN.md).
4. Fill [`pitfall-diagnosis.md`](pitfall-diagnosis.md) after one perf-analysis pass.

## Orchestration (multi-agent)

Re-use [`../perf-analysis/agents/`](../perf-analysis/agents/) (`launcher`, `tracelens_*`, `omnistat_*`, `synthesizer`) plus this folder:

| Agent | Role |
|-------|------|
| [`agents/orchestrator.md`](agents/orchestrator.md) | Loop driver, sbatch, accept/revert |
| [`agents/lever_picker.md`](agents/lever_picker.md) | Next lever JSON |
| [`agents/fom_extractor.md`](agents/fom_extractor.md) | `foms.json` + optional `kernel_correlation.csv` |
| [`agents/story_writer.md`](agents/story_writer.md) | `story.md` + `foms.png` |

**Launcher:** submit [`../examples/sbatch_train_perf_amd.sh`](../examples/sbatch_train_perf_amd.sh) (same as perf-analysis).

## Artifact layout

Matches HydraGNN layout under `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/`:

- `loop-<uuid>/STATUS.txt`, `foms.csv`, `do_not_retry.json`, `known_bad_nodes.txt`, `iter-*-lever.json`, `iter-*-env.sh`, `story.md`, `foms.png`
- `<jobid>/manifest.json`, `foms.json`, `omnistat-db/`, `tracelens/`, `combined_report.md`, `traces/`

## Loop control

- **Max iters:** user arg (default 5 in driver).
- **One lever per iter**; **one SLURM job at a time**.
- **Accept** when `throughput_samples_per_s` improves vs current best and `loss_sanity_pass` stays true.
- **Reject** on throughput regression or analysis failure (see [`agents/orchestrator.md`](agents/orchestrator.md)).
- **STOP file:** `touch …/loop-<uuid>/STOP`.

## Two-node gate (and beyond)

After one-node loop converges:

1. `sbatch --nodes=2 … earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh` with `.cluster-config.yaml` RCCL/ANP envs (already in script for `NODES>1`).
2. Re-baseline throughput; compare **XGMI** (intra-node) vs **NIC** (`network` category) in omnistat.
3. Only then enable **`nccl_minchannels`** class levers (historically risky at N=2 on HydraGNN).

## Knowledge bases (lever citations)

- [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html) — torch.compile, TunableOp, general ROCm tuning.
- [ROCm multi-node AI setup](https://rocm.docs.amd.com/en/docs-7.0.1/how-to/rocm-for-ai/system-setup/multi-node-setup.html) — NCCL / FSDP context.
- [MI300/MI200 performance counters](https://rocm.docs.amd.com/en/develop/conceptual/gpu-arch/mi300-mi200-performance-counters.html) — `SQ_INSTS_VALU_MFMA_MOPS_BF16` etc.
- [ROCm 7 MI355X training performance blog](https://rocm.blogs.amd.com/artificial-intelligence/ROCm7-MI355X-training-performance/README.html)
- [PyTorch SDPA benchmark (upstream)](https://github.com/pytorch/pytorch/blob/main/benchmarks/transformer/sdpa.py) — studio copy `examples/upstream_pytorch_sdpa_benchmark.py`

## Levers tried and rejected

Recorded so they are not re-attempted. Full evidence in [`gemm-attribution.md`](gemm-attribution.md); catalog entries carry `status: rejected`.

| Lever | Verdict | Why |
|-------|---------|-----|
| `channels_last` (NHWC) | **7–10× slower** | No tuned NHWC implicit-GEMM solver on ROCm 7.2.2 → falls back to brute-force `naive_conv_*_nhwc`. NCHW + im2col→hipBLASLt is the fast path. |
| TunableOp | **No uplift** | Bottleneck is the im2col-lowered low-channel conv GEMM, which TunableOp can't touch; it only tunes the already-optimal `nn.Linear` GEMMs. Also multi-hour to tune the giant patch-token GEMMs. |
| MIOpen ORNL Winograd flags | **Neutral (±1–2%)** | MIOpen never selects Winograd for the 3/4-channel convs, so toggling it changes nothing here. Flags kept as harmless defaults. |
| `torch.compile` (EDM) | **Neutral (−0.3%)** | EDM is already GEMM-heavy; Inductor finds little to fuse. |
| `num_workers` > 1 | **Not a throughput lever** | Workload is compute-bound; >4 oversubscribes CPUs. Keep ≤4 for I/O overlap only. |
| Conv channel-padding | **Tabled** | −25% step time but **changes model capacity** — not adopted without a matched convergence study (`ORBIT2_CONV_PAD` default 0). |
| xFormers CK / bf16 Flash attn | **Crashes** | CK MEA SIGSEGVs; ROCm Flash has a 65535 batch-grid cap that `var_agg` exceeds. Fixed by forcing SDPA EFFICIENT+MATH (now the bf16 default). |

## Machine-readable levers

[`lever_catalog.yaml`](lever_catalog.yaml)

## Reference

HydraGNN recipe (pattern port): [`material_science/models/HydraGNN/recipes/perf-optimizer-loop/`](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/)
