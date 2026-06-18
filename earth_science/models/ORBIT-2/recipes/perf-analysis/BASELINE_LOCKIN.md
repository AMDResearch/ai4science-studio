# ORBIT-2 — baseline lock-in for GPU saturation (MI355-class, 1×8)

Use this file to **freeze** what “baseline” means while you sweep knobs. Update the **Reference job** row when you promote a new winner.

## Reference job (current)

| Field | Value (example: job **9145**) |
|--------|--------------------------------|
| **Job dir** | `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/` |
| **Template** | `interm_8m_era5.yaml` (ERA5 1.0°, same-dir, `spatial_resolution` 111×111) |
| **Parallelism** | `fsdp=8`, `simple_ddp=1` (1 node × 8 ranks) |
| **dtype** | `bfloat16` (Studio **`sbatch_train_*` / `run_scaling_study.sh`** default). Set **`ORBIT2_DATA_TYPE=float32`** if `xformers.ops` / CK attention misbehaves; **`ORBIT2_FUSED_ATTN=DEFAULT`** (already in sbatch) prefers PyTorch SDPA. |
| **Epochs / cap** | `ORBIT2_MAX_EPOCH=6`, `ORBIT2_MAX_BATCHES=20` |
| **Model block** | `preset: res_slimvit`, `embed_dim: 256`, `depth: 6`, `decoder_depth: 4`, `num_heads: 4`, `mlp_ratio: 4` |

## Bayes-CAST EDM — bf16 + SDPA (sysopt default, 1×8)

**Perf-optimizer-loop** uses **`ORBIT2_DATA_TYPE=bfloat16`**, **`ORBIT2_FUSED_ATTN=DEFAULT`** (PyTorch SDPA), template **`edm_8m_era5_1x8.yaml`**, and **`ORBIT2_ROOT`** pointing at **Bayes-CAST** (`…/code/bayes-cast`).

| Step | Action |
|------|--------|
| 1 | If VRAM plateaus **below** target HBM fraction, **stage more / higher-res ERA5** — see [`../perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md`](../perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md) (breaks the ~1704 effective-batch cap on tiny staging trees). |
| 2 | Run a **bf16 batch sweep** (binary search): [`examples/sweep_orbit2_batch_bf16_amd.sh`](../../examples/sweep_orbit2_batch_bf16_amd.sh) |
| 3 | Per job: `python3 …/run_fom_extractor.py --job-dir …/perf-runs/<jobid>` then `report_orbit2_gpu_baseline.py` → **`baseline_report.md`** beside `manifest.json`. |
| 4 | Lock **~85–90%** `memory_reserved` (leave fragmentation headroom) and best **throughput** (`run_fom_extractor` → `throughput_samples_per_s`) vs **`steady_batch_time_s`**. |

**Reference job (bf16):** ✅ **bf16 works** after a compute-side attention fix (2026-06-11). The earlier crash (`var_agg` SDPA, `HIP error: invalid argument`) was **ROCm Flash attention's 65535 batch-grid cap**: `var_agg` flattens `(B,History,L)` into the SDPA batch dim `B = batch·648`, which exceeds 65535 at **batch ≥ 128**. Fix: force `CrossAttention` SDPA to **EFFICIENT_ATTENTION+MATH** (not Flash) in `src/climate_learn/models/hub/components/attention.py`. Validated job **9238** (bf16 b256 COMPLETES, decreasing loss). Root cause + fix: see the earth-science SKILL (bf16 Flash 65535 cap). Note the **~1704-sample data cap**: bf16 won't reach 85–90% HBM at 1704 samples (fp32@1704 ≈ 183 GiB ≈ 64%; bf16 ≈ half) — stage more ERA5 to saturate.

**Reference job (bf16, working) — LOCKED 2026-06-11:** `jobid=9240`, `ORBIT2_BATCH_SIZE=1024`, `ORBIT2_DATA_ROOT=${AI4S_SHARED_DIR}/models/ORBIT-2/data/superres/era5/1.0_deg`, `ORBIT2_ERA5_SPATIAL_RES=111`, `ORBIT2_DATA_TYPE=bfloat16`, `ORBIT2_FUSED_ATTN=DEFAULT`, `fsdp=8`/`simple_ddp=1`. **Throughput = 4098 samples/s** (`steady_batch_time_s=2.00`, gb=8192, loss sanity ✅). Reproduced by job 9257 (4097 s/s).

**Batch-scaling story (bf16, num_workers=1, 1704-sample data, sweep-batch-nw1 2026-06-12):** throughput rises monotonically with batch size — the textbook compute-bound GEMM-efficiency curve (tiny batches starve the MFMA units; larger batches amortize launch/overhead):

| batch | global batch | throughput | steady step | job |
|------:|------:|-----------:|------------:|-----|
| 32   | 256  | 1869 s/s | 0.137s | 9265 |
| 64   | 512  | 2530 s/s | 0.202s | 9266 |
| 128  | 1024 | 2529 s/s | 0.405s | 9267 |
| 256  | 2048 | 2819 s/s | 0.727s | 9268 |
| 512  | 4096 | 3117 s/s | 1.314s | 9269 |
| 1024 | 8192 | **3987 s/s** | 2.055s | 9270 |

**⚠️ Throughput ≠ VRAM at the current 1704-sample data cap** — bigger batch saturates more HBM but is *slower* here, because batch > sample-count just repeats/partial-fills the same step:

| batch | gb | HBM | throughput | steady step | job |
|------:|----:|----:|-----------:|------------:|-----|
| 512  | 4096  | ~11% | 3930 s/s | 1.04s | 9239 |
| **1024** | **8192** | **~21%** | **4098 s/s (best)** | **2.00s** | **9240** |
| 1704 | 13632 | ~34% | 2975 s/s | 4.58s | 9241 |

So the **throughput-optimal baseline is batch 1024 (~21% HBM)**, not the VRAM-max batch 1704 (-27% throughput). To make HBM saturation *and* throughput agree, you must **stage more ERA5** (raise the 1704-sample cap) and re-sweep — until then the optimizer loop ranks levers against the batch-1024 baseline.

**bf16 VRAM calibration (post-fix, MI355X 1×8, ERA5 1.0° `edm_8m_era5_1x8.yaml`, FSDP 8 / DDP 1, rank-0 end-of-run):**

| Per-rank `batch_size` | `max_memory_allocated` | `memory_reserved` | ~% of 288 GB | Job |
|----------------------:|-----------------------:|------------------:|-------------:|-----|
| 512  | 18.93 GiB | 31.87 GiB | ~11% | 9239 |
| 1024 | 37.69 GiB | 61.62 GiB | ~21% | 9240 |
| 1704 (sample cap) | 62.61 GiB | 97.28 GiB | ~34% | 9241 |

**Takeaway:** with only **~1704 train samples**, bf16 caps at **~97 GiB reserved (~34% HBM)** — batch ≥1704 just repeats the same 1704-sample step. To reach **85–90%** you must **stage more ERA5** (more years and/or higher resolution; see [`../perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md`](../perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md)) then re-sweep. Growth is ~linear here (512→1704), so ~3× more samples at batch≈5000 would approach ~85%.

## Bayes-CAST EDM reference (`edm_8m_era5_1x8`, float32, 1×8, ERA5 1.0°)

Smoke **9158** used **`ORBIT2_BATCH_SIZE=2`** (~0.35 GiB **`max_memory_allocated`**, ~0.6 GiB reserved). Follow-up calibration jobs (same template, **`ORBIT2_DATA_TYPE=float32`**, FSDP 8 / DDP 1) on a single MI355X node — rank-0 **`torch.cuda.max_memory_allocated`** / **`memory_reserved`** at **end of epoch 0** (upstream logs once per epoch):

| Per-rank `batch_size` | ~`max_memory_allocated` | ~`memory_reserved` | Job |
|----------------------:|------------------------:|-------------------:|-----|
| 32 | 2.31 GiB | 1.10 GiB | 9159 |
| 64 | 4.81 GiB | 4.23 GiB | 9160 |
| 96 | 6.72 GiB | 7.49 GiB | 9161 |
| 128 | 8.66 GiB | 4.29 GiB | 9162 |
| 160 | 10.78 GiB | 11.58 GiB | 9163 |
| 192 | 12.90 GiB | 18.39 GiB | 9164 |
| 224 | 15.02 GiB | 15.04 GiB | 9165 |
| 256 | 17.12 GiB | 18.39 GiB | 9166 |
| 1024 | 67.94 GiB | ~73 GiB | 9167 |
| 2048 | 112.93 GiB | ~183 GiB | 9168 |
| 3072 | 112.93 GiB | ~183 GiB | 9169 |

**Takeaway:** growth is **not linear** in batch (see jump **256 → 1024**). Naive linear fit from small batches **over-predicts** headroom. For **~288 GiB** HBM, **`ORBIT2_BATCH_SIZE=2048`** already uses **~183 GiB reserved** (~**64 %**). Jobs **9168** and **9169** logged the **same** peak memory: rank-0 **`y.shape`** was **`[1704, 3, 108, 216]`** in both (not 2048/3072), so for this ERA5 staging + 8-way layout the **effective per-step batch dim is capping near 1704** — raising **`trainer.batch_size` above ~2048** did not increase the tensor batch for those epochs; re-measure if you add data or change sampler/shard rules. Re-measure after switching to **`bfloat16`**.

## What “maximal utilization” means here

1. **HBM capacity** — Rank-0 log `torch.cuda.memory_reserved` near a **practical** fraction of available per-GPU HBM (leave headroom for fragmentation and checkpoint spikes). Example **float32** ERA5 same-dir calibration on this stack: job **9145** @ per-rank batch **4** → ~**12.1 GB**; job **9146** @ batch **8** → ~**28.1 GB** (late-epoch `batch_idx 19` lines) — still far below **~288 GB** HBM-class MI355 OAM → **raise batch** (and prefer **`bfloat16`** to stretch further) before widening the ViT.
2. **Compute / bandwidth** — Omnistat rocprofiler-derived **MFMA / FETCH** activity during **epoch ≥ 2** (after `run_fom_extractor.py` steady window). Use perf-analysis agents on `omnistat-db/` + `traces/` once batch is pinned.
3. **Step time** — `steady_batch_time_s` from logs (sparse: upstream prints `Batch N: … seconds` mainly for **batch 0 and 10** per epoch). Compare across batch sweep, not only VRAM.

## Ordered next steps (do in sequence)

1. **Lock topology + data + template** — Keep **ERA5** + **`interm_8m_era5.yaml`** + **1×8** + **`fsdp=8`/`simple_ddp=1`** fixed until VRAM plateaus.
2. **Sweep `ORBIT2_BATCH_SIZE`** — Prefer **binary search** between a known-good batch and a guessed upper bound (fewer SLURM jobs than stepping by +1). Re-run `run_fom_extractor.py` + `report_orbit2_gpu_baseline.py` per candidate; pick the best **time vs throughput** under high VRAM.
3. **Optional: denser batch timing in logs** — If you need more than two `Batch … seconds` lines per epoch, raise upstream logging frequency or increase `ORBIT2_MAX_BATCHES` (still capped per epoch in trainer) so steady stats are statistically tighter.
4. **Omnistat + TraceLens** — On the **winning** job id only: confirm MFMA/HBM story matches “saturated” intent ([`one-node-gpu-baseline.md`](one-node-gpu-baseline.md) §6–7, [`ai4science-perf-analysis`](../../../../.cursor/skills/ai4science-perf-analysis/SKILL.md)).
5. **Only if VRAM is high and MFMA still low** — Then widen **`embed_dim`/`depth`** (with `embed_dim % num_heads == 0`) or revisit `num_workers` / I/O.

## Max batch: estimate vs measure

**There is no substitute for a real forward+backward** on your exact **dtype, FSDP layout, and config** — analytical “FLOPs-only” caps ignore fragmentation, activation checkpoints, Lightning overhead, and checkpoint spikes.

1. **Two-point linear fit (starting guess only)** — From two short jobs at batches \(B_1, B_2\) with similar steady **`memory_reserved`** (e.g. last batch in epoch), fit a line and extrapolate to a **target GiB** (e.g. 220–260 GiB under 288 GiB HBM). Script: [`examples/orbit2_estimate_batch_from_memory.py`](../../examples/orbit2_estimate_batch_from_memory.py).  
   *Example (float32, same logs as above, target 240 GiB):* slope ≈ **4.0 GiB per batch-step** between \(B=4\) and \(B=8\) → naive linear intercept gives **\(B \approx 61\)** — **often over-optimistic** once attention and allocator effects bend the curve; treat as an **upper exploratory bound**, not the first batch to run overnight.
2. **After switching to `bfloat16`** — Re-measure two batches (e.g. 8 and 16): BF16 rarely halves *all* tensors (master weights, NCCL buffers, I/O), so **do not** assume “2× the float32 batch” without logs.
3. **Better than linear stepping:** **binary search** on `ORBIT2_BATCH_SIZE` with **1–2 epoch** smoke jobs (`ORBIT2_MAX_EPOCH=3` is enough to see reserved memory plateau mid-epoch if you watch `batch_idx 19` lines). Optionally add a **single-node one-rank** micro-probe (not shipped here) if you only need allocator OOM without full 8-way FSDP — less representative but fast.

## Promotion rule

When a new job beats the old one on **(steady VRAM fraction, MFMA evidence, acceptable `steady_batch_time_s`)**, replace the **Reference job** section with that job id and the chosen `ORBIT2_BATCH_SIZE`.
