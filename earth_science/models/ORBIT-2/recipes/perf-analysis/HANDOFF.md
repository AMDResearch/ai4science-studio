# ORBIT-2 perf-analysis HANDOFF

_Last updated: 2026-06-11 (perf-optimizer-loop + bf16 omnistat profile + manifest fields)._

## What works

| Item | Status | Notes |
|------|--------|-------|
| 1-node × 8-GPU training on **ERA5 1.0°** same-dir (default perf template) | **Green** | Heavier **111×111** latent vs PRISM **18×18**; better default for GPU saturation baselines. Representative steady batch time workload-dependent. |
| 1-node × 8-GPU on **PRISM** 10.0_arcmin same-dir | **Green** | Lighter forwards; override `ORBIT2_DATA_ROOT` + `interm_8m_lux.yaml` if ERA5 not staged. |
| `sbatch_train_amd.sh` / `sbatch_train_perf_amd.sh` | **Green** | Prefer `launch_diffusion.sh` under `ORBIT2_ROOT` when present; else `run_orbit2_train.py`. `orbit2_rank_hook_runner.py` runs profiler hook before shell launcher. |
| **1-node default parallelism** | **fsdp=8, simple_ddp=1** | Override with `ORBIT2_FSDP` / `ORBIT2_SIMPLE_DDP`. Multi-node: omit for `fsdp=nodes`, `simple_ddp=8`. |
| `ORBIT2_CONFIG_TEMPLATE` + `render_orbit2_config.py` | **Green** | Default perf: **`edm_8m_era5_1x8.yaml`** (Bayes-CAST `edm_8m_era5` + **`fsdp=8`/`simple_ddp=1`/`seq_par:1`**, templated `data_dir` + **`ORBIT2_ERA5_SPATIAL_RES`** default 111). Lux **`interm_8m_lux_era5.yaml`** for public ORBIT-2 `res_slimvit`. `__DATA_TYPE__` defaults **bfloat16**. |
| Node health probe (exit 42) | **Green** | Catches broken home/shared/SIF/data mounts before training |
| `superres_mag: 1` same-dir config | **Required** | Without it, 4× superres vs same-resolution targets causes tensor shape mismatch |
| Baseline lock-in + VRAM-first sweep | **Doc** | [BASELINE_LOCKIN.md](BASELINE_LOCKIN.md) — freeze topology/template; sweep `ORBIT2_BATCH_SIZE` before widening ViT |

## Landmines

1. **`max_epochs` must be ≥ 2** — upstream loop is `while (epoch_start + 1) < max_epochs`; `max_epochs=1` runs zero epochs.
2. **`xformers.ops` missing in overlay** — `run_orbit2_train.py` can fall back to `FusedAttn.DEFAULT` (PyTorch SDPA). Rebuild overlay per `build_overlay_amd.sh` if you need the CK attention path.
3. **`bfloat16` + xformers CK path** — needs working `xformers.ops`. Studio sbatch sets **`ORBIT2_FUSED_ATTN=DEFAULT`** so PyTorch SDPA is used when CK is unavailable. If you still see dtype-related crashes, fall back to **`ORBIT2_DATA_TYPE=float32`** for isolation.
4. **Same-dir data** — both `low_res_dir` and `high_res_dir` point at the same grid; set `superres_mag: 1` in `interm_8m_lux.yaml` / `interm_8m_lux_era5.yaml`.
5. **Batch cap** — must patch `intermediate_downscaling` before `main()` (not via `runpy.run_path` reload).
6. **Bayes-CAST `launch/launch_diffusion.sh`** — upstream file is often an OLCF/Crusher **job wrapper** (nested `srun`, ignores argv). Studio **`sbatch_train_*`** does **not** exec it when **`$ORBIT2_ROOT/launch/train_edm.py`** exists; ranks run **`python3 train_edm.py /config/config.yaml`** from **`/orbit2/launch`** (no `examples/` required). For Lux/public ORBIT-2, **`launch_diffusion.sh`** + argv still applies when **`train_edm.py`** is absent.
7. **`train_edm.py` + `ERA5_1`** — some Bayes-CAST snapshots only define **`std_delta`** for IMERG/HRRR; **`data_key=="ERA5_1"`** then hits **`UnboundLocalError`** at the first step. Fix: add an **`elif data_key == "ERA5_1"`** branch (per-channel std aligned with your staged **`normalize_std.npz`** and **`dict_out_variables` order**), or use the small patch applied on your shared clone (example values `[20.601086, 5.542934, 4.7573]` for T2m / u10 / v10 @ 1.0°).

## FOM contract (steady-state + loss sanity)

**Primary FOM:** `steady_batch_time_s` — mean per-batch wall time from **epoch ≥ 2**, excluding **batch index &lt; 1** within each epoch (skips compile/warmup batch 0). Epochs 0–1 are warmup.

**Loss sanity:** epoch-completed losses must **strictly decrease** epoch-over-epoch (`loss_sanity_pass`). Same-dir data makes absolute loss non-physical; use as crash/hang detector only.

**Scaling runs:** use `ORBIT2_MAX_EPOCH=6` (trains epochs 0–4) so the steady window spans epochs 2–4. Parse with:

```bash
python3 parse_training_log.py --log ... --steady-epoch-start 2 --warmup-batches-per-epoch 1
python3 collate_scaling_study.py --log-dir .../logs --jobs ... --steady-epoch-start 2 --require-loss-sanity
```

## GEMM-time bottleneck + ORNL MIOpen flags (2026-06-15, jobs 12562/12648)

Full TraceLens+Omnistat analyst/verifier run (`agents/orchestrator_gemm_analysis.md`, driver
`examples/run_gemm_analysis.sh`) on 1- and 2-node profiled EDM (bf16, batch 4096, ERA5 1.0°). Report:
`perf-runs/gemm-analysis-<uuid>/GEMM_TIME_REPORT.md`.

- **The hot kernel is NOT attention/MLP.** A single hipBLASLt GEMM `Cijk_…MT16x16x32_MI16` is
  **~43 %** of the step at *both* scales (scale-invariant). It is the **im2col-lowered low-channel
  conv backward** for the **3- and 4-channel stem/head projection convs**, stuck on a 16×16 macro-tile
  at ~0.03 TFLOP/s. The tunable `nn.Linear` bf16 GEMMs (256×256 tiles, 600–1074 TFLOP/s) are only
  ~2 %. **This explains why TunableOp gave no uplift** — it tuned already-optimal GEMMs and can't
  touch the conv-lowered one. `im2col` (`Im2d2Col_v2`) adds ~14 %; conv **forward** is a `naive_conv`
  with **fp64 accumulate** (`..._ushort_double_ushort`) at ~3 %.
- **TraceLens `report.xlsx:ops_summary_by_category` MISLABELS the conv-GEMM as `CONV_bwd`.** The
  2-node analyst trusted it and claimed "naive conv 65.8 %"; the verifier refuted it from raw
  `traceEvents` (naive_conv_bwd = 0.46 %, GEMM = 50.7 %). **Always rank by raw kernel-name device
  time, not the category rollup,** for conv-heavy models. (`examples/compare_trace_kernels.py` does
  this tool-independently.)
- **~43–44 % whole-job idle at both scales** (Omnistat util mean ~56 %) is the second lever — it is
  structural inter-step overhead, **not** comm-bound (exposed comm 0.13 %→0.47 %; the Omnistat
  "comm-bound" claim was refuted by the trace).
- **EDM trainer now has an env-gated profiler.** `train_edm.py:_orbit2_make_profiler` writes a rank-0
  `*.pt.trace.json` for `PROFILE_TARGET_EPOCH` into `ORBIT2_PROFILE_DIR` (no-op unless set). This is
  what makes "where does GEMM time go" answerable for the EDM path (the old Lux `res_slimvit` hook
  did not apply).

### ORNL Frontier MIOpen flags ported to the sbatch (validate on Lux)

From bayes-cast `launch/launch_diffusion.sh` (ORNL, gfx90a/ROCm7.1.1, "tested many times"): ORNL
**disables Winograd** and unbounds the multi-pass Winograd workspace. Now defaults in
`sbatch_train_perf_amd.sh` / `sbatch_train_amd.sh` / `sbatch_infer_*.sh` (override to A/B):

```
MIOPEN_DEBUG_CONV_WINOGRAD=0            # ORBIT2_MIOPEN_CONV_WINOGRAD (0=off ORNL default, 1=on)
MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1   # ORBIT2_MIOPEN_WINOGRAD_MPASS_WS_MAX
MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1   # ORBIT2_MIOPEN_MP_BD_WINOGRAD_WS_MAX
HSA_FORCE_FINE_GRAIN_PCIE=1
```

`MIOPEN_DISABLE_CACHE=1` + `MIOPEN_USER_DB_PATH=/tmp/<job>` already matched ORNL — keep them (do NOT
"persist the find-db"; that deviates from the validated setup). **Deliberately NOT ported** (Frontier
Slingshot/Cray-specific, would break Lux IB/ionic): `FI_CXI_*`, `NCCL_NET="AWS Libfabric"`,
`NCCL_NET_PLUGIN=librccl-net.so`, `NCCL_SOCKET_IFNAME=hsn0`, Frontier `LD_PRELOAD` host paths.
A/B (ORNL Winograd-off vs Winograd-on vs pre-flags
baseline) → `perf-runs/miopen-ornl-validation-<uuid>/MIOPEN_ORNL_VALIDATION.md`.

**MIOpen ORNL-flags A/B result (2026-06-16, jobs 12997/12998 vs 12562): performance-NEUTRAL on Lux**
(steady 8.62→8.66→8.74 s, within ~1-2 % noise; loss sanity OK). MIOpen never selected Winograd for
the 3/4-channel convs (0 % Winograd kernels in all arms), so toggling it changes nothing — our
bottleneck is shape-driven, not algorithm-selection-driven. Keeping the flags as defaults (faithful,
harmless, may matter elsewhere). They are **not** the lever here.

### Conv channel-padding — TABLED candidate, timing-only (NOT ADOPTED, 2026-06-16, jobs 14633/14634)

> **Status: tabled.** The −25 % is a *timing-only* result. Channel-padding **changes model capacity**
> (wider internal convs ⇒ different params), and **we have not run a convergence/quality study**, so it
> is **not adopted** and `ORBIT2_CONV_PAD` stays **default 0 (off)**. Treat the numbers below as a
> *potential* speedup pending a same-quality convergence check (matched final loss/skill at equal or
> fewer steps). Do not enable for real training runs until that check passes.

`edm.py` `path2`/`refine` convs run at out_channels(=3)/cnn_ratio(=4) at full resolution → the
starved `Cijk_…MT16x16x32` GEMM @ ~0.03 TFLOP/s. New env knob **`ORBIT2_CONV_PAD=N`** (in `edm.py`,
default 0 = original) widens those convs' INTERNAL width to N (tile-friendly) keeping the 3-channel
I/O. Single-node A/B (identical bf16/4096/ERA5 config), N=16:

| Arm | steady_batch_time_s | total GPU-kernel ms | dominant `MT16x16x32` GEMM |
|---|---|---|---|
| control (pad=0) | 8.646 | 21082 | 9245 ms (43.9 %) |
| **widened (pad=16)** | **6.498 (−24.8 %)** | 15150 (−28 %) | **gone** (top GEMM now `MT16x16x128`, 4123 ms) |

**~25 % faster step, loss *sanity* holds (not convergence).** Confirms the GEMM report: low channel
count forced the bad tile; widening makes it tile-friendly and the efficiency gain beats the extra
FLOPs. **Why tabled:** "loss sanity" only checks the loss is finite/decreasing over a few epochs — it
is **not** evidence the wider model reaches the same quality. Before adopting, run a matched
convergence study (same data/schedule, compare final validation loss/skill and steps-to-target for
pad=0 vs pad=16); only adopt if quality is equal-or-better. `channels_last`/implicit-GEMM as a way to
kill the residual `im2col` (~21 %) is **already disproven** (see next section — naive_conv fallback,
7-10× slower). If/when revisited, the open timing question is **N=32** (top GEMM `MT16x16x128` at 27 %
still has headroom) — but only worth it after convergence is settled.
Report: `perf-runs/conv-pad-validation-<uuid>/CONV_PAD_VALIDATION.md`.

### `channels_last` + implicit-GEMM — DEAD END on ROCm 7.2.2 (DISPROVEN, 2026-06-16, jobs 14645-14648)

Tested the "stack `channels_last` on top" idea above as a clean 2×2 ({NCHW, channels_last} × {pad=0,
pad=16}) via the env knob **`ORBIT2_CHANNELS_LAST`** in `edm.py`,
default 0; sets `MIOPEN_FIND_MODE=1` on the NHWC arms). **It is 7-10× SLOWER, not faster:**

| Arm | CONV_PAD | channels_last | steady_batch_time_s | vs control |
|---|---|---|---|---|
| nchw_pad0 (control) | 0 | off | 8.62 | — |
| **nchw_pad16** | 16 | off | **6.38** | **−26 %** ✅ |
| cl_pad0 | 0 | on | 60.8 | +606 % 💥 |
| cl_pad16 | 16 | on | 80.1 | +829 % 💥 |

**Root cause (kernel trace):** MIOpen on this build has **no tuned NHWC implicit-GEMM solver** for
these conv shapes. Forcing channels_last makes it fall back to the brute-force **`naive_conv_…wrw_nhwc`**
direct-convolution kernel, which dominates **~88-91 %** of GPU-kernel time (cl_pad0: 139,522 ms;
cl_pad16: 185,198 ms). The `im2col` (`Im2d2Col_v2`, 14-21 % in NCHW) is **not** wasted overhead — it is
the price of reaching the tuned **hipBLASLt GEMM** kernels (`Cijk_…`, 31-54 % in NCHW). NHWC discards
that path. **Lesson: do NOT use `channels_last` for these low-channel convs on MI355X/ROCm 7.2.2.**
**The lever is NCHW + `ORBIT2_CONV_PAD` (−26 %); channels_last is abandoned.** Knob kept (default off)
only so the dead end is reproducible. Report: `perf-runs/conv-layout-validation-<uuid>/CONV_LAYOUT_VALIDATION.md`.

## Scaling / wall-clock lessons

- Long runs on full 180×360 grids need **enough wall time** (multi-epoch × many batches can exceed 1 h on some sites).
- Point SLURM `--output` / `--error` at a **shared project filesystem** your compute nodes can always read during outages (not only the submission directory on a login node).
- **Multi-node hangs** are often **infrastructure** (drained nodes, stuck scheduler states, shared filesystem hiccups) — confirm node state with `sinfo` / site ops before assuming a code bug.
- For exclusive GPU jobs, **avoid stacking two jobs on the same node**. Pin nodes with `--nodelist=...` only when your site policy allows it and you have confirmed the nodes are healthy.

## Debug session handoff — Bayes-CAST EDM, `bfloat16`, xFormers CK (SIGSEGV)

**Why this session felt confusing**

- **Two trees:** Instrumentation and hotfixes for Bayes-CAST live on the **compute-side clone** (e.g. **`$AI4S_SHARED_DIR/models/ORBIT-2/code/bayes-cast/`**), not inside this git repo. AI4Science Studio only documents launchers and perf recipes; agents may edit `/shared/...` while you expect everything under `ai4science-studio/`.
- **Debug log path vs cluster:** Agent NDJSON defaults to **`<repo-root>/.cursor/debug-<session-id>.log`**. Compute nodes inside Apptainer often **cannot** write that path. Unless you set **`DEBUG_AGENT_LOG`** to something bind-mounted and writable, then **copy** the file back to the workspace path, the next chat turn will show **“log not found”** even after a good repro — that is an artifact workflow gap, not necessarily “no crash.”
- **Session id:** the agent debug `sessionId` is reused as the log filename `debug-<session-id>.log`.

**What was already instrumented (do not rip out until CK is understood)**

| Location | Purpose |
|----------|---------|
| `.../bayes-cast/launch/train_edm.py` after `print("model_kwargs", ...)` | **H5:** one NDJSON line — `torch` / `xformers` versions, HIP/CUDA strings, `FusedAttn_option`, `data_type`, `gpu_type`, `ORBIT2_FUSED_ATTN` (rank 0 only). |
| `.../bayes-cast/src/climate_learn/models/hub/components/attention.py` — `_agent_dbg` | **H1/H3:** `Attention` and `CrossAttention` **pre_ck** (shapes, dtypes, `dropout_p`, `N_a`/`N_i` for cross) immediately before `memory_efficient_attention(..., op=MemoryEfficientAttentionCkOp)`; **post_ck** after a successful return. **Rank 0 only** when `dist.is_initialized()` to avoid interleaved NDJSON from 8 ranks. |

**Hypotheses to resolve from the next log (first evidence wins)**

1. **H1** — Self-attention CK bad layout/shapes: last line `Attention.pre_ck`, no `Attention.post_ck`.
2. **H2** — `dropout_p` / training vs CK: correlate with fields on last `pre_ck`.
3. **H3** — Cross-attention CK when `N_a ≠ N_i`: last line `CrossAttention.pre_ck`, no `CrossAttention.post_ck`.
4. **H4** — Dtype: `CrossAttention` does `q, k = q.to(x.dtype), k.to(x.dtype)` but **not** `v`; if log shows `v_dtype != q_dtype` before SIGSEGV, add **`v = v.to(x.dtype)`** before CK (then post-fix verification run **with logs still on**).
5. **H5** — Stack skew: use `train_edm.py:post_model_kwargs` line for versions.

**Operational checklist for the next session**

1. In sbatch or env: `export DEBUG_AGENT_LOG=/path/on/shared/fs/agent-ck.ndjson` (must be visible inside the container).
2. Run minimal repro: `ORBIT2_DATA_TYPE=bfloat16`, `ORBIT2_FUSED_ATTN=CK`, small batch.
3. After job ends: copy `agent-ck.ndjson` → workspace **`<repo-root>/.cursor/debug-<session-id>.log`** (or paste last ~50 lines into chat).
4. Until logs prove otherwise, **`ORBIT2_FUSED_ATTN=DEFAULT`** (SDPA) remains the safe default in Studio sbatch for AMD+bf16.

**Symptom reminder:** CK + bf16 jobs ended with **`Segmentation fault (core dumped)`** on all ranks (no Python traceback). Example slurm text may live under `.../outputs/train/batch-cal-edm/*bf16*.out`.

### NEW (2026-06-11) — bf16 blocked even with SDPA (`FusedAttn.DEFAULT`): `var_agg` proj GEMM `hipErrorInvalidValue`

**This is broader than the CK SIGSEGV above:** with the *safe* default **`ORBIT2_FUSED_ATTN=DEFAULT`** (PyTorch SDPA), **bf16 EDM training still crashes** on the **first `training_step`**:

```
File "/orbit2/src/climate_learn/models/hub/edm.py", line 316, in aggregate_variables
  x = self.var_agg(var_query, x)
File "/orbit2/src/climate_learn/models/hub/components/attention.py", line 280, in forward
  x = self.proj(x)            # nn.Linear(256, 256)
torch.AcceleratorError: HIP error: invalid argument (hipErrorInvalidValue)
```

- **Reproduced across 6 jobs (lux MI355X, `pytorch_rocm7.2.2` SIF + ORBIT-2 overlay):**
  - **9228/9229/9230** — bf16 batch 1024/2048/4096 → crash. (Also confirmed restaged ERA5 lifted the old ~1704 sample cap: batch 1024 → `y.shape[0]=1024`; batch ≥2048 → caps at **1704** = total train samples.)
  - **9231** — bf16 batch **256** → **same crash** (not large-M dependent).
  - **9232** — bf16 batch 1024 + **`TORCH_BLAS_PREFER_HIPBLASLT=0`** (rocBLAS fallback) → **same crash** (not hipBLASLt-specific).
  - **9233** — bf16 batch 256 + **`HIP_LAUNCH_BLOCKING=1` / `AMD_SERIALIZE_KERNEL=3`** → **same crash, synchronous** → the failing op is genuinely the **`var_agg` output `proj` `F.linear`**, not an async misattribution.
  - **9234** — **float32** batch 256, identical config → **COMPLETED**. ✅
- **Shape at failure:** `aggregate_variables` (edm.py:309) flattens to `(B·H·L, V, D)`; after var cross-attention `proj` runs on `(B·H·L, 1, 256)` → GEMM **M = batch·648, K=256, N=256** in bf16. fp32 identical GEMM works.
- **Initial (wrong) verdict:** looked like a stack-level bf16 GEMM bug. **Superseded — see RESOLVED below.**

#### RESOLVED (2026-06-11) — root cause: ROCm **Flash** SDPA 65535 batch-grid cap in `var_agg`; fix: force EFFICIENT/MATH

**Root cause (isolated with a 1-GPU op-replica probe, no data):**
- The standalone bf16 **`proj` GEMM passes** even at M=663552 → the Linear is *not* the problem (the async HIP error was misattributed to `proj` line 280).
- A **faithful `CrossAttention` replica** reproduces the crash: bf16 **PASS at batch ≤64** (B=batch·648=41472), **FAIL at batch ≥128** (B=82944). The trigger is the **collapsed attention batch dim `B = batch·tokens`**: `aggregate_variables` flattens `(B,History,L)` into the SDPA batch dim, and ROCm **Flash** attention maps that to a HIP grid dimension capped at **65535** (`65535/648 ≈ 101` → matches batch 64 pass / 128 fail).
- **Backend sweep at batch 128/256 (bf16):** `default`(Flash) **FAIL**; **`math` PASS**, **`efficient` (EFFICIENT_ATTENTION) PASS**, **`eager` PASS**. fp32 "works" only because it dispatches to math, not Flash.
- **Independently corroborated by the upstream PyTorch SDPA benchmark** ([`examples/upstream_pytorch_sdpa_benchmark.py`](../../examples/upstream_pytorch_sdpa_benchmark.py) `--orbit-varagg`, which replicates the var_agg shape `batch_size=batch*648, q_len=1, kv_len=N_i`):

  | backend | batch 64 (B=41472) | batch 128 (B=82944) | batch 256 (B=165888) |
  |---|---|---|---|
  | default | PASS | **FAIL** | **FAIL** |
  | flash   | PASS | **FAIL** | **FAIL** |
  | efficient | PASS | PASS | PASS |
  | math    | PASS | PASS | PASS |

  Confirms `default ≡ flash` for these shapes and the Flash 65535 batch-grid cap (`64·648=41472` ok, `128·648=82944` fails). Run: `python3 upstream_pytorch_sdpa_benchmark.py --orbit-varagg --orbit-varagg-batches 64,128,256 --orbit-sweep-backends default,flash,efficient,math` (1 GPU; `pip install --target=/tmp/... tabulate tqdm` first per §"Apptainer venv").

**Fix (compute-side clone `src/climate_learn/models/hub/components/attention.py`):** wrap **all four** SDPA call sites — **both `Attention` (self) and `CrossAttention`**, DEFAULT + CK branches — in `with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])` to avoid Flash. MATH is fallback-only (never selected for supported shapes).

  *Why global, not just CrossAttention:* the [`--orbit-selfattn`](../../examples/upstream_pytorch_sdpa_benchmark.py) benchmark (q=kv=648, heads=8, bf16) shows **EFFICIENT within ±3–5% of Flash** at batch 64/256/1024 (efficient faster at 64; flash faster fwd by ~1–2% at large batch; backward a wash), so unifying on EFFICIENT removes the 65535 grid foot-gun at ~noise-level perf cost. **MATH is 10–15× slower** (≈15 vs ≈290 TFLOPS) so it must stay fallback-only.

  | self-attn fwd/bwd µs | batch 64 | batch 256 | batch 1024 |
  |---|---|---|---|
  | flash | 123 / 449 | 408 / 1516 | 1515 / 5940 |
  | efficient | 111 / 408 | 419 / 1514 | 1534 / 5909 |
  | math | — | — | 29220 / 30940 |

**Validated end-to-end:** job **9238** = the exact bf16 batch-256 config that crashed as **9231** now **COMPLETED (exit 0)**, `y.shape=[256,3,108,216]`, decreasing loss (`batch_idx 0→0.971, 1→0.963, 2→0.930, …`), full 6 epochs. Global fix (both `Attention`+`CrossAttention` → EFFICIENT) re-validated by job **9257** (bf16 **b1024**, COMPLETED, no errors).

**CK-flash flags are inert on this wheel:** setting `TORCH_ROCM_FA_PREFER_CK=1` + `USE_CK_FLASH_ATTENTION=1` does **not** change anything — the flash/efficient kernel stays AOTriton `attn_fwd`, flash still crashes at B>65535 (var_agg batch 128), and self-attn perf is unchanged. `USE_CK_FLASH_ATTENTION` is a **build-time** flag and this `pytorch_rocm7.2.2` wheel wasn't built with CK flash, so `TORCH_ROCM_FA_PREFER_CK` has nothing to prefer. Getting CK flash would require rebuilding PyTorch with `USE_CK_FLASH_ATTENTION=1` — not worth it given EFFICIENT (AOTriton) already matches Flash perf and has no grid cap.

**Caveats / follow-ups:**
- **`run_fom_extractor.py` / `parse_training_log.py` gap:** completed EDM runs log per-step loss as `epoch: N batch_idx M loss tensor(L)` and there's **no `Batch N: … seconds`** line in this build → the parser reports `epoch_records=0`, `final_loss=null`, and no `steady_batch_time_s`. **Throughput/loss FOMs need a parser update** to read the `epoch:/batch_idx/loss` format (and to time steps from omnistat or added timing).
- **HBM saturation vs data cap:** dataset has **~1704 train samples** (batch ≥1704 caps). bf16 at the cap will use roughly half the fp32 footprint (fp32@1704 ≈ 183 GiB ≈ 64%), so **bf16 alone won't reach 85–90% at 1704 samples** — stage more ERA5 ([`STAGING_ERA5_FOR_HBM.md`](../perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md)) to push higher.

### xFormers CK probe (MI355, `pytorch_rocm7.2.2` SIF + ORBIT-2 overlay)

**`python3 -m xformers.info`:** `memory_efficient_attention.ckF`, `.ckB`, `.ck_splitKF` report **available** (Cutlass/FA paths unavailable on AMD, expected).

**PyTorch wheel index (`https://download.pytorch.org/whl/rocm7.2`):** `pip index versions xformers` listed only **`0.0.34`** and **`0.0.35`** — no newer wheel than overlay **0.0.35** without changing ROCm/PyTorch image.

**A one-GPU CK probe (bf16) SIGSEGVs (exit 139) on every case** — `small`, `asym`, `long_p0`, `long_p01` — after printing the case header, i.e. **`memory_efficient_attention` + `MemoryEfficientAttentionCkOp` + bf16** fails even on tiny tensors. PyTorch SDPA on the same shapes **succeeds**.

**Implication:** do not treat **`xformers.info` “ckF available”** as safe for training MEA on this stack; keep **`ORBIT2_FUSED_ATTN=DEFAULT`** (SDPA) unless a new wheel/image is validated.

### PyTorch SDPA benchmark (upstream `sdpa.py`)

Studio copy: [`examples/upstream_pytorch_sdpa_benchmark.py`](../../examples/upstream_pytorch_sdpa_benchmark.py) — exercises **`torch.nn.functional.scaled_dot_product_attention`** under `sdpa_kernel(SDPBackend)` (same idea as [PyTorch `benchmarks/transformer/sdpa.py`](https://github.com/pytorch/pytorch/blob/main/benchmarks/transformer/sdpa.py)).

**Naming (corrected 2026-06-11):** the old CLI label **`ck`** for `--orbit-sweep-backends` was **removed** — it mapped to `SDPBackend.EFFICIENT_ATTENTION`, which on this ROCm build is **AOTriton**, *not* Composable Kernel. **Use `efficient`.** Verified by profiling: both `flash` and `efficient` dispatch to the same AOTriton kernel **`attn_fwd`**; `math` is the unfused reference (`Cijk_...` Tensile GEMMs + `softmax_warp_forward`). Real **CK** exists only in the xFormers path (`--orbit-include-xformers-ck` / `--orbit-xformers-modes ck` → `MemoryEfficientAttentionCkOp`, the one that SIGSEGVs). Flash vs efficient differ only in the **grid-launch wrapper** (flash maps the batch dim to a 65535-capped grid dim; efficient does not) — same `attn_fwd` math kernel.

**xFormers CK on generic shapes (no ORBIT code):** pass **`--orbit-include-xformers-ck`** with **`--orbit-micro`**. Each case runs in a **fresh subprocess** so a child **SIGSEGV** is reported as `KILLED_BY_SIGNAL` / `SIGSEGV` in the summary table while PyTorch SDPA results (already printed) stay intact. Tune with **`--orbit-xformers-modes ck,dispatch`**, **`--orbit-xformers-dropout 0.0,0.1`**, **`--orbit-xformers-timeout SEC`**.

**Validated (MI355X, SLURM job **9207**, `pytorch_rocm7.2.2` SIF + ORBIT-2 overlay, 2026-06-11):** **`--orbit-micro`** SDPA sweeps (**`default`**, **`math`**, **`efficient`**, CLI label **`ck`** → `SDPBackend.EFFICIENT_ATTENTION`, **`flash`**) all **completed** for bf16 shapes 128×128 and 648×648. **`--orbit-include-xformers-ck`** — every row (**`micro_128_sym`**, **`micro_648_sym`**, **`asym_1_vs_6`** × **`ck`** and **`dispatch`**, `dropout_p=0`) reported **`KILLED_BY_SIGNAL` / `SIGSEGV`** (standalone tensors; no Bayes-CAST). Example CSV dir: **`$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/sdpa-ck-microbench-1781204729/`**.

**Apptainer venv:** `pip install --user` fails (“User site-packages are not visible in this virtualenv”). Use **`pip install -q --target=/tmp/orbit2_sdpa_pip tabulate tqdm`** and **`export PYTHONPATH=/tmp/orbit2_sdpa_pip:$PYTHONPATH`** before running the script.

**One GPU, bind-mount Studio `examples/`** (adjust `AI4S_*` paths):

```bash
export AI4S_SHARED_DIR=/path/to/shared
STUDIO_ORBIT2_EXAMPLES=/path/to/ai4science-studio/earth_science/models/ORBIT-2/examples
srun -N1 -n1 --gres=gpu:1 apptainer exec --rocm \
  --overlay "${ORBIT2_OVERLAY:?}":ro \
  --bind "$STUDIO_ORBIT2_EXAMPLES":/sdpa-bench \
  "${ORBIT2_SIF:?}" bash -lc '
    set -e
    PIP_LOCAL=/tmp/orbit2_sdpa_pip
    mkdir -p "$PIP_LOCAL"
    python3 -c "import tabulate, tqdm" 2>/dev/null || pip install -q --target="$PIP_LOCAL" tabulate tqdm
    export PYTHONPATH="$PIP_LOCAL:${PYTHONPATH:-}"
    export ORBIT2_SDPA_CSV_DIR="${ORBIT2_SDPA_CSV_DIR:-/tmp/orbit2_sdpa_benchmark_results}"
    mkdir -p "$ORBIT2_SDPA_CSV_DIR"
    cd /sdpa-bench
    python3 upstream_pytorch_sdpa_benchmark.py --orbit-micro \
      --orbit-sweep-backends default,math,efficient,ck,flash \
      --orbit-include-xformers-ck \
      --orbit-xformers-modes ck,dispatch \
      --orbit-xformers-dropout 0.0,0.1
  '
```

Expect **`math`** and often **`default`** / **`efficient`** to complete on ROCm; **`flash`** may skip or error if Flash is unavailable — compare timings, not only success/fail. CSV timestamps land in **`ORBIT2_SDPA_CSV_DIR`**.

## Next work (after scaling baseline)

- **Run the sysopt loop:** [`recipes/perf-optimizer-loop/README.md`](../recipes/perf-optimizer-loop/README.md) — `examples/run_optimizer_loop.sh`, [`lever_catalog.yaml`](../recipes/perf-optimizer-loop/lever_catalog.yaml), staging + bf16 sweep ([`STAGING_ERA5_FOR_HBM.md`](../recipes/perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md), [`examples/sweep_orbit2_batch_bf16_amd.sh`](../../examples/sweep_orbit2_batch_bf16_amd.sh)).
- Rebuild overlay; re-test `bfloat16` + CK attn when `xformers.ops` is available.
- Stage **2.5_arcmin** PRISM targets for scientifically meaningful loss (not same-dir timing).

- **One-node GPU baseline (pre-sysopt):** [one-node-gpu-baseline.md](one-node-gpu-baseline.md) — batch sweep, Omnistat + TraceLens evidence, `report_orbit2_gpu_baseline.py` for `baseline_report.md`.

## Key paths (parameterize with `AI4S_SHARED_DIR`)

- Data: `$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin` (PRISM same-dir) or `.../era5/1.0_deg` (ERA5 same-dir sanity template).
- Overlay: `$AI4S_SHARED_DIR/models/ORBIT-2/overlays/orbit2-overlay.img`
- Perf runs: `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/`
- Training logs (recommended): `$AI4S_SHARED_DIR/models/ORBIT-2/outputs/train/logs/`
- Omnistat / TraceLens tools: set **`OMNIHUB_TOOLS_DIR`** to your site-local checkout (see root `.cursor/skills/ai4science-perf-analysis/SKILL.md` and `.cluster-config.yaml` key `omnihub.tools_dir` if your site maintains one).

## Operational source tree

- Set **`ORBIT2_ROOT`** to your clone (public **ORBIT-2** or institutional **Bayes-CAST**). Scripts bind-mount it at **`/orbit2`**.
- **`launch_diffusion.sh`:** if `$ORBIT2_ROOT/launch_diffusion.sh` or `examples/launch_diffusion.sh` exists, ranks **`exec bash …/launch_diffusion.sh /config/config.yaml`** after optional `ORBIT2_RANK_PRE_TRAIN_HOOK` (via `orbit2_rank_hook_runner.py`). Override path with **`ORBIT2_LAUNCH_SCRIPT`** (must live under `ORBIT2_ROOT`).
- **`manifest.json`** (perf runs): includes `git_sha`, `git_branch`, `git_remote_origin`, `rendered_config`, `parallelism.fsdp` / `parallelism.simple_ddp`, `global_batch_size`, `runtime_seconds`, `orbit2_batch_size`, `total_ranks`, `max_epochs`, `data_type`, and `workload` (`bayes_edm` vs `intermediate_downscaling`).
