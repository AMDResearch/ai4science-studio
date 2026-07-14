# GEMM-time attribution — Bayes-CAST EDM (bf16, ERA5 1.0°)

ORBIT-2 analog of HydraGNN's [`dispatch-attribution.md`](../../../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/dispatch-attribution.md): the durable "where does the step time go" finding for the EDM path, plus the levers tried against it (see [`lever_catalog.yaml`](lever_catalog.yaml) for the accept/reject record).

**Workload:** profiled EDM, bf16, batch 4096, ERA5 1.0°, 1- and 2-node (jobs 12562 / 12648).
**Driver:** [`agents/orchestrator_gemm_analysis.md`](../perf-analysis/agents/orchestrator_gemm_analysis.md) → [`examples/run_gemm_analysis.sh`](../../examples/run_gemm_analysis.sh) (TraceLens + Omnistat analyst/verifier).
**Report:** `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/gemm-analysis-<uuid>/GEMM_TIME_REPORT.md`.

## Attribution (authoritative for this workload)

- **The hot kernel is NOT attention/MLP.** A single hipBLASLt GEMM `Cijk_…MT16x16x32_MI16` is **~43%** of the step at *both* scales (scale-invariant). It is the **im2col-lowered low-channel conv backward** for the **3- and 4-channel stem/head projection convs**, stuck on a 16×16 macro-tile at ~0.03 TFLOP/s. The tunable `nn.Linear` bf16 GEMMs (256×256 tiles, 600–1074 TFLOP/s) are only ~2%. **This is why TunableOp gives no uplift** — it tunes already-optimal GEMMs and can't touch the conv-lowered one. `im2col` (`Im2d2Col_v2`) adds ~14%; conv **forward** is a `naive_conv` with **fp64 accumulate** (`..._ushort_double_ushort`) at ~3%.
- **TraceLens `report.xlsx:ops_summary_by_category` MISLABELS the conv-GEMM as `CONV_bwd`.** The 2-node analyst trusted it and claimed "naive conv 65.8%"; the verifier refuted it from raw `traceEvents` (naive_conv_bwd = 0.46%, GEMM = 50.7%). **Always rank by raw kernel-name device time, not the category rollup,** for conv-heavy models. [`examples/compare_trace_kernels.py`](../../examples/compare_trace_kernels.py) does this tool-independently.
- **~43–44% whole-job idle at both scales** (Omnistat util mean ~56%) is the second lever — it is structural inter-step overhead, **not** comm-bound (exposed comm 0.13%→0.47%; the Omnistat "comm-bound" claim was refuted by the trace).
- **EDM trainer has an env-gated profiler.** `train_edm.py:_orbit2_make_profiler` writes a rank-0 `*.pt.trace.json` for `PROFILE_TARGET_EPOCH` into `ORBIT2_PROFILE_DIR` (no-op unless set). This is what makes "where does GEMM time go" answerable for the EDM path (the old `res_slimvit` hook did not apply).

## Levers tried against the GEMM bottleneck

### ORNL Frontier MIOpen flags — performance-NEUTRAL (jobs 12997/12998 vs 12562)

From bayes-cast `launch/launch_diffusion.sh` (ORNL, gfx90a/ROCm7.1.1, "tested many times"): ORNL **disables Winograd** and unbounds the multi-pass Winograd workspace. Now defaults in `sbatch_train_perf_amd.sh` / `sbatch_train_amd.sh` / `sbatch_infer_*.sh` (override to A/B):

```
MIOPEN_DEBUG_CONV_WINOGRAD=0            # ORBIT2_MIOPEN_CONV_WINOGRAD (0=off ORNL default, 1=on)
MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1   # ORBIT2_MIOPEN_WINOGRAD_MPASS_WS_MAX
MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1   # ORBIT2_MIOPEN_MP_BD_WINOGRAD_WS_MAX
HSA_FORCE_FINE_GRAIN_PCIE=1
```

`MIOPEN_DISABLE_CACHE=1` + `MIOPEN_USER_DB_PATH=/tmp/<job>` already matched ORNL — keep them (do NOT "persist the find-db"; that deviates from the validated setup). **Deliberately NOT ported** (Frontier Slingshot/Cray-specific, would break non-Slingshot IB/ionic): `FI_CXI_*`, `NCCL_NET="AWS Libfabric"`, `NCCL_NET_PLUGIN=librccl-net.so`, `NCCL_SOCKET_IFNAME=hsn0`, Frontier `LD_PRELOAD` host paths.

**Result (2026-06-16): performance-NEUTRAL on MI355X** (steady 8.62→8.66→8.74 s, within ~1–2% noise; loss sanity OK). MIOpen never selected Winograd for the 3/4-channel convs (0% Winograd kernels in all arms), so toggling it changes nothing — the bottleneck is shape-driven, not algorithm-selection-driven. Keeping the flags as faithful, harmless defaults; they are **not** the lever here. Report: `perf-runs/miopen-ornl-validation-<uuid>/MIOPEN_ORNL_VALIDATION.md`.

### Conv channel-padding (`ORBIT2_CONV_PAD`) — TABLED, timing-only (NOT ADOPTED, jobs 14633/14634)

> **Status: tabled.** The −25% is a *timing-only* result. Channel-padding **changes model capacity** (wider internal convs ⇒ different params) and **has no convergence/quality study**, so it is **not adopted** and `ORBIT2_CONV_PAD` stays **default 0 (off)**. Treat the numbers below as a *potential* speedup pending a same-quality convergence check. Do not enable for real training runs until that check passes.

`edm.py` `path2`/`refine` convs run at out_channels(=3)/cnn_ratio(=4) at full resolution → the starved `Cijk_…MT16x16x32` GEMM @ ~0.03 TFLOP/s. New env knob **`ORBIT2_CONV_PAD=N`** (in `edm.py`, default 0 = original) widens those convs' INTERNAL width to N (tile-friendly) keeping the 3-channel I/O. Single-node A/B (identical bf16/4096/ERA5 config), N=16:

| Arm | steady_batch_time_s | total GPU-kernel ms | dominant `MT16x16x32` GEMM |
|---|---|---|---|
| control (pad=0) | 8.646 | 21082 | 9245 ms (43.9%) |
| **widened (pad=16)** | **6.498 (−24.8%)** | 15150 (−28%) | **gone** (top GEMM now `MT16x16x128`, 4123 ms) |

**~25% faster step, loss *sanity* holds (not convergence).** Confirms the attribution: low channel count forced the bad tile; widening makes it tile-friendly and the efficiency gain beats the extra FLOPs. **Why tabled:** "loss sanity" only checks the loss is finite/decreasing over a few epochs — it is **not** evidence the wider model reaches the same quality. Before adopting, run a matched convergence study (same data/schedule; compare final validation loss/skill and steps-to-target for pad=0 vs pad=16); only adopt if quality is equal-or-better. If revisited, the open timing question is **N=32** (top GEMM `MT16x16x128` at 27% still has headroom). Report: `perf-runs/conv-pad-validation-<uuid>/CONV_PAD_VALIDATION.md`.

### `channels_last` + implicit-GEMM — DEAD END on ROCm 7.2.2 (DISPROVEN, jobs 14645–14648)

Tested stacking `channels_last` on the padding idea as a clean 2×2 ({NCHW, channels_last} × {pad=0, pad=16}) via env knob **`ORBIT2_CHANNELS_LAST`** in `edm.py` (default 0; sets `MIOPEN_FIND_MODE=1` on the NHWC arms). **It is 7–10× SLOWER, not faster:**

| Arm | CONV_PAD | channels_last | steady_batch_time_s | vs control |
|---|---|---|---|---|
| nchw_pad0 (control) | 0 | off | 8.62 | — |
| **nchw_pad16** | 16 | off | **6.38** | **−26%** ✅ |
| cl_pad0 | 0 | on | 60.8 | +606% 💥 |
| cl_pad16 | 16 | on | 80.1 | +829% 💥 |

**Root cause (kernel trace):** MIOpen on this build has **no tuned NHWC implicit-GEMM solver** for these conv shapes. Forcing channels_last makes it fall back to the brute-force **`naive_conv_…wrw_nhwc`** direct-convolution kernel (~88–91% of GPU-kernel time). The `im2col` (`Im2d2Col_v2`, 14–21% in NCHW) is **not** wasted overhead — it is the price of reaching the tuned **hipBLASLt GEMM** kernels (`Cijk_…`, 31–54% in NCHW); NHWC discards that path. **Lesson: do NOT use `channels_last` for these low-channel convs on MI355X/ROCm 7.2.2.** The lever is NCHW + `ORBIT2_CONV_PAD`; channels_last is abandoned (knob kept default-off only so the dead end is reproducible). Report: `perf-runs/conv-layout-validation-<uuid>/CONV_LAYOUT_VALIDATION.md`.

## Reproduce

```bash
export AI4S_SHARED_DIR=/path/to/shared
export OMNIHUB_TOOLS_DIR=/shared/omnihub/tools
# 1- and 2-node profiled GEMM analysis (TraceLens + Omnistat analyst/verifier)
bash earth_science/models/ORBIT-2/examples/run_gemm_analysis.sh
# Tool-independent kernel ranking from a single trace:
python3 earth_science/models/ORBIT-2/examples/compare_trace_kernels.py \
    --trace $AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/traces/orbit2-epoch*-rank0.pt.trace.json
```
