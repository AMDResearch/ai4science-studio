---
name: ai4science-earth-science
description: Applies to Earth-system ML in AI4Science Studio—climate, weather, geospatial data, and model folders under earth_science/models/. Includes validated AMD/ROCm HPC patterns for earth science models.
---

# Earth science domain

## Scope

The `earth_science/` domain covers **climate**, **weather**, and broader **Earth-system** machine learning: gridded fields, forecasting, downscaling, remote sensing, geospatial tensors, reanalysis-style inputs, etc. Do **not** create separate top-level climate or weather trees.

## Layout

- Models: `earth_science/models/<model-slug>/`
- Template: `_template/` (repo root)
- Conventions: `earth_science/models/README.md`

## Agent guidance

- Place new Earth-related HF models under **`earth_science/models/`** unless the model clearly fits another domain better (e.g. pure protein LM → `protein_folding/`).
- Recipes should state **spatial/temporal resolution**, **coordinate conventions** if relevant, and **data sources** (ERA5, satellite products, etc.) without bundling large raw archives in git.
- When suggesting AMD-specific notes, keep them **optional** and tied to tested stack versions (e.g. PyTorch + ROCm).
- **Institutional AMD** clusters and **data staging** (**Globus**, Constellation DOI pages, Hugging Face Hub CLI) onto shared filesystems are in-scope for recipe text; align guidance with **gridded** / reanalysis-style datasets and citation requirements.
- For models whose **authoritative code** lives on GitHub (e.g. **ORBIT-2** under `earth_science/models/ORBIT-2/`), the Studio may add **`examples/`** with **thin** Python or SLURM scripts that **delegate** to upstream entry points (same `cwd` / `PYTHONPATH` / scheduler layout upstream expects). Do **not** copy large upstream training or distributed inference files into Studio—wrappers plus recipe links stay maintainable.
- **Distributed inference** on cluster jobs must match upstream assumptions (e.g. SLURM task count vs YAML **parallelism** product). Site-specific **partition** and **account** names (e.g. HPCFund-style queues) belong in comments or placeholders, not hard-coded secrets.

## Validated AMD HPC patterns for earth science models

### StormCast (earth2studio)

- **No local data needed:** earth2studio DataSource classes fetch HRRR/GFS live from NOAA HTTPS archives. Compute nodes need outbound HTTPS to NOAA — no pre-staging required.
- **Overlay size:** 4 GB ext3; ~1.7 GB content (earth2studio[stormcast] + cartopy, stripped of torch)
- **ZarrBackend pitfall:** `run_inference.py` raises an error if the output zarr already exists. Always `rm -rf <output>.zarr` before each run in the sbatch script.
- **No CUDA packages:** install earth2studio with `--no-deps` then add deps manually; physicsnemo and timm also need `--no-deps` (they declare torch, which would pull CUDA torch).

### ORBIT-2 (pytorch-lightning + xformers + MPI)

- **Synthetic data generator:** `make_synthetic_data.py` generates a ~2 MB synthetic dataset (low_res + high_res NetCDF shards) so the full pipeline can be smoke-tested without ERA5/PRISM data. Use `ORBIT2_USE_SYNTHETIC=1` in the sbatch script.
- **Overlay size:** 7 GB ext3; ~2 GB content (pytorch-lightning, xformers, mpi4py, wandb, tensorboard, etc.)
- **pytorch-lightning must be installed with `--no-deps`** — it declares `torch` as a dep; pip resolves CUDA torch from PyPI which silently replaces the ROCm torch in PYTHONPATH.
- **xformers for ROCm:** install from the PyTorch ROCm wheel index, not PyPI. PyPI xformers links against `libcudart.so.12`. Use `--index-url https://download.pytorch.org/whl/${ROCM_WHL_TAG}` with `--no-deps`.
- **xFormers MEA vs `xformers.info` on MI355 (2026-06):** On `pytorch_rocm7.2.2` SIF + ORBIT-2 overlay, `python3 -m xformers.info` reports **`memory_efficient_attention.ckF` available**, but **`xformers.ops.memory_efficient_attention` (even with no explicit `op`, bf16, tiny shapes) can SIGSEGV**; `torch.nn.functional.scaled_dot_product_attention` is fine. The **`rocm7.2`** PyTorch wheel index only published **xformers 0.0.34 and 0.0.35** at probe time — no newer wheel to “upgrade CK” without changing ROCm/PyTorch image. Use **`ORBIT2_FUSED_ATTN=DEFAULT`** / SDPA-style paths for production. **Do not conflate** that path with PyTorch’s **`SDPBackend.EFFICIENT_ATTENTION`** (Studio benchmark label `ck` in [`upstream_pytorch_sdpa_benchmark.py`](../../../earth_science/models/ORBIT-2/examples/upstream_pytorch_sdpa_benchmark.py) — upstream [`sdpa.py`](https://github.com/pytorch/pytorch/blob/main/benchmarks/transformer/sdpa.py)); the latter is SDPA-only and is the right knob to compare “efficient vs math” on ROCm. To show segfaults are **stack/shape-generic** rather than Bayes-CAST-specific, run the same script with **`--orbit-include-xformers-ck`** (subprocess-isolated xFormers **`MemoryEfficientAttentionCkOp`** on micro + asymmetric cases).
- **xformers.components shim:** ORBIT-2 imports `xformers.components.attention.core.scaled_dot_product_attention`. This subpackage was removed in xformers ≥0.0.28. Write a three-file shim after install:
  ```python
  (xf / "components" / "__init__.py").write_text("")
  (xf / "components" / "attention" / "__init__.py").write_text("")
  (xf / "components" / "attention" / "core.py").write_text(
      "import torch.nn.functional as F\n\n"
      "def scaled_dot_product_attention(q, k, v, att_mask=None, dropout=0.0):\n"
      "    return F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask, dropout_p=dropout)\n"
  )
  ```
- **Distributed launch:** `srun --mpi=pmix apptainer exec ... --env HOSTNAME="$MASTER_ADDR"` — see studio SKILL.md for full pattern. Confirmed working for 1-GPU and 8-GPU.
- **Data blocker:** ORBIT-2 real training data lives on Frontier Lustre (`/lustre/orion/<YOUR_PROJECT>/...`) — not publicly accessible. Synthetic data covers the smoke-test path.
- **Training on institutional clusters:** `sbatch_train_amd.sh` prefers **`launch/train_edm.py`** (Bayes-CAST EDM: **`exec python3 train_edm.py /config/config.yaml`** from **`/orbit2/launch`** — upstream **`launch/launch_diffusion.sh` is not used**: OLCF-style wrapper, ignores argv, nests `srun`). Otherwise, if **`launch_diffusion.sh`** exists under `ORBIT2_ROOT`, ranks run it after **`orbit2_rank_hook_runner.py`**; else **`run_orbit2_train.py`** wraps `intermediate_downscaling.py` (gptl4py stub, FusedAttn fallback, batch cap). For **1 node × 8 GPUs**, rendered YAML defaults to **`fsdp=8`**, **`simple_ddp=1`** unless `ORBIT2_FSDP` / `ORBIT2_SIMPLE_DDP` are set. Set `ORBIT2_CONFIG_TEMPLATE=interm_8m_era5.yaml` or **`edm_8m_era5_1x8.yaml`** (Bayes EDM) and point `ORBIT2_DATA_ROOT` at staged ERA5 1.0° same-dir NPZ.
- **Training landmines (2026-06):** stub `gptl4py` in `run_orbit2_train.py`; overlay may lack `xformers.ops` → **`ORBIT2_FUSED_ATTN=DEFAULT`** in sbatch. **`ORBIT2_DATA_TYPE`** defaults **`bfloat16`** (matches upstream **`configs/interm_8m.yaml`** and Bayes-CAST-style **`edm_8m_era5.yaml`**); use **`float32`** only to isolate ROCm/hipBLAS issues. `max_epochs` must be ≥ 2; batch cap patches module before `main()` not via `runpy.run_path`.
- **bf16 + ROCm Flash SDPA 65535 grid cap (2026-06, FIXED):** Bayes-CAST EDM `var_agg`/`temporal_agg` cross-attention flattens `(B,History,L)` into the SDPA **batch** dim (`B = batch·648`), and ROCm **Flash** SDPA caps that dim at **65535** → bf16 crashes with `HIP error: invalid argument` at **batch ≥ ~128** (batch 64 OK). Fix applied **globally** to the compute-side clone: all `Attention` + `CrossAttention` SDPA calls wrapped in `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])` (MATH fallback-only). Going global is safe because `--orbit-selfattn` benchmarks EFFICIENT within ±3–5% of Flash on the main 648-token self-attention. Validated job 9238 (bf16 b256, decreasing loss). **Generalizes** to any model folding extra dims into the attention batch.
- **Perf/scaling:** `sbatch_train_perf_amd.sh` defaults **Bayes-CAST** template **`edm_8m_era5_1x8.yaml`** + ERA5 1.0° data; **`ORBIT2_ROOT`** prefers **`…/code/bayes-cast`** when present. Bayes EDM clears **`ORBIT2_RANK_PRE_TRAIN_HOOK`** only when unset — export an absolute hook path **before** `sbatch` for sysopt (`torch.compile`, etc.). **`ORBIT2_MAX_EPOCH=6`** default. See **[one-node GPU baseline](../../../earth_science/models/ORBIT-2/recipes/perf-analysis/one-node-gpu-baseline.md)**, **[perf-optimizer-loop](../../../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/README.md)** (throughput-primary loop), and **`ORBIT2_NUM_WORKERS`** / template token **`__NUM_WORKERS__`** in `edm_8m_era5_1x8.yaml` via `render_orbit2_config.py`.
- **Scaling sweep = WEAK scaling + HSDP (2026-06):** `examples/run_scaling_study.sh` (1/2/4/8 nodes) holds **per-rank batch fixed** so each GPU stays equally filled at every node count (global batch = `batch x 8 x N` grows) — this is **weak** scaling; `collate_scaling_study.py`'s "efficiency" is a comm-overlap / RCCL-bandwidth metric (`t_1node/t_Nnode`), not a classic strong-scaling speedup. (HydraGNN's recipe labels the same fixed-per-rank setup "strong-scaling efficiency" — mechanically it is weak.) Parallelism is **HSDP**: set **`ORBIT2_FSDP=8`** (shard within a node over XGMI) + **`ORBIT2_SIMPLE_DDP=N`** (data-parallel replica per node over IB/ANP); the script does this per node count via `ORBIT2_SCALING_FSDP`. **Do NOT** fall back to the render default `fsdp=N, simple_ddp=8` for multi-node — it shards weights across the slow inter-node fabric. EDM compute-saturated baseline: `edm_8m_era5_1x8.yaml`, bf16, per-rank batch **1024** (~4098 samples/s, ~21% HBM on the ~1704-sample ERA5 1.0° tree; reaching ~80% HBM needs more staged ERA5 — see STAGING_ERA5_FOR_HBM.md). **Pitfall — data cap inflates weak-scaling efficiency >1 at high node counts (validated 2026-06, jobs 14826-9):** with fixed per-rank batch the global batch grows as `1024 x 8 x N`, so on a small ERA5 tree the steps/epoch collapse (e.g. 57→27→**9**→9 steady batches at 1/2/4/8 nodes); steady step time over ~3 batches/epoch is noise-dominated and `throughput = global_batch/step_time` reports **fake super-linear efficiency (1.3x)** while the loss actually gets *worse* (too few steps to train). Only trust node counts where `steady_batch_count` stays large (here 1→2 nodes: eff ≈ 1.01, the real clean result). Fix = stage more ERA5 (Phase 2), not a code change. Always check the `steady_batch_count`/steps-per-epoch before quoting multi-node throughput.
- **Perf saturation vs ~10M params:** Default `interm_8m_prism.yaml`/`interm_8m_era5.yaml` uses `res_slimvit` @ `embed_dim` 256 / `depth` 6 → ~10M weights; **PRISM `spatial_resolution` 18** yields tiny activations per step — raise **batch**, use **ERA5 111×111** template for heavier forwards, or widen **`embed_dim`/`depth`** (with `embed_dim % num_heads == 0`) to approach MI-class MFMA; see [one-node-gpu-baseline.md](../../../earth_science/models/ORBIT-2/recipes/perf-analysis/one-node-gpu-baseline.md) §3. **VRAM-first sweep:** [BASELINE_LOCKIN.md](../../../earth_science/models/ORBIT-2/recipes/perf-analysis/BASELINE_LOCKIN.md) (batch sweep before widening ViT).
- **Multi-node scheduling:** Before large submits, check partition state (`sinfo`, site dashboards). Prefer **idle, healthy** nodes; use `--nodelist` / `--exclude` only per your site's policy. Do not stack two exclusive full-node GPU jobs on the same host.
- **GEMM-time bottleneck = the 3/4-channel conv stem/head, NOT attention/MLP (2026-06, MI355X):** Full TraceLens+Omnistat analyst/verifier run (`recipes/perf-analysis/agents/orchestrator_gemm_analysis.md` + `examples/run_gemm_analysis.sh`) on profiled EDM (bf16, batch 4096, ERA5) found one hipBLASLt GEMM `Cijk_…MT16x16x32` = **~43 % of the step at both 1- and 2-node** — the **im2col-lowered low-channel conv backward** (3/4-channel projections, 16×16 tile @ ~0.03 TFLOP/s). The tunable `nn.Linear` GEMMs are ~2 % and near-peak → **this is why TunableOp gave no uplift.** **Most promising lever = channel-padding in NCHW (timing-only −26 %, but TABLED pending a convergence study — see next bullet); `channels_last` is a DEAD END on ROCm 7.2.2 (also next bullet).** **Profile any EDM run** via `train_edm.py:_orbit2_make_profiler` (env-gated: set `ORBIT2_PROFILE_DIR`+`PROFILE_TARGET_EPOCH`; sbatch does this). **Trust raw kernel-name device-time, not TraceLens `report.xlsx:ops_summary_by_category`** — it mislabels conv-lowered GEMMs as `CONV_bwd` (a 2-node analyst claim was refuted on this). `examples/compare_trace_kernels.py` aggregates kernels tool-independently. Separately, **~43 % whole-job idle** (inter-step overhead, NOT comm-bound) is a second lever.
- **Conv channel-padding = TABLED timing-only candidate (−26 %, NOT adopted); `channels_last`/implicit-GEMM is a DEAD END (2026-06, MI355X/ROCm 7.2.2):** Env knob **`ORBIT2_CONV_PAD=N`** (`edm.py`, **default 0/off, keep off**) widens the `path2`/`refine` internal conv width to a tile-friendly N (=16) while keeping 3-channel I/O → collapses the starved `MT16x16x32` GEMM into `MT16x16x128`, **−25/26 % steady step time, −28 % GPU-kernel time** (timing-only A/B). **Tabled because it changes model capacity (wider convs) and convergence/quality was never validated** — "loss sanity" ≠ convergence. Do NOT enable for real training until a matched convergence study (final val loss/skill, steps-to-target, pad=0 vs pad=16) passes. **Do NOT stack `channels_last`** (knob `ORBIT2_CHANNELS_LAST`, default 0): on this build MIOpen has no tuned NHWC implicit-GEMM solver for these shapes and falls back to brute-force `naive_conv_…wrw_nhwc` (~88-91 % of kernel time) → **7-10× SLOWER** (cl_pad0 60.8 s, cl_pad16 80.1 s vs 6.38 s for nchw_pad16). The `im2col` (`Im2d2Col_v2`) in the NCHW path is NOT wasted overhead — it is the entry to the tuned hipBLASLt GEMM kernels; NHWC discards that path. **Stay NCHW + `ORBIT2_CONV_PAD`.**
- **ORNL Frontier MIOpen flags now default in the sbatch (validate per-cluster):** From bayes-cast `launch/launch_diffusion.sh`, ORNL **disables Winograd** + unbounds the multi-pass Winograd workspace, "tested many times on Frontier" (gfx90a/ROCm7.1.1). Ported to `sbatch_train_perf_amd.sh`/`sbatch_train_amd.sh`/`sbatch_infer_*.sh` as overridable defaults: `MIOPEN_DEBUG_CONV_WINOGRAD=0` (`ORBIT2_MIOPEN_CONV_WINOGRAD`), `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1`, `MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1`, `HSA_FORCE_FINE_GRAIN_PCIE=1`. `MIOPEN_DISABLE_CACHE=1`+`MIOPEN_USER_DB_PATH=/tmp/<job>` already matched ORNL (keep; do NOT persist the find-db). **Do NOT port** the Frontier Slingshot/Cray-specific env (`FI_CXI_*`, `NCCL_NET="AWS Libfabric"`, `librccl-net.so`, `NCCL_SOCKET_IFNAME=hsn0`, Frontier `LD_PRELOAD` paths) — they break non-Slingshot fabrics (e.g. IB/ionic).

### ArchesWeather (geoarches)

- **Not on PyPI:** install with `pip install "git+https://github.com/INRIA/geoarches.git"`.
- **Data blocker:** `encode_dataset` requires ERA5 zarr (~35 GB). Not live-fetchable — must be staged on cluster before this model can run end-to-end.
- **CLI syntax:** `geoarches.inference.encode_dataset` uses argparse (`--uids`, `--input-path`, `--output-path`), not Hydra `++` overrides.
- **Environment:** installs cleanly; checkpoint downloads from HF (`gcouairon/ArchesWeather`). Only the data is blocked.

### Aurora, GenCast, PanguWeather (ai-models framework)

- **Shared recipe repo:** all three use `silogen/ai-samples` (`ai4sciences/ai-weather-forecasting`).
- **Two Docker images:** Aurora uses `pytorchweather:latest` (PyTorch); GenCast and PanguWeather share `jaxweather:latest` (JAX).
- **CDS credentials required:** all three fetch ERA5 initial conditions from the Copernicus Climate Data Store. Users must configure `env_file` with their `CDSAPI_KEY`.
- **PanguWeather ONNX patch:** one-line patch adds the ROCm execution provider — handled automatically by the Docker image.
- **GenCast ensemble sizing:** `NUM_ENSEMBLE_MEMBERS` must be a multiple of the GPU count.
- **Aurora memory:** 0.1° resolution (~6× more grid points than 0.25°) demands the MI300X's full 192 GB HBM3.

### NeuralGCM (JAX + GCS)

- **No CDS/HF needed:** weights from GCS (`gs://neuralgcm/models/`) and ERA5 from public ARCO-ERA5 Zarr — both anonymous, no auth required.
- **JAX for ROCm:** requires a custom JAX ROCm wheel (`jaxlib-0.6.0`) from `ROCm/rocm-jax` GitHub releases, plus `jax-rocm7-pjrt` and `jax-rocm7-plugin`.
- **Docker image:** `rocm/dev-ubuntu-22.04:7.0.2-complete` — deps installed at container start by `docker_run.sh`.
- **`libdw1`:** required JAX runtime dependency on Ubuntu 22.04; installed via `apt install -y libdw1`.
- **Performance:** 0.7° ~110s, 1.4° ~48s, 2.8° ~33s for a 4-day forecast on MI300X.

## Overlay pip install pitfalls (common to all earth science models)

**`pip --target` ignores the SIF's installed packages** — it resolves everything fresh. When torch is a transitive dep, pip downloads a fresh ROCm torch wheel (~6 GB), filling the overlay before other packages install.

**Constraints file approach does NOT work:**
- The venv's torch has a local version string: `torch==2.10.0+rocm7.2.2.gitXXX`
- pip cannot match this against `>=2.5.0` from `--target` mode → `ResolutionImpossible`
- Bare package names in constraints files (e.g. `torchvision` with no `==X.Y`) are also invalid pip constraint syntax → same error

**Correct pattern — NFS staging + strip + copy:** see studio SKILL.md for full code. Use `--extra-index-url https://download.pytorch.org/whl/<ROCM_WHL_TAG>` (no constraint); ROCm torch has no `nvidia-*` CUDA co-package deps so the entire CUDA chain is never resolved.

## Validated dep quirks (earth science models)

- **`tensordict`** (used by StormCast via earth2studio) requires **`pyvers`** at runtime. `--no-deps` for tensordict is needed to avoid pulling CUDA torch, but `pyvers` must be installed separately alongside the other runtime deps.
- **StormCast no-overlay path**: uses the install-to-tmp + bind-mount pattern (see studio SKILL.md). Packages go to `/opt/stormcast-pkgs` via `--target` + bind-mount. `PYTHONPATH` must include this path in the inference `bash -c` block.
- **ORBIT-2 no-overlay path**: same pattern, packages at `/opt/orbit2-pkgs`. All three `apptainer exec` calls (checkpoint download, synthetic data gen, srun inference) must include `"${PKGDIR_BIND[@]}"`.

## Typical pitfalls

- Mixing incompatible projection or time semantics — document CRS and time axis assumptions.
- Omitted license for underlying **datasets** — mention dataset terms alongside the model license when recipes depend on them.
- Hardcoding HF repo filenames without verifying: use `list_repo_files()` to discover actual checkpoint names.
