# ORBIT-2 examples

Ready-to-run scripts for [ORBIT-2](../README.md). These wrap the upstream
[`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2) code
with AMD/ROCm container launch, overlay builds, and synthetic smoke tests.

## Files

The folder is flat (every script resolves siblings, configs, and the
`/examples` container bind-mount relative to its own directory), but the files
fall into the groups below.

### Setup & inference

| Script | Purpose |
|--------|---------|
| [`preflight_orbit2.py`](preflight_orbit2.py) | Check your environment before submitting jobs |
| [`run_visualize.py`](run_visualize.py) | Run upstream `visualize.py` with Studio env-var overrides |

### Container & launch drivers

| Script | Purpose |
|--------|---------|
| [`docker_run.sh`](docker_run.sh) | Docker launcher for local workstations and interactive nodes |
| [`build_overlay_amd.sh`](build_overlay_amd.sh) | One-time overlay build (pre-bakes pip deps, skips ~15 min per job) |
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | Multi-node training (Apptainer + MPI); defaults PRISM 10.0_arcmin + `interm_8m_prism.yaml` |
| [`sbatch_train_perf_amd.sh`](sbatch_train_perf_amd.sh) | 1-node × 8-GPU perf (Omnistat + profiler); defaults **ERA5 1.0°** + `edm_8m_era5_1x8.yaml`; multi-node: `sbatch --nodes=N …` |
| [`sbatch_infer_amd.sh`](sbatch_infer_amd.sh) | SLURM driver for inference on AMD Instinct (MI250X/MI300X/MI350X) |
| [`sbatch_infer_docker.sh`](sbatch_infer_docker.sh) | SLURM inference driver using Docker instead of Apptainer |

### In-container launch glue

Run on the host (`render_*`) or inside the container via the `/examples`
bind-mount; the sbatch drivers depend on these by name, so they stay flat.

| Script | Purpose |
|--------|---------|
| [`render_orbit2_config.py`](render_orbit2_config.py) | Render a per-job YAML from a template (parallelism, data root, trainer caps); invoked by the sbatch drivers |
| [`run_orbit2_train.py`](run_orbit2_train.py) | Studio training entry run inside the container (wraps upstream `intermediate_downscaling.py`; gptl4py stub, FusedAttn fallback, batch cap) |
| [`orbit2_rank_hook_runner.py`](orbit2_rank_hook_runner.py) | Runs an optional per-rank pre-train hook (`ORBIT2_RANK_PRE_TRAIN_HOOK`) before the shell launcher |
| [`orbit2_profiler_hook.py`](orbit2_profiler_hook.py) | Optional per-rank PyTorch-profiler hook (`res_slimvit` path) invoked via `ORBIT2_RANK_PRE_TRAIN_HOOK` |

### Configs (YAML)

Kept beside the sbatch drivers — they are resolved as `${SCRIPT_DIR}/${ORBIT2_CONFIG_TEMPLATE}`.

| Script | Purpose |
|--------|---------|
| [`interm_8m_prism.yaml`](interm_8m_prism.yaml) | PRISM config template (10.0_arcmin same-dir) |
| [`interm_8m_era5.yaml`](interm_8m_era5.yaml) | ERA5 1.0° config template (same-dir sanity / pipeline test) |
| [`interm_8m_synthetic.yaml`](interm_8m_synthetic.yaml) | Template YAML for synthetic mode |
| [`edm_8m_era5_1x8.yaml`](edm_8m_era5_1x8.yaml) | Bayes-CAST `edm_8m_era5.yaml` + **`fsdp=8`/`simple_ddp=1`** + `seq_par:1`; `ORBIT2_ROOT` → `code/bayes-cast` (auto if present) |

### Data prep

| Script | Purpose |
|--------|---------|
| [`make_synthetic_data.py`](make_synthetic_data.py) | Generate ~2 MB synthetic dataset for smoke tests |
| [`stage_era5_3x_symlink.sh`](stage_era5_3x_symlink.sh) | Symlink-replicate a staged ERA5 year (Nx) to break the small-corpus per-step batch cap for HBM saturation |

### Perf & scaling orchestration

| Script | Purpose |
|--------|---------|
| [`submit_perf_baseline_era5_amd.sh`](submit_perf_baseline_era5_amd.sh) | Thin `sbatch` wrapper (sets ERA5 defaults, forwards extra `sbatch` flags) |
| [`sweep_orbit2_batch_bf16_amd.sh`](sweep_orbit2_batch_bf16_amd.sh) | Submit multiple `ORBIT2_BATCH_SIZE` probes (bf16 + SDPA) for HBM saturation sweeps |
| [`run_scaling_study.sh`](run_scaling_study.sh) | Submit matched 1/2/4/8-node scaling sweep |
| [`run_2node_scaleout_loop.sh`](run_2node_scaleout_loop.sh) | Unattended 2-node scale-out lever loop; writes a 1-node-vs-2-node `REPORT.md` |
| [`run_gemm_analysis.sh`](run_gemm_analysis.sh) | Unattended GEMM-time bottleneck analysis (TraceLens + Omnistat analyst/verifier) at 1 and 2 nodes |
| [`run_optimizer_loop.sh`](run_optimizer_loop.sh) | Driver for the iterative **perf-optimizer-loop** (Claude CLI optional) |
| [`validate_orbit2_optimizer_loop_recipe.sh`](validate_orbit2_optimizer_loop_recipe.sh) | Repo smoke: required perf-optimizer-loop files exist |

### Post-processing & analysis

Host-side; run after a job against its logs / traces.

| Script | Purpose |
|--------|---------|
| [`parse_training_log.py`](parse_training_log.py) | Extract batch/epoch metrics from SLURM log (handles both `intermediate_downscaling.py` and Bayes-CAST `train_edm.py` `epoch:/batch_idx/tic4-tic1` formats) |
| [`run_fom_extractor.py`](run_fom_extractor.py) | Write `foms.json` (throughput + steady batch time + optional Omnistat PromQL via `ORBIT2_TSDB_URL`) |
| [`report_orbit2_gpu_baseline.py`](report_orbit2_gpu_baseline.py) | `baseline_report.md` / JSON for perf runs |
| [`orbit2_estimate_batch_from_memory.py`](orbit2_estimate_batch_from_memory.py) | Heuristic max-batch from two `memory_reserved` calibrations or SLURM logs (see [BASELINE_LOCKIN.md](../recipes/perf-analysis/BASELINE_LOCKIN.md)) |
| [`collate_scaling_study.py`](collate_scaling_study.py) | Aggregate a multi-node scaling sweep into `scaling_study` outputs |
| [`compare_trace_kernels.py`](compare_trace_kernels.py) | Tool-independent kernel aggregation from a `*.pt.trace.json` (rank by raw device time, not category rollup) |
| [`upstream_pytorch_sdpa_benchmark.py`](upstream_pytorch_sdpa_benchmark.py) | Vendored [PyTorch `benchmarks/transformer/sdpa.py`](https://github.com/pytorch/pytorch/blob/main/benchmarks/transformer/sdpa.py) with **`--orbit-micro`** + backend sweep (`ck` → `SDPBackend.EFFICIENT_ATTENTION`, **not** xFormers `MemoryEfficientAttentionCkOp`) for comparing SDPA backends off the critical path |

## Quick start — ERA5 1.0_deg sanity (new dataset)

Verifies the staged ERA5 NPZ tree loads and training runs (same-dir mode; loss not meaningful).

```bash
export AI4S_SHARED_DIR=/path/to/shared
export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
export ORBIT2_CONFIG_TEMPLATE=interm_8m_era5.yaml
export ORBIT2_MAX_EPOCH=2 ORBIT2_MAX_BATCHES=5 ORBIT2_BATCH_SIZE=2 ORBIT2_DATA_TYPE=float32
sbatch sbatch_train_amd.sh
```

For production ERA5→PRISM downscaling you will pair `era5/1.0_deg` (low) with `prism/2.5_arcmin` (high) once both splits and year shards align.

## Quick start — training (10.0_arcmin PRISM, scaling)

```bash
export AI4S_SHARED_DIR=/path/to/shared
export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin
export ORBIT2_MAX_EPOCH=2 ORBIT2_MAX_BATCHES=10 ORBIT2_DATA_TYPE=float32
sbatch sbatch_train_amd.sh

# Strong scaling sweep:
export SBATCH_PARTITION=... SBATCH_ACCOUNT=...
./run_scaling_study.sh
python3 collate_scaling_study.py --log-dir . --jobs <id1>,<id2>,... -o scaling_study
```

## Quick start — Docker

```bash
./docker_run.sh shell
# Inside the container, clone upstream and run
```

## Quick start — SLURM (synthetic smoke test)

```bash
export ORBIT2_ROOT=/path/to/ORBIT-2-clone
export ORBIT2_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
export ORBIT2_USE_SYNTHETIC=1
sbatch sbatch_infer_amd.sh
```

## Quick start — SLURM (real data)

```bash
export ORBIT2_ROOT=/path/to/ORBIT-2-clone
export ORBIT2_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
export ORBIT2_OVERLAY=/path/to/orbit2-overlay.img  # optional
export ORBIT2_CHECKPOINT=/path/to/model.ckpt
sbatch sbatch_infer_amd.sh
```

Set `ORBIT2_OVERLAY` to a pre-built overlay from `build_overlay_amd.sh` to
skip the ~15 min pip install on each job.
