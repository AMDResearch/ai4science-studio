# ORBIT-2 examples

Ready-to-run scripts for [ORBIT-2](../README.md). These wrap the upstream
[`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2) code
with AMD/ROCm container launch, overlay builds, and synthetic smoke tests.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_orbit2.py`](preflight_orbit2.py) | Check your environment before submitting jobs |
| [`run_visualize.py`](run_visualize.py) | Run upstream `visualize.py` with Studio env-var overrides |
| [`make_synthetic_data.py`](make_synthetic_data.py) | Generate ~2 MB synthetic dataset for smoke tests |
| [`docker_run.sh`](docker_run.sh) | Docker launcher for local workstations and interactive nodes |
| [`sbatch_infer_amd.sh`](sbatch_infer_amd.sh) | SLURM driver for inference on AMD Instinct (MI250X/MI300X/MI350X) |
| [`build_overlay_amd.sh`](build_overlay_amd.sh) | One-time overlay build (pre-bakes pip deps, skips ~15 min per job) |
| [`interm_8m_synthetic.yaml`](interm_8m_synthetic.yaml) | Template YAML for synthetic mode |
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | Multi-node training (Apptainer + MPI) |
| [`sbatch_train_perf_amd.sh`](sbatch_train_perf_amd.sh) | 2-node instrumented perf-analysis run |
| [`run_scaling_study.sh`](run_scaling_study.sh) | Submit matched 1/2/4/8-node scaling sweep |
| [`collate_scaling_study.py`](collate_scaling_study.py) | Parse scaling logs → CSV/MD table |
| [`parse_training_log.py`](parse_training_log.py) | Extract batch/epoch metrics from SLURM log |
| [`interm_8m_lux.yaml`](interm_8m_lux.yaml) | Lux config template (10.0_arcmin same-dir) |
| [`interm_8m_lux_era5.yaml`](interm_8m_lux_era5.yaml) | Lux config template (ERA5 1.0_deg same-dir sanity) |

## Quick start — ERA5 1.0_deg sanity (new dataset)

Verifies the staged ERA5 NPZ tree loads and training runs (same-dir mode; loss not meaningful).

```bash
export AI4S_SHARED_DIR=/path/to/shared
export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
export ORBIT2_CONFIG_TEMPLATE=interm_8m_lux_era5.yaml
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
