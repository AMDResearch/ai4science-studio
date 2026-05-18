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
