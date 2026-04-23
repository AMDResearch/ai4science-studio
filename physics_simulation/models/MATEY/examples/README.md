# MATEY examples

Ready-to-run scripts for [MATEY](../README.md). These wrap the upstream
[`ORNL/MATEY`](https://github.com/ORNL/MATEY) with AMD/ROCm container
launch, Apptainer SIF build, and SLURM support.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_matey.py`](preflight_matey.py) | Check your environment before submitting jobs |
| [`run_train.sh`](run_train.sh) | Single-GPU or DDP training with JHTDB demo config |
| [`run_inference.py`](run_inference.py) | Autoregressive rollout from initial condition |
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones MATEY, installs deps) |
| [`build_sif.sh`](build_sif.sh) | Build Apptainer SIF + overlay for HPC clusters |
| [`sbatch_train_mi300x.sh`](sbatch_train_mi300x.sh) | SLURM driver for training on AMD Instinct |

## Quick start — Docker

```bash
# Training with JHTDB demo
./docker_run.sh train

# Inference (requires trained checkpoint)
./docker_run.sh inference

# Interactive shell
./docker_run.sh shell
```

## Quick start — SLURM

```bash
# Edit #SBATCH directives first, then:
sbatch sbatch_train_mi300x.sh
```

## Environment variables (training)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MATEY_YAML` | Yes | -- | YAML configuration file path |
| `MATEY_RUN_NAME` | No | `my_run` | Run name for logging |
| `MATEY_CONFIG` | No | `basic_config` | Config preset name |
| `MATEY_DATA_DIR` | No | `/data/JHTDB` | Training data directory |
| `MATEY_EPOCHS` | No | `100` | Training epochs |
| `MATEY_BATCH_SIZE` | No | `4` | Per-GPU batch size |
| `MATEY_LR` | No | `1e-4` | Learning rate |
| `MATEY_USE_DDP` | No | `0` | Enable multi-GPU DDP |

## Environment variables (inference)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MATEY_CHECKPOINT` | Yes | -- | Path to trained `.pt` checkpoint |
| `MATEY_CONFIG` | Yes | -- | Training YAML config |
| `MATEY_INPUT` | Yes | -- | HDF5 initial condition file |
| `MATEY_STEPS` | No | `100` | Rollout steps |
| `MATEY_OUTPUT` | No | `outputs/rollout.h5` | Output HDF5 path |
