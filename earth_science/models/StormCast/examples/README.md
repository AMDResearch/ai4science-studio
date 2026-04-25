# StormCast examples

Ready-to-run scripts for [StormCast](../README.md).  All scripts call
`earth2studio` directly — no code needs to be copied out of a blog post.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_stormcast.py`](preflight_stormcast.py) | Check your environment before submitting jobs |
| [`run_inference.py`](run_inference.py) | Deterministic single-member forecast |
| [`run_ensemble.py`](run_ensemble.py) | Multi-member ensemble forecast |
| [`docker_run.sh`](docker_run.sh) | Docker launcher for local workstations and interactive nodes |
| [`sbatch_inference_amd.sh`](sbatch_inference_amd.sh) | SLURM driver for `run_inference.py` on AMD Instinct |
| [`sbatch_ensemble_amd.sh`](sbatch_ensemble_amd.sh) | SLURM driver for `run_ensemble.py` on MI300X |

## Quick start — Docker (local workstation / interactive node)

```bash
# Deterministic forecast — 6 steps from 2025-01-01 06Z
./docker_run.sh inference --start 2025-01-01T06 --steps 6

# Ensemble — 4 members, 12 steps from 2025-08-09 12Z
./docker_run.sh ensemble --start 2025-08-09T12 --steps 12 --members 4

# Interactive shell inside the container
./docker_run.sh shell

# Stop and remove the container when done
./docker_run.sh stop
```

The script auto-detects the AMD Container Toolkit; if unavailable it falls
back to passing `/dev/kfd` and `/dev/dri` directly.

## Quick start — bare metal

```bash
# 1. Install
pip install "earth2studio[stormcast]" cartopy

# 2. Check environment
python preflight_stormcast.py

# 3a. Deterministic forecast
python run_inference.py --start 2025-01-01T06 --steps 6

# 3b. Ensemble forecast
python run_ensemble.py --start 2025-08-09T12 --steps 12 --members 4
```

## SLURM submission

Edit the `#SBATCH` directives (`--partition`, `--account`) in the relevant
script, then submit:

```bash
# Deterministic
SC_START=2025-01-01T06 SC_STEPS=6 sbatch sbatch_inference_amd.sh

# Ensemble
SC_START=2025-08-09T12 SC_STEPS=12 SC_MEMBERS=4 sbatch sbatch_ensemble_amd.sh

```

Set `SC_SIF=/path/to/stormcast.sif` to run inside a Singularity/Apptainer
container built from the ROCm PyTorch image:

```
rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

## Output

Both run scripts write a Zarr store under `outputs/` by default:

| Script | Default output |
|--------|---------------|
| `run_inference.py` | `outputs/pred-<YYYY-MM-DD>.zarr` |
| `run_ensemble.py` | `outputs/ens-<YYYY-MM-DD>.zarr` |

Override with `--output /your/path.zarr`.  The Zarr backend does not support
overwriting — delete the path before re-running.
