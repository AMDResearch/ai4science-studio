# NeuralGCM examples

Ready-to-run scripts for [NeuralGCM](../README.md).  All scripts fetch model
weights directly from GCS and ERA5 initial conditions from the public ARCO-ERA5
Zarr store — no manual downloads required.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_neuralgcm.py`](preflight_neuralgcm.py) | Check your environment before submitting jobs |
| [`run_inference.py`](run_inference.py) | Deterministic or stochastic global forecast from ERA5 |
| [`docker_run.sh`](docker_run.sh) | Docker launcher for local workstations and interactive nodes |
| [`sbatch_inference_amd.sh`](sbatch_inference_amd.sh) | SLURM driver for `run_inference.py` on MI300X |

## Quick start — Docker (local workstation / interactive node)

```bash
# 4-day deterministic forecast at 1.4°
./docker_run.sh inference --date 2020-01-01 --steps 16

# 10-day stochastic forecast
./docker_run.sh inference \
    --checkpoint v1/stochastic_1_4_deg.pkl --steps 40 --seed 42

# Interactive shell inside the container
./docker_run.sh shell

# Stop and remove the container when done
./docker_run.sh stop
```

The script auto-detects the AMD Container Toolkit; if unavailable it falls
back to passing `/dev/kfd` and `/dev/dri` directly.

## Quick start — bare metal

```bash
# 1. Install JAX for ROCm (inside a ROCm 7.0.2 environment)
python3 -m pip install \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.0-cp310-cp310-manylinux2014_x86_64.whl
python3 -m pip install jax==0.6.0 jax-rocm7-pjrt jax-rocm7-plugin
apt update && apt install -y libdw1
python3 -m pip install neuralgcm gcsfs xarray matplotlib

# 2. Check environment
python preflight_neuralgcm.py

# 3. Run a 4-day forecast (16 × 6h steps)
python run_inference.py --date 2020-01-01 --steps 16
```

## SLURM submission

Edit the `#SBATCH` directives (`--partition`, `--account`) in
`sbatch_inference_amd.sh`, then submit:

```bash
NGC_DATE=2020-01-01 NGC_STEPS=16 sbatch sbatch_inference_amd.sh
```

To run inside an Apptainer/Singularity container built from
`rocm/dev-ubuntu-22.04:7.0.2-complete`:

```bash
NGC_SIF=/path/to/neuralgcm.sif NGC_STEPS=16 sbatch sbatch_inference_amd.sh
```

## Checkpoint selection

Pass `--checkpoint` to select a resolution or model type:

| Flag value | Type | Resolution |
|-----------|------|-----------|
| `v1/deterministic_0_7_deg.pkl` | Deterministic | 0.7° (~78 km) |
| `v1/deterministic_1_4_deg.pkl` | Deterministic (default) | 1.4° (~156 km) |
| `v1/deterministic_2_8_deg.pkl` | Deterministic | 2.8° (~312 km) |
| `v1/stochastic_1_4_deg.pkl` | Stochastic | 1.4° |
| `v1_precip/stochastic_precip_2_8_deg.pkl` | Precipitation | 2.8° |
| `v1_precip/stochastic_evap_2_8_deg.pkl` | Evaporation | 2.8° |

## Output

`run_inference.py` writes a NetCDF file under `outputs/` by default:

| Script | Default output |
|--------|---------------|
| `run_inference.py` | `outputs/neuralgcm-<checkpoint>-<date>.nc` |

Override with `--output /your/path.nc`.
