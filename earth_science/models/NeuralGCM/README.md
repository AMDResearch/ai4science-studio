# NeuralGCM

**Hugging Face:** N/A — model weights are distributed via Google Cloud Storage (`gs://neuralgcm/models/`). See [Obtaining model weights](#obtaining-model-weights) below.
**Upstream code:** [`google-research/neuralgcm`](https://github.com/google-research/neuralgcm)
**Papers:**
- [Neural General Circulation Models for Weather and Climate](https://arxiv.org/abs/2311.07222) (arXiv:2311.07222)
- [Neural GCMs optimized to predict satellite-based precipitation](https://arxiv.org/abs/2412.11973) (arXiv:2412.11973)

**License:** Code: Apache-2.0 · Model weights: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## What it does

NeuralGCM is a **hybrid physics + machine-learning atmospheric model** developed by Google Research. It couples a differentiable dynamical core (the "GCM" half) with learned encoder/decoder networks that correct discretization errors and parameterize sub-grid physics. This design avoids the blurring common to pure-ML models on long-range forecasts while remaining orders of magnitude faster than full numerical GCMs.

The model is pretrained on **ERA5 reanalysis** data and supports two families of checkpoints:

| Family | Task |
|--------|------|
| Weather / climate forecasting | Deterministic and stochastic atmosphere |
| Precipitation & evaporation | Stochastic surface flux |

## Available checkpoints

| Checkpoint filename | Type | Resolution |
|---------------------|------|-----------|
| `v1/deterministic_0_7_deg.pkl` | Deterministic | 0.7° (~78 km) |
| `v1/deterministic_1_4_deg.pkl` | Deterministic | 1.4° (~156 km) |
| `v1/deterministic_2_8_deg.pkl` | Deterministic | 2.8° (~312 km) |
| `v1/stochastic_1_4_deg.pkl` | Stochastic | 1.4° (~156 km) |
| `v1_precip/stochastic_precip_2_8_deg.pkl` | Precipitation | 2.8° |
| `v1_precip/stochastic_evap_2_8_deg.pkl` | Evaporation | 2.8° |

## Grid and coordinates

- **Horizontal:** global, lat–lon; resolutions of 0.7°, 1.4°, or 2.8°
- **Vertical:** 37 pressure levels (ERA5 levels) converted internally to sigma/terrain-following coordinates
- **Temporal:** arbitrary step length; tested at 6-hour steps; typical forecast horizon 15 days – several decades

Output arrays have shape `(level, longitude, latitude)`. Example at 0.7°: `(37, 512, 256)`.

## Output variables

**Weather / climate checkpoints:**

| Variable | Description |
|----------|-------------|
| Geopotential | 3-D geopotential height |
| Temperature | 3-D air temperature |
| Specific humidity | 3-D moisture |
| Wind (u, v) | 3-D horizontal wind components |
| Cloud ice / liquid water | 3-D condensates |

**Precipitation checkpoints add:**

| Variable | Description |
|----------|-------------|
| `precipitation_cumulative_mean` | Accumulated precipitation |
| `evaporation` | Surface evaporative flux |

## Data sources

- **Initial conditions:** ERA5 reanalysis from `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (public GCS, no auth required)
- **Precipitation training target:** Global Precipitation Measurement (IMERG) satellite dataset

## Obtaining model weights

Weights are stored on Google Cloud Storage and can be fetched directly via `gcsfs` (no authentication required):

```python
import gcsfs
import pickle

fs = gcsfs.GCSFileSystem(token="anon")
with fs.open("gs://neuralgcm/models/v1/deterministic_1_4_deg.pkl", "rb") as f:
    ckpt = pickle.load(f)
```

See [`examples/run_inference.py`](examples/run_inference.py) for the complete fetch-and-run pattern.

## Recipes

| Recipe | Summary |
|--------|---------|
| [`recipes/inference/`](recipes/inference/) | Run deterministic or stochastic forecasts from ERA5 initial conditions |

## Installation

```bash
pip install neuralgcm gcsfs xarray matplotlib
```

JAX must be installed separately. For AMD/ROCm see the [AMD / ROCm notes](#amd--rocm-notes) section below.

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** with ROCm 7.0.2 and a custom JAX ROCm build:

```
rocm/dev-ubuntu-22.04:7.0.2-complete
```

Install JAX for ROCm inside the container:

```bash
python3 -m pip install \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.0-cp310-cp310-manylinux2014_x86_64.whl
python3 -m pip install jax==0.6.0 jax-rocm7-pjrt jax-rocm7-plugin
apt update && apt install -y libdw1
python3 -m pip install neuralgcm gcsfs matplotlib
```

### Performance (4-day forecast, MI300X)

| Resolution | Inference time |
|-----------|---------------|
| 0.7° | ~110 seconds |
| 1.4° | ~48 seconds |
| 2.8° | ~33 seconds |

See [`recipes/inference/`](recipes/inference/) for full container launch instructions.

## On-disk folder name

This directory is named **`NeuralGCM`** for clarity. Model weights are distributed via GCS (`gs://neuralgcm/models/`), not via Hugging Face. The upstream Python package is `neuralgcm` on PyPI.
