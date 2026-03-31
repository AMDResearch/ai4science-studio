# NeuralGCM: inference

Run deterministic or stochastic global weather / climate forecasts using ERA5 as initial conditions.

Source: [AMD ROCm blog — NeuralGCM inference](https://rocm.blogs.amd.com/artificial-intelligence/neuralgcm-inference/README.html)

> **Ready-to-run scripts** live in [`../../examples/`](../../examples/).
> Use [`run_inference.py`](../../examples/run_inference.py) directly instead of
> copying snippets from this doc.  The SLURM driver is
> [`sbatch_inference_mi300x.sh`](../../examples/sbatch_inference_mi300x.sh).

## Environment

### Option A — AMD Container Toolkit

```bash
docker run -d \
    --runtime=amd \
    -e AMD_VISIBLE_DEVICES=all \
    --name neuralgcm \
    -v $(pwd):/workspace \
    rocm/dev-ubuntu-22.04:7.0.2-complete \
    tail -f /dev/null

docker exec -it neuralgcm /bin/bash
```

### Option B — without AMD Container Toolkit

```bash
docker run -d \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --name neuralgcm \
    -v $(pwd):/workspace \
    rocm/dev-ubuntu-22.04:7.0.2-complete \
    tail -f /dev/null

docker exec -it neuralgcm /bin/bash
```

## Installation (inside container)

```bash
# 1. JAX for ROCm 7.0.2
python3 -m pip install \
    https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.0-cp310-cp310-manylinux2014_x86_64.whl
python3 -m pip install jax==0.6.0 jax-rocm7-pjrt jax-rocm7-plugin

# 2. libdw (JAX dependency on Ubuntu 22.04)
apt update && apt install -y libdw1

# 3. NeuralGCM and supporting packages
python3 -m pip install neuralgcm gcsfs xarray matplotlib
```

Verify the GPU is visible:

```python
import jax
print("Available devices:", jax.devices())  # expected: [RocmDevice(id=0)]
print("JAX version:", jax.__version__)       # expected: 0.6.0
```

## Running inference

### Python API

```python
import pickle
import gcsfs
import jax
import jax.numpy as jnp
import neuralgcm

# --- load checkpoint (anonymous GCS, no credentials needed) ---
fs = gcsfs.GCSFileSystem(token="anon")
with fs.open("gs://neuralgcm/models/v1/deterministic_1_4_deg.pkl", "rb") as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

# --- load ERA5 initial conditions (example via xarray) ---
# See examples/run_inference.py for the full ERA5 fetch
inputs, forcings, targets = ...  # loaded from ERA5 xarray dataset

# --- run forecast ---
predictions = model.unroll(inputs, forcings, steps=4 * 4)  # 4 days at 6h steps
```

### Script

```bash
cd /workspace
python run_inference.py --checkpoint v1/deterministic_1_4_deg.pkl --steps 16
```

Output is written to `outputs/neuralgcm-<checkpoint>-<date>.nc` by default.

## Available checkpoints

| Path on GCS | Type | Resolution |
|-------------|------|-----------|
| `gs://neuralgcm/models/v1/deterministic_0_7_deg.pkl` | Deterministic | 0.7° |
| `gs://neuralgcm/models/v1/deterministic_1_4_deg.pkl` | Deterministic | 1.4° |
| `gs://neuralgcm/models/v1/deterministic_2_8_deg.pkl` | Deterministic | 2.8° |
| `gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl` | Stochastic | 1.4° |
| `gs://neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl` | Precipitation | 2.8° |
| `gs://neuralgcm/models/v1_precip/stochastic_evap_2_8_deg.pkl` | Evaporation | 2.8° |

## Performance (MI300X)

| Resolution | 4-day forecast time |
|-----------|-------------------|
| 0.7° | ~110 seconds |
| 1.4° | ~48 seconds |
| 2.8° | ~33 seconds |

## Output variables

**Weather checkpoints:** geopotential, temperature, specific humidity, wind (u, v), cloud ice/liquid water — all on 37 pressure levels.

**Precipitation checkpoints add:** `precipitation_cumulative_mean`, `evaporation`.

Output array shape: `(level, longitude, latitude)` — e.g., `(37, 512, 256)` at 0.7°.

## Data sources

| Data | Location |
|------|---------|
| ERA5 initial conditions | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (public, anon) |
| Forcing fields | Derived from ERA5 (SST, sea ice held at initial state beyond day 0) |

## Known caveats

- **Persistence forcing:** sea-surface temperature and sea-ice concentration are held at their initial values for the entire forecast. Accuracy degrades for extended (multi-week) runs over ocean-influenced regions.
- **Initialization shock:** the model's learned encoder/decoder suppress spurious gravity waves, but artifacts can appear in the first few steps.
- **Stochastic members:** for ensemble spread, use the stochastic checkpoints and vary the JAX PRNG key per member.
- **Resolution trade-off:** the 0.7° checkpoint produces sharper output but runs ~3× slower than 2.8°.
