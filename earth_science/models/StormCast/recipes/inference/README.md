# StormCast: deterministic inference

Run a single-member 3 km convection-allowing forecast over the central US using HRRR analysis as initial conditions.

Source: [AMD ROCm blog — StormCast inference](https://rocm.blogs.amd.com/artificial-intelligence/stormcast-inference/README.html)

## Environment

### Option A — with AMD Container Toolkit

```bash
docker run -d \
    --runtime=amd \
    -e AMD_VISIBLE_DEVICES=all \
    --name stormcast \
    -v $(pwd):/workspace/ \
    rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0 \
    tail -f /dev/null

docker exec -it stormcast /bin/bash
```

### Option B — without AMD Container Toolkit

```bash
docker run -d \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --name stormcast \
    -v $(pwd):/workspace/ \
    rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0 \
    tail -f /dev/null

docker exec -it stormcast /bin/bash
```

## Installation (inside container)

```bash
cd /workspace
pip install "earth2studio[stormcast]" cartopy
```

Model weights are fetched automatically from [`nvidia/stormcast-v1-era5-hrrr`](https://huggingface.co/nvidia/stormcast-v1-era5-hrrr) on first run.

## Running inference

### Python API

```python
import earth2studio.run as run
from earth2studio.models.px import StormCast
from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from datetime import datetime

package = StormCast.load_default_package()
model = StormCast.load_model(package)
data = HRRR()
io = ZarrBackend("model-output.zarr")

starting_datetime = datetime(year=2025, month=1, day=1, hour=6)
nsteps = 6
io = run.deterministic([starting_datetime], nsteps, model, data, io)
```

### Script (from the AMD blog)

Download `run-stormcast.py` from the [blog post](https://rocm.blogs.amd.com/artificial-intelligence/stormcast-inference/README.html), then:

```bash
python run-stormcast.py 2025-01-01T06:00 6
```

Output lands in `outputs/pred-2025-01-01.zarr`.

> **Note:** The Zarr backend does not support overwriting outputs. Delete or rename the output directory before re-running.

## Performance (MI300X)

| Metric | Value |
|--------|-------|
| VRAM | ~9.6 GB (~5% of MI300X capacity) |
| Runtime | ~2 min for 6 × 1-hour steps |

## Output variables

8 predicted quantities: `u`, `v`, `z`, `t`, `q`, `p` (multi-level), `refc`, `mslp` (surface). See the [model README](../../README.md) for level counts.

## Visualization

### Python

```python
import cartopy
import cartopy.feature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from earth2studio.data import HRRR

hrrr_lat_lim = (273, 785)
hrrr_lon_lim = (579, 1219)
hrrr_lat, hrrr_lon = HRRR.grid()
model_lat = hrrr_lat[hrrr_lat_lim[0]:hrrr_lat_lim[1], hrrr_lon_lim[0]:hrrr_lon_lim[1]]
model_lon = hrrr_lon[hrrr_lat_lim[0]:hrrr_lat_lim[1], hrrr_lon_lim[0]:hrrr_lon_lim[1]]

projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

ds = xr.open_zarr("model-output.zarr", consolidated=False)

fig, axis = plt.subplots(
    nrows=1, ncols=1,
    subplot_kw={"projection": projection},
    figsize=(5, 6), layout="compressed",
)

variable = "refc"
nstep = 2
axis.pcolormesh(
    model_lon, model_lat, ds[variable][0, nstep],
    transform=ccrs.PlateCarree(), cmap="Spectral_r",
)
axis.coastlines()
axis.gridlines()
axis.add_feature(
    cartopy.feature.STATES.with_scale("50m"),
    linewidth=0.5, edgecolor="black", zorder=2,
)

plt.savefig("stormcast-plot.jpg")
```

> Use `consolidated=False` when opening Zarr datasets to suppress metadata warnings.

### Script (from the AMD blog)

```bash
python plot-stormcast.py outputs/pred-2025-01-01.zarr
```

Produces per-frame images named `scast-2025-01-01-refc-frame-00.jpg`, etc.

### Animation (requires ImageMagick)

```bash
convert -loop 0 -delay 150 scast-2025-01-01-*.jpg animation.gif
```

## Known caveats

- 1-hour temporal resolution is coarser than convective timescales — fast-moving storms may be underresolved.
- Model is trained on HRRR analysis data; accuracy degrades when HRRR analysis quality is poor.
- Coordinate system is Lambert Conformal Conic internally; use `ccrs.PlateCarree()` as the data transform when plotting.
