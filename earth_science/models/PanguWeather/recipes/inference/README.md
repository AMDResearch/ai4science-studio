# PanguWeather — Inference Recipe

Run deterministic global weather forecasts up to 7 days ahead using the `ai-models` framework.

> **Ready-to-run scripts** live in [`../../examples/`](../../examples/).
> Use [`run_inference.sh`](../../examples/run_inference.sh) directly instead of
> copying snippets from this doc.

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Container** | `jaxweather:latest` from silogen/ai-samples (shared with GenCast) |
| **GPU** | AMD Instinct with ROCm kernel-mode driver (`amdgpu-dkms`) |
| **Runtime** | Docker |
| **Weights** | Auto-downloaded by `ai-models` framework (~1.1 GB) |
| **Data** | Free [Copernicus CDS account](https://cds.climate.copernicus.eu/) for ERA5 initial conditions |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CDSAPI_KEY` | **Yes** | — | CDS API key (set in `env_file`) |
| `DATE` | No | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | No | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | No | `24` | Forecast horizon in hours (max ~168 = 7 days) |

## Quick Start

```bash
cp examples/env_file.template examples/env_file
# Edit env_file: set CDSAPI_KEY
bash examples/docker_run.sh
# Inside container:
bash /examples/run_inference.sh
```

## Step 1 — Configure credentials

```bash
cd earth_science/models/PanguWeather/examples/
cp env_file.template env_file
# Edit env_file: replace YOUR_CDS_API_KEY_HERE with your real key
```

Get your API key at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) → Profile → API key.

## Step 2 — Launch the container

```bash
bash examples/docker_run.sh
```

This script will:
1. Clone `silogen/ai-samples` (if not already present) for the Dockerfiles
2. Build the `jaxweather:latest` image (first run only; ~10 min)
3. Launch an interactive container with your credentials, cache, and example scripts mounted

> The same `jaxweather:latest` image works for both PanguWeather and GenCast.

## Step 3 — Run a forecast (inside the container)

```bash
bash /examples/run_inference.sh
```

Default: 24-hour forecast from yesterday's 00Z ERA5 analysis. Customize with environment variables:

```bash
DATE=20240101 TIME=0000 LEAD_TIME=168 bash /examples/run_inference.sh
```

| Variable | Default | Description |
|---|---|---|
| `DATE` | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | `24` | Forecast horizon in hours (max ~168) |

## Output

- **Format:** GRIB2 at 0.25° resolution
- **File:** `/predictions/panguweather.grib`
- **Variables:** z, q, t, u, v (13 pressure levels) + msl, 10u, 10v, 2t (surface)

Read in Python:

```python
import cfgrib, xarray as xr
datasets = cfgrib.open_datasets("panguweather.grib")
```

## Visualize

```bash
python3 /recipe/grib_visualizer.py --input /predictions/panguweather.grib
```

Produces per-level GIFs under `/predictions/outputs/level_*/`.

## Model assets

PanguWeather weights (~1.1 GB) are downloaded automatically on first run into `examples/predictions/assets/panguweather/`. Subsequent runs reuse the cached weights.

## AMD / ROCm notes

The `ai-models-panguweather` plugin applies a one-line ONNX backend patch to add the ROCm execution provider. This is handled automatically by the Docker image — no user action required.

## References

- [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html)
- [ai-models framework](https://github.com/ecmwf-lab/ai-models)
- [PanguWeather paper (Nature 2023)](https://www.nature.com/articles/s41586-023-06185-3)
