# Aurora — Inference Recipe

Run deterministic global weather forecasts at 0.1° resolution (the highest resolution of the AMD-validated weather models).

> **Ready-to-run scripts** live in [`../../examples/`](../../examples/).
> Use [`run_inference.sh`](../../examples/run_inference.sh) directly instead of
> copying snippets from this doc.

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Container** | `pytorchweather:latest` from silogen/ai-samples (separate from JAX image) |
| **GPU** | AMD Instinct MI300X recommended (192 GB HBM3 for 0.1° resolution) |
| **Runtime** | Docker with ROCm kernel-mode driver (`amdgpu-dkms`) |
| **Data** | Free [Copernicus CDS account](https://cds.climate.copernicus.eu/) for ERA5 initial conditions |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CDSAPI_KEY` | **Yes** | — | CDS API key (set in `env_file`) |
| `DATE` | No | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | No | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | No | `24` | Forecast horizon in hours (6h rollout steps) |

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
cd earth_science/models/Aurora/examples/
cp env_file.template env_file
# Edit env_file: replace YOUR_CDS_API_KEY_HERE with your real key
```

## Step 2 — Launch the container

```bash
bash examples/docker_run.sh
```

This builds `pytorchweather:latest` (a separate image from the JAX image used by PanguWeather/GenCast) and launches an interactive container.

## Step 3 — Run a forecast (inside the container)

```bash
bash /examples/run_inference.sh
```

Default: 24-hour forecast from yesterday's 00Z analysis. Customize with environment variables:

```bash
DATE=20240101 TIME=0000 LEAD_TIME=120 bash /examples/run_inference.sh
```

| Variable | Default | Description |
|---|---|---|
| `DATE` | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | `24` | Forecast horizon in hours (rollout in 6h steps) |

## Output

- **Format:** GRIB2 at 0.1° resolution
- **File:** `/predictions/aurora.grib`
- **Variables:** z, u, v, t, q (13 levels) + 2t, 10u, 10v, msl (surface)

Read in Python:

```python
import cfgrib
datasets = cfgrib.open_datasets("aurora.grib")
```

## Visualize

```bash
python3 /recipe/grib_visualizer.py --input /predictions/aurora.grib
```

## Memory note

Aurora operates at **0.1° resolution** — approximately 6× more grid points than PanguWeather/GenCast at 0.25°. The MI300X's 192 GB HBM3 comfortably handles this footprint. On smaller GPUs, reduce lead-time or check `HIP_VISIBLE_DEVICES` in `env_file` to restrict GPU count.

## AMD / ROCm notes

- No code changes required — runs out-of-the-box on ROCm
- Uses `pytorchweather:latest` (PyTorch-based), not the JAX image used by other weather models

## References

- [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html)
- [Aurora paper (Nature, 2025)](https://www.nature.com/articles/s41586-025-08897-0)
- [Hugging Face model card](https://huggingface.co/microsoft/aurora)
- [ai-models framework](https://github.com/ecmwf-lab/ai-models)
