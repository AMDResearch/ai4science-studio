# Aurora examples

Ready-to-run scripts for [Aurora](../README.md). Aurora uses the PyTorch
Docker image (`pytorchweather:latest`) built from
[silogen/ai-samples](https://github.com/silogen/ai-samples)
(`ai4sciences/ai-weather-forecasting`).

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_aurora.py`](preflight_aurora.py) | Check your environment before running |
| [`docker_run.sh`](docker_run.sh) | Build the PyTorch Docker image and launch an interactive container |
| [`run_inference.sh`](run_inference.sh) | Run a forecast (inside container) |
| [`env_file.template`](env_file.template) | Template for CDS API credentials |

## Prerequisites

1. Copy `env_file.template` to `env_file` and set your `CDSAPI_KEY`
   (free account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/))
2. ROCm kernel-mode driver (`amdgpu-dkms`)

## Quick start

```bash
# 1. Configure credentials
cp env_file.template env_file
# Edit env_file: set CDSAPI_KEY

# 2. Build the image and launch container
./docker_run.sh

# 3. Inside container — run a 24-hour forecast
bash /examples/run_inference.sh

# Customize
DATE=20240101 LEAD_TIME=120 bash /examples/run_inference.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CDSAPI_KEY` | *(required)* | Copernicus CDS API key (set in `env_file`) |
| `DATE` | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | `24` | Forecast horizon in hours (6h rollout steps) |
| `CACHE_DIR` | `/tmp/earthkit-cache` | Host path for ERA5 data cache |
| `AI_SAMPLES_DIR` | `<script_dir>/ai-samples` | Where to clone silogen/ai-samples |

## Notes

- Aurora uses `pytorchweather:latest` — a **separate image** from the JAX image used by PanguWeather and GenCast
- Aurora operates at 0.1° resolution (~11 km) — the highest of the AMD-validated weather models
- The MI300X's 192 GB HBM3 comfortably handles the memory footprint
