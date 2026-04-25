# PanguWeather examples

Ready-to-run scripts for [PanguWeather](../README.md). PanguWeather uses the JAX
Docker image (`jaxweather:latest`) built from
[silogen/ai-samples](https://github.com/silogen/ai-samples)
(`ai4sciences/ai-weather-forecasting`). The same image works for GenCast.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_panguweather.py`](preflight_panguweather.py) | Check your environment before running |
| [`docker_run.sh`](docker_run.sh) | Build the JAX Docker image and launch an interactive container |
| [`run_inference.sh`](run_inference.sh) | Run a deterministic forecast (inside container) |
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

# Customize (7-day forecast from a specific date)
DATE=20240101 LEAD_TIME=168 bash /examples/run_inference.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CDSAPI_KEY` | *(required)* | Copernicus CDS API key (set in `env_file`) |
| `DATE` | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | `24` | Forecast horizon in hours (max ~168 = 7 days) |
| `CACHE_DIR` | `/tmp/earthkit-cache` | Host path for ERA5 data cache |
| `AI_SAMPLES_DIR` | `<script_dir>/ai-samples` | Where to clone silogen/ai-samples |

## Notes

- PanguWeather and GenCast share the same `jaxweather:latest` Docker image
- Model weights (~1.1 GB) are downloaded automatically on first run via the `ai-models` framework
- One ONNX backend patch adds the ROCm execution provider — handled by the Docker image automatically
