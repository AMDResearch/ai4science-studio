# GenCast examples

Ready-to-run scripts for [GenCast](../README.md). GenCast uses the JAX
Docker image (`jaxweather:latest`) built from
[silogen/ai-samples](https://github.com/silogen/ai-samples)
(`ai4sciences/ai-weather-forecasting`). The same image works for PanguWeather.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_gencast.py`](preflight_gencast.py) | Check your environment before running |
| [`docker_run.sh`](docker_run.sh) | Build the JAX Docker image and launch an interactive container |
| [`run_inference.sh`](run_inference.sh) | Run an ensemble forecast (inside container) |
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

# 3. Inside container — run a 10-day ensemble forecast
bash /examples/run_inference.sh

# Customize (4-member ensemble, 1.0° variant)
MODEL_VARIANT=gencast-1.0 NUM_ENSEMBLE_MEMBERS=4 bash /examples/run_inference.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CDSAPI_KEY` | *(required)* | Copernicus CDS API key (set in `env_file`) |
| `MODEL_VARIANT` | `gencast-0.25` | Checkpoint: `gencast-0.25`, `gencast-0.25-Oper`, `gencast-1.0`, `gencast-1.0-Mini` |
| `DATE` | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | `240` | Forecast horizon in hours (10 days) |
| `NUM_ENSEMBLE_MEMBERS` | `1` | Ensemble size (must be multiple of GPU count) |
| `CACHE_DIR` | `/tmp/earthkit-cache` | Host path for ERA5 data cache |
| `AI_SAMPLES_DIR` | `<script_dir>/ai-samples` | Where to clone silogen/ai-samples |

## Notes

- GenCast and PanguWeather share the same `jaxweather:latest` Docker image
- XLA memory parameters (`set_XLA_params.sh`) are sourced automatically if present on the recipe mount
- `NUM_ENSEMBLE_MEMBERS` must be a multiple of the number of GPUs visible to the container
