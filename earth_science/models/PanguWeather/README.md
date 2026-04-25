# PanguWeather

**Hugging Face:** N/A — model assets downloaded automatically by the `ai-models` framework at runtime
**Upstream code:** [`198808xc/Pangu-Weather`](https://github.com/198808xc/Pangu-Weather)
**AMD recipe code:** [`silogen/ai-samples`](https://github.com/silogen/ai-samples) (path: `ai4sciences/ai-weather-forecasting`)
**Paper:** [Accurate medium-range global weather forecasting with 3D neural networks](https://www.nature.com/articles/s41586-023-06185-3) (Nature, 2023)
**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial)

## What it does

PanguWeather is a **transformer-based deterministic global weather forecasting model** developed by Huawei. It predicts 6-hourly atmospheric states at 0.25° horizontal resolution (~28 km) across 13 pressure levels, up to 7 days ahead.

The model is structured as a hierarchy of four separate transformers trained at 1-hour, 3-hour, 6-hour, and 24-hour forecast intervals, combined at inference time for multi-day forecasts.

| Variable type | Variables |
|---|---|
| Upper-air (13 levels) | Geopotential (z), specific humidity (q), temperature (t), u/v wind |
| Surface | Mean sea level pressure (msl), 10m u/v wind, 2m temperature |

## How it runs

PanguWeather is invoked through the [ECMWF `ai-models`](https://github.com/ecmwf-lab/ai-models) framework, which handles:
- Fetching ERA5 initial conditions from the Copernicus Climate Data Store (CDS)
- Downloading model assets automatically
- Running inference and writing output as GRIB2

**Data requirement:** A free [Copernicus CDS account](https://cds.climate.copernicus.eu/) is required for initial conditions.

**AMD note:** One small patch is applied to the ONNX backend — the upstream code only registered CUDA and CPU execution providers; the patch adds a ROCm-compatible path.

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/inference/`](recipes/inference/) | Run 1–7 day deterministic forecasts from CDS initial conditions |

## Quick start

```bash
# 1. Copy and configure credentials
cp examples/env_file.template examples/env_file
#    → set CDSAPI_KEY in env_file

# 2. Build Docker images and launch container
bash examples/docker_run.sh

# 3. Inside container — run a 24-hour forecast
bash /examples/run_inference.sh
```

See [`recipes/inference/`](recipes/inference/) for the full walkthrough.

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** with the JAX ROCm Docker image built from `silogen/ai-samples`.

- One ONNX backend patch required — included in the silogen/ai-samples repo
- No other code changes needed

## References

- [AMD ROCm blog — SOTA Weather Forecasting on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html)
- [PanguWeather paper (Nature, 2023)](https://www.nature.com/articles/s41586-023-06185-3)
- [GitHub upstream](https://github.com/198808xc/Pangu-Weather)
