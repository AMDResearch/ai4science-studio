# Aurora

**Hugging Face:** [`microsoft/aurora`](https://huggingface.co/microsoft/aurora)
**Upstream code:** [`microsoft/aurora`](https://github.com/microsoft/aurora)
**AMD recipe code:** [`silogen/ai-samples`](https://github.com/silogen/ai-samples) (path: `ai4sciences/ai-weather-forecasting`)
**Paper:** [A foundation model of the Earth system](https://www.nature.com/articles/s41586-025-08897-0) (Nature, 2025)
**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial)

## What it does

Aurora is a **foundation model for the Earth system** developed by Microsoft Research. Trained on over a million hours of diverse weather, climate, and ocean data, it supports fine-tuning to new tasks and variables beyond those in the training set.

| Property | Value |
|---|---|
| Resolution | 0.1° (~11 km) — highest of the three AMD-validated weather models |
| Pressure levels | 13 (50–1000 hPa) |
| Forecast horizon | Up to 5 days (via 6h rollout) |
| Framework | PyTorch |

**Output variables:** z, u, v, t, q (upper-air) + 2t, 10u, 10v, msl (surface)

## How it runs

Aurora is invoked through the [ECMWF `ai-models`](https://github.com/ecmwf-lab/ai-models) framework using a **PyTorch** Docker image (`pytorchweather:latest` from `silogen/ai-samples`). This is a separate image from the JAX image used by PanguWeather and GenCast.

**Data requirement:** A free [Copernicus CDS account](https://cds.climate.copernicus.eu/) is required for initial conditions.

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/inference/`](recipes/inference/) | Run 0.1° deterministic forecasts from CDS initial conditions |

## Quick start

```bash
# 1. Copy and configure credentials
cp examples/env_file.template examples/env_file
#    → set CDSAPI_KEY in env_file

# 2. Build the PyTorch Docker image and launch container
bash examples/docker_run.sh

# 3. Inside container — run a 24-hour forecast
bash /examples/run_inference.sh
```

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** using the PyTorch ROCm Docker image from `silogen/ai-samples`. No code changes required — runs out-of-the-box.

Aurora's 0.1° resolution demands significantly more memory than PanguWeather or GenCast; the MI300X's 192 GB HBM3 is well-suited.

## References

- [AMD ROCm blog — SOTA Weather Forecasting on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html)
- [Aurora paper (Nature, 2025)](https://www.nature.com/articles/s41586-025-08897-0)
- [Hugging Face model card](https://huggingface.co/microsoft/aurora)
- [GitHub repo](https://github.com/microsoft/aurora)
