# GenCast

**Hugging Face:** N/A — model assets downloaded automatically by the `ai-models` framework at runtime
**Upstream code:** [`google-deepmind/graphcast`](https://github.com/google-deepmind/graphcast)
**AMD recipe code:** [`silogen/ai-samples`](https://github.com/silogen/ai-samples) (path: `ai4sciences/ai-weather-forecasting`)
**Paper:** [GenCast: Diffusion-based ensemble weather forecasting for improved prediction accuracy and uncertainty quantification](https://www.nature.com/articles/s41586-024-08252-9) (Nature, 2025)
**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial)

## What it does

GenCast is a **diffusion-based probabilistic ensemble weather forecasting model** developed by Google DeepMind. Unlike deterministic models, GenCast generates multiple plausible forecast trajectories, quantifying forecast uncertainty via an ensemble.

| Property | Value |
|---|---|
| Resolution | 0.25° (~28 km) |
| Pressure levels | 13 (50–1000 hPa) |
| Forecast horizon | Up to 10 days (240 hours) |
| Ensemble | Multiple members (must be a multiple of GPU count) |
| Framework | JAX |

**Output variables:** t, z, u, v, w, q (upper-air), lsm, 2t, sst, msl, 10u, 10v, tp (surface)

## Available variants

| Variant | Description |
|---|---|
| `gencast-0.25` | Full 0.25° model |
| `gencast-0.25-Oper` | Operational variant |
| `gencast-1.0` | 1.0° lower-resolution model |
| `gencast-1.0-Mini` | Lightweight 1.0° variant |

## How it runs

GenCast is invoked through the [ECMWF `ai-models`](https://github.com/ecmwf-lab/ai-models) framework. It uses the same JAX Docker image as PanguWeather (`jaxweather:latest` from `silogen/ai-samples`).

**Data requirement:** A free [Copernicus CDS account](https://cds.climate.copernicus.eu/) is required for initial conditions.

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/inference/`](recipes/inference/) | Run probabilistic ensemble forecasts from CDS initial conditions |

## Quick start

```bash
# 1. Copy and configure credentials
cp examples/env_file.template examples/env_file
#    → set CDSAPI_KEY in env_file

# 2. Build Docker image and launch container
bash examples/docker_run.sh

# 3. Inside container — run a 10-day ensemble forecast
bash /examples/run_inference.sh
```

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** using the JAX ROCm Docker image from `silogen/ai-samples`. No code changes required.

Optional XLA memory parameters (`set_XLA_params.sh`) can be sourced before running to reduce peak memory usage on large ensemble runs:

```bash
source /recipe/set_XLA_params.sh
```

## References

- [AMD ROCm blog — SOTA Weather Forecasting on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html)
- [GenCast paper (Nature, 2025)](https://www.nature.com/articles/s41586-024-08252-9)
- [GraphCast / GenCast GitHub](https://github.com/google-deepmind/graphcast)
