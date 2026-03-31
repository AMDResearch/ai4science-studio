# StormCast recipes

Studio runbooks for [StormCast](../README.md) (`nvidia/stormcast-v1-era5-hrrr` on Hugging Face, accessed via [`NVIDIA/earth2studio`](https://github.com/NVIDIA/earth2studio)).

Both recipes use the **same model checkpoint** — the ensemble variant is a different execution mode, not a separate model.

## Recipes

| Doc | Summary |
|-----|---------|
| [inference/](inference/) | Deterministic single-member 3 km forecast; HRRR initial conditions fetched automatically |
| [ensemble/](ensemble/) | Multi-member ensemble with uncertainty quantification via the diffusion residual sampler |

## Prerequisites (both recipes)

- AMD Instinct GPU (MI300X validated) or NVIDIA CUDA GPU
- Docker + ROCm kernel driver (for AMD path)
- Internet access to fetch HRRR data and model weights on first run
