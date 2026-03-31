# StormCast

**Hugging Face:** [`nvidia/stormcast-v1-era5-hrrr`](https://huggingface.co/nvidia/stormcast-v1-era5-hrrr)
**Upstream framework:** [`NVIDIA/earth2studio`](https://github.com/NVIDIA/earth2studio) — `earth2studio.models.px.StormCast`
**Paper:** [Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling](https://arxiv.org/abs/2408.10958) (arXiv:2408.10958)
**License:** Apache-2.0

## What it does

StormCast is a **high-resolution convection-allowing weather prediction model** developed by NVIDIA. It emulates thunderstorm-scale atmospheric evolution at **3 km horizontal resolution** over the central continental United States (1536 km × 1920 km), stepping forward in **1-hour increments**.

The model has two coupled components:

| Component | Role |
|-----------|------|
| Regression U-Net | Predicts the deterministic mean state |
| Diffusion model (DDPM++ U-Net) | Samples stochastic residuals conditioned on the mean state |

Initial conditions come from **HRRR analysis** (not forecast) data; coarse synoptic context comes from **GFS**. Both are fetched automatically via Earth2Studio's data connectors.

## Output variables

| Variable | Description | Levels |
|----------|-------------|--------|
| `u`, `v` | Zonal / meridional wind | 16 |
| `z` | Geopotential height | 16 |
| `t` | Temperature | 16 |
| `q` | Specific humidity | 16 |
| `p` | Pressure | 14 |
| `refc` | Composite radar reflectivity | surface |
| `mslp` | Mean sea level pressure | surface |

Grid uses a **Lambert Conformal Conic** projection; vertical coordinate follows HRRR v4 hybrid-sigma levels.

## Recipes

| Recipe | Summary |
|--------|---------|
| [`recipes/inference/`](recipes/inference/) | Deterministic single-member forecast |
| [`recipes/ensemble/`](recipes/ensemble/) | Multi-member ensemble with uncertainty quantification |

Both recipes share the same model weights — the ensemble variant is a different *run mode* on top of the same checkpoint.

## Installation

```bash
pip install "earth2studio[stormcast]" cartopy
```

Weights are fetched automatically from Hugging Face on first call to `StormCast.load_default_package()`. No manual download is needed.

## AMD / ROCm notes

Validated on **AMD MI300X** with ROCm 7.0.2 + PyTorch 2.8.0 using the official container:

```
rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

- VRAM: ~9.6 GB for a single-member deterministic run
- Runtime: ~2 minutes for 6 one-hour steps on one MI300X

See individual recipe docs for full container launch instructions.

## On-disk folder name

This directory is named **`StormCast`** for clarity. The canonical Hub id is **`nvidia/stormcast-v1-era5-hrrr`**, accessed via Earth2Studio's `StormCast.load_default_package()`.
