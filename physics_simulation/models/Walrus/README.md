# Walrus

| Field | Value |
|-------|-------|
| **Hugging Face model id** | [`polymathic-ai/walrus`](https://huggingface.co/polymathic-ai/walrus) |
| **Task** | Cross-domain surrogate simulation for continuum dynamical systems (next-step prediction, autoregressive rollout) |
| **License** | [MIT](https://github.com/PolymathicAI/walrus/blob/main/LICENSE) |
| **Upstream code** | <https://github.com/PolymathicAI/walrus> |
| **Paper** | [Walrus: A Cross-Domain Foundation Model for Continuum Dynamics (arXiv:2511.15684)](https://arxiv.org/abs/2511.15684) |
| **Maintained by** | Polymathic AI |

## Overview

Walrus is a **1.3B-parameter space-time Transformer** trained autoregressively across 19 diverse physical domains to predict the temporal evolution of physical fields. It is designed as a cross-domain foundation model: a single set of weights generalizes across fluid dynamics, plasma physics, astrophysics, geoscience, acoustics, and more.

Key architectural features:
- **Adaptive-compute patch embedding** — balances token count across resolutions; supports mixed 2D/3D data
- **Patch jittering** — harmonic-analysis–motivated augmentation that reduces spectral aliasing
- **Tensor-law–aware augmentation** — 2D fields embedded in 3D via correct physical transformations
- **Asymmetric normalization** — normalizes inputs by RMS over space-time; de-normalizes predicted Δu by its own RMS

The model predicts: `u(t+1) ≈ u(t) + M([u(t−τ+1), …, u(t)])`

## Domains covered

| Domain | Examples |
|--------|---------|
| Classical fluids | Compressible/incompressible Navier-Stokes, turbulence |
| Plasma physics | MHD, plasma dynamics |
| Astrophysics | Neutron star mergers, stellar convection |
| Geoscience | Planetary-scale weather systems |
| Rheology | Fluid flow and material deformation |
| Acoustics | Wave propagation |

## Recipes

| Task | Link |
|------|------|
| Inference / rollout | [recipes/inference/](recipes/inference/README.md) |

## AMD / ROCm notes

Walrus is a standard PyTorch Transformer model and runs without modification on ROCm. See [`examples/`](examples/) for Docker scripts targeting MI300X.
