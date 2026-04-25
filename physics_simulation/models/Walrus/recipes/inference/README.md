# Walrus — Inference / Rollout

> **Ready-to-run scripts:** see [`../../examples/`](../../examples/) for `docker_run.sh` and `run_inference.py`.

## Overview

Walrus performs autoregressive rollout: given a short history of physical field snapshots as initial condition, it predicts subsequent timesteps across arbitrary rollout lengths.

## Prerequisites

- Docker with AMD Container Toolkit, or bare-metal ROCm PyTorch environment
- Walrus weights downloaded from Hugging Face: [`polymathic-ai/walrus`](https://huggingface.co/polymathic-ai/walrus)

## Downloading weights

```bash
# Option A: huggingface-cli
pip install huggingface-hub
huggingface-cli download polymathic-ai/walrus --local-dir ./walrus-weights

# Option B: Python
from huggingface_hub import snapshot_download
snapshot_download("polymathic-ai/walrus", local_dir="./walrus-weights")
```

## Quick start

```bash
cd physics_simulation/models/Walrus/examples
./docker_run.sh inference
```

## Manual inference

```python
from huggingface_hub import hf_hub_download
import torch

# Load model
model = torch.hub.load("PolymathicAI/walrus", "walrus")
model.eval().cuda()

# Prepare input: (batch, time_history, channels, *spatial)
# See the upstream repo for data loading utilities
u = ...  # your physical field snapshots

with torch.no_grad():
    u_next = model(u)
```

See the [upstream README](https://github.com/PolymathicAI/walrus) for full data format documentation and domain-specific examples.

## Input format

| Dimension | Description |
|-----------|-------------|
| Batch | Number of independent simulations |
| Time | Short history window (τ timesteps) |
| Channels | Physical field variables |
| Spatial | 2D or 3D spatial grid |

## AMD / ROCm notes

- Walrus uses standard PyTorch ops (attention, conv, layer norm) — runs on ROCm without modification
- `torch.compile(mode="max-autotune")` is a good optimization candidate for long rollouts — not yet benchmarked on MI300X
- MI300X 192 GB HBM is well-suited for large spatial grids or long history windows
