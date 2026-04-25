# MATEY — Inference / Rollout

> **Ready-to-run scripts:** see [`../../examples/`](../../examples/) for `docker_run.sh` and `run_inference.py`.

## Overview

After training a MATEY model, you can perform autoregressive rollout to generate predictions over arbitrary time horizons from a given initial condition.

## Prerequisites

- A trained MATEY checkpoint (see [../train/](../train/README.md))
- ROCm PyTorch environment or Docker container

## Quick start

```bash
cd physics_simulation/models/MATEY/examples
./docker_run.sh inference
```

## Manual inference

```bash
python run_inference.py \
    --checkpoint /path/to/checkpoint.pt \
    --config     /path/to/config.yaml \
    --input      /path/to/initial_condition.h5 \
    --steps      100 \
    --output     outputs/rollout.h5
```

All parameters can be set via environment variables (see script header for defaults).

## Output format

Rollout predictions are written as HDF5 files containing the predicted physical field at each timestep. Fields match the layout of the training data.

## AMD / ROCm notes

- Inference runs on a single MI300X without modification
- `torch.compile(mode="max-autotune")` may improve throughput for repeated rollouts — not yet validated
- For large rollouts (many timesteps), HBM capacity of MI300X (192 GB) allows keeping long histories in memory
