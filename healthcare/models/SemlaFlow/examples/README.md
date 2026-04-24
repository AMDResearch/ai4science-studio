# SemlaFlow examples

Ready-to-run scripts for [SemlaFlow](../README.md). These wrap the upstream
[`rssrwn/semla-flow`](https://github.com/rssrwn/semla-flow) with AMD/ROCm
container launch.

> **Research / engineering use only.** Not for clinical or diagnostic use.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_semlaflow.py`](preflight_semlaflow.py) | Check your environment before submitting jobs |
| [`run_inference.sh`](run_inference.sh) | Generate 3D molecular structures |
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones semla-flow, builds conda env) |

## Quick start — Docker

```bash
# Set up the container and run inference
./docker_run.sh inference

# Interactive shell
./docker_run.sh shell
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CHECKPOINT` | Yes | -- | Path to pretrained `.ckpt` file |
| `DATASET` | No | `drugs` | Target dataset (`qm9` or `drugs`) |
| `OUTPUT_FILE` | No | `/workspace/generated.sdf` | Output SDF file |
| `NUM_STEPS` | No | `20` | ODE integration steps |
| `USE_COMPILE` | No | `1` | Enable `torch.compile` |
| `NO_EMA` | No | `1` | Disable EMA for speed |
