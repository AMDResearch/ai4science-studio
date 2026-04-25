# HydraGNN examples

Ready-to-run scripts for [HydraGNN](../README.md). These wrap the upstream
[`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) branch `Predictive_GFM_2024`
with AMD/ROCm container launch.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_hydragnn.py`](preflight_hydragnn.py) | Check your environment before submitting jobs |
| [`run_inference.sh`](run_inference.sh) | Load a checkpoint and run predictions |
| [`run_train.sh`](run_train.sh) | Train on bundled or staged ADIOS datasets |
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones HydraGNN, installs deps) |

## Quick start — Docker

```bash
# Inference
./docker_run.sh inference

# Training
./docker_run.sh train

# Interactive shell
./docker_run.sh shell
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HG_CHECKPOINT` | Yes (inference) | -- | Path to `.pk` checkpoint |
| `HG_CONFIG` | Yes | -- | Matching `config.json` |
| `HG_OUTPUT_DIR` | No | `/workspace/results` | Output directory |
| `HG_EXAMPLE` | No | `multidataset_hpo` | Upstream example folder (train) |
| `HG_DATA_DIR` | No | `/data` | Dataset staging directory (train) |
