# Walrus examples

Ready-to-run scripts for [Walrus](../README.md). These wrap the upstream
[`PolymathicAI/walrus`](https://github.com/PolymathicAI/walrus) with
AMD/ROCm container launch and automatic weight download.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_walrus.py`](preflight_walrus.py) | Check your environment before submitting jobs |
| [`run_inference.py`](run_inference.py) | Autoregressive rollout (downloads weights automatically) |
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones walrus, installs deps) |

## Quick start — Docker

```bash
# Run inference (auto-downloads weights on first run)
./docker_run.sh inference

# Interactive shell
./docker_run.sh shell
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WALRUS_WEIGHTS_DIR` | No | `./walrus-weights` | Local weights directory (auto-downloaded if missing) |
| `WALRUS_INPUT` | No | -- | Input field file (random noise demo if unset) |
| `WALRUS_STEPS` | No | `50` | Rollout steps |
| `WALRUS_OUTPUT` | No | `outputs/walrus_rollout.pt` | Output `.pt` file |
