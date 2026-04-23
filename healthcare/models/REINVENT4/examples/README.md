# REINVENT4 examples

Ready-to-run scripts for [REINVENT4](../README.md). These wrap the upstream
[`MolecularAI/REINVENT4`](https://github.com/MolecularAI/REINVENT4) with
AMD/ROCm container launch.

> **Research / engineering use only.** Not for clinical or diagnostic use.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_reinvent4.py`](preflight_reinvent4.py) | Check your environment before submitting jobs |
| [`run_tl.sh`](run_tl.sh) | Run transfer learning with a TOML config |
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones REINVENT4, installs deps) |
| [`tl_config.toml.template`](tl_config.toml.template) | Template TOML for transfer learning |

## Quick start — Docker

```bash
# Set up the container
./docker_run.sh

# Run transfer learning
docker exec reinvent4 bash /workspace/run_tl.sh
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONFIG_FILE` | Yes | `/workspace/tl_config.toml` | TOML configuration file |
| `RESULTS_DIR` | No | `/workspace/results` | Logs and results directory |
