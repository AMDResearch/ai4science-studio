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
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | SLURM/Apptainer multi-node training (HPC) |
| [`run_scaling_study.sh`](run_scaling_study.sh) | Submit matched 1/2/4/8-node strong-scaling sweep |
| [`collate_scaling_study.py`](collate_scaling_study.py) | Parse SLURM logs → steady-state throughput table |
| [`parse_convergence.py`](parse_convergence.py) | Extract loss/epoch metrics from training logs |
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

## Quick start — HPC scaling study

Matched strong-scaling runs (identical env at 1, 2, 4, 8 nodes; steady-state timing):

```bash
export AI4S_SHARED_DIR=/path/to/shared
export SBATCH_PARTITION=... SBATCH_ACCOUNT=...
./run_scaling_study.sh

# After jobs complete (logs in cwd as hydragnn-train-<jobid>.out):
python3 collate_scaling_study.py --log-dir . --jobs <id1>,<id2>,... -o scaling_study
```

See [`../recipes/train/README.md`](../recipes/train/README.md) for validated MI355X numbers and network notes.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HG_CHECKPOINT` | Yes (inference) | -- | Path to `.pk` checkpoint |
| `HG_CONFIG` | Yes | -- | Matching `config.json` |
| `HG_OUTPUT_DIR` | No | `/workspace/results` | Output directory |
| `HG_EXAMPLE` | No | `multidataset_hpo` | Upstream example folder (train) |
| `HG_DATA_DIR` | No | `/data` | Dataset staging directory (train) |
