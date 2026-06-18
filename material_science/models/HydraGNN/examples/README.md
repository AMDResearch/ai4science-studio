# HydraGNN examples

Ready-to-run scripts for [HydraGNN](../README.md). These wrap the upstream
[`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) branch `Predictive_GFM_2024`
with AMD/ROCm container launch.

## Files

The folder is flat (scripts resolve siblings and the container bind-mount
relative to their own directory), but the files fall into the groups below.

### Setup & diagnostics

| Script | Purpose |
|--------|---------|
| [`preflight_hydragnn.py`](preflight_hydragnn.py) | Check your environment before submitting jobs |
| [`microbench_node_health.sh`](microbench_node_health.sh) | ~30 s per-node MI355X (gfx950) health survey (host/GPU inventory, dual-NUMA STREAM, HIP launch latency); safe to run in a SLURM Prolog |

### Run entrypoints

| Script | Purpose |
|--------|---------|
| [`run_train.sh`](run_train.sh) | Train on bundled or staged ADIOS datasets |
| [`run_inference.sh`](run_inference.sh) | Load a checkpoint and run predictions |

### Container & launch drivers

| Script | Purpose |
|--------|---------|
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones HydraGNN, installs deps) |
| [`build_overlay_amd.sh`](build_overlay_amd.sh) | One-time Apptainer ext3 overlay build (pre-loads pip deps; reuse across jobs to skip the per-job pip install) |
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | SLURM/Apptainer multi-node training (HPC) |
| [`sbatch_train_perf_amd.sh`](sbatch_train_perf_amd.sh) | 2-node training + PyTorch profiler + Omnistat telemetry (thin variant of `sbatch_train_amd.sh` for the perf-analysis recipe) |
| [`sbatch_infer_amd.sh`](sbatch_infer_amd.sh) | SLURM/Apptainer inference on AMD Instinct (needs a prebuilt overlay + downloaded weights) |

### Perf & scaling orchestration

| Script | Purpose |
|--------|---------|
| [`run_scaling_study.sh`](run_scaling_study.sh) | Submit matched 1/2/4/8-node strong-scaling sweep |
| [`run_optimizer_loop.sh`](run_optimizer_loop.sh) | Entrypoint for the iterative sysopt **perf-optimizer-loop** (Claude CLI; run under `tmux`) |

### Post-processing & analysis

Host-side; run after a job against its logs / traces.

| Script | Purpose |
|--------|---------|
| [`collate_scaling_study.py`](collate_scaling_study.py) | Parse SLURM logs → steady-state throughput table |
| [`parse_convergence.py`](parse_convergence.py) | Extract loss/epoch metrics from training logs |
| [`run_fom_extractor.py`](run_fom_extractor.py) | Compute FOMs + TraceLens↔Omnistat `kernel_correlation.csv` (login-node post-processing of a completed perf run) |

### Patches

| Path | Purpose |
|--------|---------|
| [`patches/`](patches/README.md) | Opt-in upstream patches applied during the overlay build (see [`patches/README.md`](patches/README.md)) |

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
