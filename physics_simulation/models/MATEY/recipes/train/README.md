# MATEY — Training

> **Ready-to-run scripts:** see [`../../examples/`](../../examples/) for `docker_run.sh`, `run_train.sh`, and `sbatch_train_mi300x.sh`.

## Overview

MATEY is trained from scratch on simulation datasets. This recipe covers single-GPU training on a workstation and multi-node distributed training on an HPC cluster with AMD GPUs.

## Prerequisites

- Docker with AMD Container Toolkit (local) or Apptainer/Singularity (HPC)
- ROCm 6.3.1+ (6.4+ recommended for MI300X)
- MATEY repo cloned: `git clone https://github.com/ORNL/MATEY.git`
- Training data available (see dataset section below)

## Quick start (Docker, single GPU)

```bash
cd physics_simulation/models/MATEY/examples
./docker_run.sh train
```

The script launches a ROCm PyTorch container, installs MATEY dependencies, and runs training with the JHTDB demo config.

## Manual setup

```bash
# 1. Create / activate environment
#    Inside the ROCm container or a conda env:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

# 2. Install MATEY
cd MATEY
pip install -e .

# 3. Download the JHTDB demo data
#    Obtain HDF5 files from https://turbulence.pha.jhu.edu/
#    and place them in ./data/JHTDB/

# 4. Single-GPU training
python basic_usage.py \
    --run_name my_run \
    --config basic_config \
    --yaml_config ./config/Demo_JHUTDB_TT.yaml
```

## Multi-node training (HPC / SLURM)

The Frontier SLURM script in `examples/` closely follows the [upstream demo submission script](https://github.com/ORNL/MATEY/blob/main/examples/submit_JHTDB_demo.sh):

```bash
# 2 nodes × 8 GPUs = 16 tasks total
sbatch sbatch_train_mi300x.sh
```

Key SLURM settings used at ORNL on Frontier:

| Directive | Value |
|-----------|-------|
| `--nodes` | 2 |
| `--ntasks-per-node` | 8 |
| `--constraint` | nvme |
| `--gpu-bind` | closest |
| `OMP_NUM_THREADS` | 1 |

## Configuration

MATEY is configured via YAML files in `./config/`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `model.*` | Architecture hyperparameters (depth, heads, patch size) |
| `data.dataset` | Dataset loader class and data path |
| `train.batch_size` | Per-GPU batch size |
| `train.epochs` | Number of training epochs |
| `train.lr` | Learning rate |

Copy and modify an existing config as a starting point:

```bash
cp config/Demo_JHUTDB_TT.yaml config/my_experiment.yaml
```

## AMD / ROCm notes

- Validated by ORNL on Frontier (MI250X) with ROCm 6.3.1 and the `matey-env-rocm631.sh` environment
- MIOpen kernel auto-tuning (`MIOPEN_FIND_MODE=1`) is recommended for repeated training runs
- MI300X (192 GB HBM) allows significantly larger batch sizes than MI250X (64 GB); set `MATEY_BATCH_SIZE` accordingly
- `torch.compile` compatibility on ROCm not yet validated — treat as an optimization opportunity

## Performance profiling

The upstream Frontier script is a good baseline for profiling:

```bash
# Profile a short run (adjust --time to 10 min)
sbatch --time=0:10:00 sbatch_train_mi300x.sh
```

ROCm profiling tools:

```bash
# rocprof trace (inside container or module)
rocprof --stats python basic_usage.py ...

# PyTorch profiler
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
python -c "
import torch.profiler as p
with p.profile(activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.CUDA]) as prof:
    # run a few training steps
    ...
print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```
