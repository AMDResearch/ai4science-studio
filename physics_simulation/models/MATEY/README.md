# MATEY

| Field | Value |
|-------|-------|
| **Hugging Face model id** | N/A — weights are trained from scratch; pre-trained checkpoints are not publicly distributed |
| **Task** | Spatiotemporal surrogate modeling for multiphysics systems (turbulence, plasma, fluid dynamics) |
| **License** | [MIT](https://github.com/ORNL/MATEY/blob/main/LICENSE) |
| **Upstream code** | <https://github.com/ORNL/MATEY> |
| **Paper** | [MATEY: multiscale adaptive foundation models for spatiotemporal physical systems (arXiv:2412.20601)](https://arxiv.org/abs/2412.20601) |
| **Maintained by** | Oak Ridge National Laboratory (ORNL) |

## Overview

MATEY is an open-source framework for developing transformer-based spatiotemporal foundation models for physical systems. It targets multiscale multiphysics simulations — including plasma edge dynamics, turbulence modeling, and fluid dynamics — with support for both structured (HDF5, NetCDF) and unstructured scientific datasets.

Key design goals:
- **Multiscale representations** for spatiotemporal physical fields
- **Scalable HPC training** via distributed data parallelism (DDP) on multi-node GPU clusters
- **Configurable YAML-driven workflows** for custom datasets and model architectures
- Validated at scale on the Frontier supercomputer (2 nodes × 8 AMD MI250X GPUs per node)

## Obtaining model weights

MATEY is a training framework; there are no centrally distributed checkpoints. Train a model from scratch using one of the included demo configurations:

```bash
# clone the upstream repo
git clone https://github.com/ORNL/MATEY.git
cd MATEY

# train with the JHTDB turbulence demo (single GPU)
python basic_usage.py \
    --run_name demo \
    --config basic_config \
    --yaml_config ./config/Demo_JHUTDB_TT.yaml
```

See [`recipes/train/`](recipes/train/README.md) for full setup and multi-node instructions.

## Datasets

MATEY ships with demo configurations for:

| Dataset | Domain | Format |
|---------|--------|--------|
| [Johns Hopkins Turbulence Database (JHTDB)](https://turbulence.pha.jhu.edu/) | Fluid dynamics / turbulence | HDF5 |
| Custom plasma/multiphysics data | Plasma edge dynamics | HDF5 / NetCDF |

Custom datasets are added by writing a loader script in `./matey/data_utils/`.

## Recipes

| Task | Link |
|------|------|
| Training | [recipes/train/](recipes/train/README.md) |
| Inference / rollout | [recipes/inference/](recipes/inference/README.md) |

## AMD / ROCm notes

MATEY was developed and validated on Frontier (AMD MI250X). The upstream repo ships an environment setup script (`matey-env-rocm631.sh`) targeting ROCm 6.3.1. For MI300X systems, ROCm 6.4+ is recommended. See [`examples/`](examples/) for Docker and SLURM scripts.
