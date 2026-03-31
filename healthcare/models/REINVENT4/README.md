# REINVENT4

> **Research / engineering use only.** This recipe is intended for computational chemistry research and drug discovery research workflows. It does not constitute medical advice and must not be used with patient-identifiable data or PHI.

**Hugging Face:** N/A — model weights distributed via GitHub repository and ChEMBL dataset
**Upstream code:** [`MolecularAI/REINVENT4`](https://github.com/MolecularAI/REINVENT4)
**Paper:** [REINVENT4: Modern AI–driven generative molecule design](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00812-5) (J Cheminformatics, 2024)
**License:** See [upstream repository](https://github.com/MolecularAI/REINVENT4) (Apache-2.0)

## What it does

REINVENT4 is a **reinforcement learning–driven molecular design tool** developed by AstraZeneca. It generates and optimizes candidate drug molecules according to user-defined property profiles expressed as multi-component scoring functions.

Supported design modes:

| Mode | Description |
|---|---|
| De novo design | Generate molecules from scratch |
| Scaffold hopping | Replace a core scaffold while preserving key interactions |
| R-group replacement | Vary substituents on a fixed scaffold |
| Linker design | Connect two fragments with a novel linker |
| Transfer learning (TL) | Fine-tune a prior model toward a target chemical space |
| Molecule optimization | Iteratively improve existing molecules |

The RL loop couples a generative prior (LSTM or transformer) with a scoring function that can integrate docking scores, ADMET predictions, and other property models.

## Data

**ChEMBL35** is the recommended pretraining corpus — an open-source bioactive molecule database with drug-like properties:

```
https://ftp.ebi.ac.uk/pub/databases/chembl/
```

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/finetune/`](recipes/finetune/) | Transfer learning to bias generation toward a target chemical space |

## Installation

**Bare metal (conda):**

```bash
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
conda env create -f reinvent.yml
conda activate reinvent4
python install.py
```

**Container (ROCm):**

Use the ROCm PyTorch base image and remove PyTorch entries from `pyproject.toml` before running the install script to avoid reinstalling PyTorch over the container's pre-installed version:

```bash
# Base image
FROM rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.10_pytorch_release_2.2.1

# Strip PyTorch/TorchVision from pyproject.toml deps, then install
RUN sed -i '/torch/d' pyproject.toml && python install.py
```

GPU device access (bare-metal Docker without AMD Container Toolkit):

```bash
docker run --device=/dev/kfd --device=/dev/dri/renderD<GPU_ID> \
    --group-add video ...
```

## AMD / ROCm notes

Validated at **AMD Instinct MI300X** (8× GPU node, TensorWave) with ROCm 6.3.3 and PyTorch 2.2.1. REINVENT4 runs on ROCm out-of-the-box — no code changes required.

### AMP optimization

Adding **Automatic Mixed Precision** to the transfer learning training loop (`reinvent/runmodes/TL/learning.py`) yielded **10–60% training time reductions**:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Optimizations that did NOT help

- MIOpen auto-tuning (`MIOPEN_FIND_MODE`, `MIOPEN_FIND_ENFORCE`) — no measurable gain
- `torch.compile` — no measurable gain
- `TunableOp` — no measurable gain

Batch size increases did yield observable additional speedups.

## References

- [AMD ROCm blog — Part 1: REINVENT4](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html)
- [AstraZeneca × AMD collaboration blog](https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html)
