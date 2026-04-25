# SemlaFlow

> **Research / engineering use only.** This recipe is intended for computational chemistry research and drug discovery research workflows. It does not constitute medical advice and must not be used with patient-identifiable data or PHI.

**Hugging Face:** N/A — pretrained checkpoints distributed via [Google Drive](https://github.com/rssrwn/semla-flow) (see upstream repo for links)
**Upstream code:** [`rssrwn/semla-flow`](https://github.com/rssrwn/semla-flow)
**Paper:** [SemlaFlow: Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching](https://openreview.net/forum?id=bee2G6pEh0)
**License:** MIT

## What it does

SemlaFlow is a **3D molecular structure generation model** developed in collaboration between AstraZeneca and academic partners. It generates novel drug-like molecules as complete 3D graphs, producing a joint distribution over atom types, coordinates, bond types, and formal charges.

Key design choices:

- **E(3)-equivariant Semla architecture** — latent attention layers that respect 3D rotational/translational symmetry
- **Equivariant flow matching** — continuous ODE-based sampling; as few as **20 steps** required
- **>100× faster** sampling than diffusion-based baselines at equivalent quality
- No post-hoc inference of bond orders or charges — generated directly

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/inference/`](recipes/inference/) | Sample novel 3D molecules from a pretrained checkpoint |

## Installation

```bash
git clone https://github.com/rssrwn/semla-flow.git
cd semla-flow
```

Create the conda environment (**replace `pytorch-cuda` with `rocm-pytorch`** for AMD):

```bash
# Edit environment.yml: swap pytorch-cuda=<ver> → rocm-pytorch
conda env create -f environment.yml
conda activate semlaflow
```

Key dependencies: PyTorch 2.6.0, Python 3.12, rdkit, lightning, openbabel-wheel.

## AMD / ROCm notes

Validated at **AMD Instinct MI300X** (8× GPU node, TensorWave) with **ROCm 6.4.1** and PyTorch 2.6.0. All code ran without changes to model logic.

> **Important:** ROCm 6.3.3 caused import errors. Upgrading to **ROCm 6.4.1** (Ubuntu 24.04, Python 3.12) resolved all issues and also provided a **~15% training speed improvement**.

Docker base image:

```
rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

### Performance optimizations

| Technique | Speedup |
|---|---|
| ROCm 6.3.3 → 6.4.1 base image | ~15% |
| `torch.compile(model, dynamic=True, fullgraph=False)` | ~44% |
| `--no_ema` flag (disable exponential moving average) | ~8% |
| **Combined** | **~52% total** (12.57 → 5.14 min/epoch) |

Compiler cache: set limit to 1000 in config to avoid recompilation bottlenecks during training.

### Optimizations that did NOT help

- `torch.compile` with `fullgraph=True` — caused errors
- Other compile modes (beyond default + dynamic) — no additional gain

## References

- [AMD ROCm blog — Part 2: SemlaFlow](https://rocm.blogs.amd.com/artificial-intelligence/semlaflow/README.html)
- [AstraZeneca × AMD collaboration blog](https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html)
- [SemlaFlow paper (OpenReview)](https://openreview.net/forum?id=bee2G6pEh0)
