# SwinUNETR

> **Research / engineering use only.** This recipe is intended for medical imaging research workflows only. It does not constitute medical advice, clinical diagnosis, or treatment recommendations. Must not be used with patient-identifiable data or PHI outside of properly governed research environments.

**Hugging Face:** N/A — weights obtained via MONAI model zoo or trained from scratch
**Upstream code:** [`Project-MONAI/research-contributions/SwinUNETR`](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)
**AMD recipe code:** [`silogen/ai-samples — life-science/medical-imaging/swinunetr`](https://github.com/silogen/ai-samples/tree/main/life-science/medical-imaging/swinunetr)
**Paper:** [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266)
**License:** Apache-2.0 (MONAI); CC BY 4.0 (figures)

## What it does

SwinUNETR is a **3D medical image segmentation model** built on Swin Transformers. It uses a hierarchical encoder-decoder architecture (U-Net style) to segment volumetric CT and MRI scans. In the AMD-validated recipe, SwinUNETR is applied to **lung tumor segmentation** using the NSCLC-Radiomics dataset from The Cancer Imaging Archive (TCIA).

The MI300X's **192 GB HBM3** memory enables ROI sizes up to **25× larger** than what is possible on typical 24 GB GPUs, enabling full-resolution volumetric inference.

## Data

**NSCLC-Radiomics** — Lung tumor CT segmentation dataset from The Cancer Imaging Archive:

```python
from monai.apps import TciaDataset
# Automated download via MONAI's TciaDataset API
dataset = TciaDataset(collection="NSCLC-Radiomics", ...)
```

No manual download required; MONAI handles fetching automatically.

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/train/`](recipes/train/) | Train SwinUNETR on NSCLC-Radiomics with MIOpen auto-tuning |
| [`recipes/inference/`](recipes/inference/) | Optimized inference with AMP and torch.compile |

## Installation

```bash
git clone https://github.com/silogen/ai-samples.git
cd life-science/medical-imaging/swinunetr
```

The recipe ships a Docker Compose file for a one-command setup:

```bash
docker compose up --build
```

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** with two ROCm versions across the training and inference blogs:

| Blog | ROCm version | PyTorch | OS |
|---|---|---|---|
| Training | 6.4 | 2.6.0 | Ubuntu 22.04 |
| Inference optimization | 7.0 | 2.6.0 | Ubuntu 24.04 |

### Training optimizations

| Technique | Speedup |
|---|---|
| MIOpen auto-tuning (`MIOPEN_FIND_MODE=1`, `MIOPEN_FIND_ENFORCE=3`) | ~3× training speedup; >5× forward/backward pass |
| `num_workers > 32` + `persistent_workers=True` | Saves ~14 s/epoch |
| `pin_memory=True` | Reduces CPU→GPU transfer overhead |

> On **ROCm 6.4+ / PyTorch 2.6.0+** MIOpen auto-tuning is enabled by default — manual env vars no longer needed.

**Float16 vs bfloat16:** bfloat16 underperforms compared to float16 for this model — use `float16` for AMP.

### Inference optimizations

| Technique | Speedup |
|---|---|
| `autocast` (AMP, float16) | 35–39% |
| `torch.compile(mode="max-autotune")` | Additional gain |
| **Combined** | **2.9× faster** vs. baseline |

Memory footprint reduced by ~25% with combined AMP + compile.

### Optimizations that did NOT help

- `torch.compile` and `TunableOps` for training (minimal benefit)

## References

- [AMD ROCm blog — Training SwinUNETR](https://rocm.blogs.amd.com/artificial-intelligence/running-swinunetr-amd/README.html)
- [AMD ROCm blog — Inference optimization](https://rocm.blogs.amd.com/artificial-intelligence/swinunetr-inference-optimization/README.html)
- [AstraZeneca × AMD collaboration blog](https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html)
- [MONAI model zoo](https://monai.io/model-zoo.html)
- [SwinUNETR paper (arXiv)](https://arxiv.org/abs/2201.01266)
