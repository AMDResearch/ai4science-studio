# SwinUNETR — Inference Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

Optimized inference for 3D lung tumor segmentation using a trained SwinUNETR checkpoint, with AMP and `torch.compile` for maximum throughput.

## Prerequisites

- Container: `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0`
- GPU: AMD Instinct with ROCm 7.0+ driver
- Runtime: Docker with GPU device access
- Weights: trained SwinUNETR `.pth` checkpoint (see [../train/](../train/))

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CHECKPOINT` | Yes | -- | Path to trained `.pth` checkpoint |
| `INPUT_DIR` | No | `/data/test` | Input NIfTI directory |
| `OUTPUT_DIR` | No | `/workspace/results` | Output segmentation masks |
| `ROI_X` | No | `96` | Sliding-window patch size X |
| `ROI_Y` | No | `96` | Sliding-window patch size Y |
| `ROI_Z` | No | `96` | Sliding-window patch size Z |
| `USE_COMPILE` | No | `1` | Enable `torch.compile` max-autotune |
| `AMP_DTYPE` | No | `float16` | AMP dtype |

## Setup

```bash
git clone https://github.com/silogen/ai-samples.git
cd life-science/medical-imaging/swinunetr
```

**Base image for inference:** `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0`

## Optimized inference

Apply both AMP and `torch.compile` for a **2.9× speedup** over unoptimized baseline:

```python
import torch
from monai.inferers import SlidingWindowInferer

# Load model
model = SwinUNETR(...)
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval().cuda()

# Apply optimizations
model = torch.compile(model, mode="max-autotune")

inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    output = inferer(inputs, model)
```

## Performance results

All benchmarks on AMD Instinct MI300X (ROCm 7.0):

| ROI size | Baseline | AMP only | AMP + compile |
|---|---|---|---|
| 96×96×96 | ~5 s/case | ~3.2 s/case | **1.74 s/case** |
| 128×128×128 | ~4.8 s/case | ~3.1 s/case | **1.66 s/case** |
| 256×256×128 | ~3.4 s/case | ~2.2 s/case | **1.17 s/case** |

- AMP alone: 35–39% improvement
- Combined AMP + `torch.compile`: **2.9× overall**
- Memory footprint: ~25% reduction with combined approach
- Dice score: consistent within ±2% — no accuracy degradation

## Large ROI advantage of MI300X

The 192 GB HBM3 on MI300X enables ROIs up to **480×480×96** — approximately 25× larger than what fits on a typical 24 GB GPU. This allows full-resolution volumetric inference without tiling artifacts.

## Notes on precision

Use **float16** for AMP; **bfloat16 underperforms** for SwinUNETR on ROCm and should be avoided.

## References

- [AMD ROCm blog — Inference optimization](https://rocm.blogs.amd.com/artificial-intelligence/swinunetr-inference-optimization/README.html)
- [AMD ROCm blog — Training SwinUNETR](https://rocm.blogs.amd.com/artificial-intelligence/running-swinunetr-amd/README.html)
- [silogen/ai-samples recipe](https://github.com/silogen/ai-samples/tree/main/life-science/medical-imaging/swinunetr)
