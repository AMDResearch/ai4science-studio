# ArchesWeather / ArchesWeatherGen

**Hugging Face:** [`gcouairon/ArchesWeather`](https://huggingface.co/gcouairon/ArchesWeather)
**Upstream code:** [`silogen/ai-samples`](https://github.com/silogen/ai-samples) (path: `ai4sciences/geoarches-training`)
**Paper:** [ArchesWeather & ArchesWeatherGen: efficient AI weather forecasting](https://arxiv.org/abs/2412.12971) (Couairon et al., 2024)
**License:** See upstream repository and Hugging Face model card

## What it does

ArchesWeather and ArchesWeatherGen are **global AI weather forecasting models** operating at 1.5° horizontal resolution, trained on ERA5 reanalysis data (1979–2018).

| Model | Type | Description |
|---|---|---|
| `archesweather-m-seed{0..3}` | Deterministic | 3D Swin U-Net transformer; 4 seeds for ensemble averaging |
| `archesweathergen` | Generative | Flow-matching model trained on ArchesWeather residuals; 25 neural-network calls per forecast |

**Architecture:** 3D Swin U-Net with 2D local horizontal attention blocks.

| Variant | Parameters | Layers |
|---|---|---|
| S-model | ~44M | 16 |
| M-model | ~84M | 32 |

## Input variables

6 upper-air + 4 surface ERA5 variables across 13 pressure levels.

## Training data

| Split | Years | Purpose |
|---|---|---|
| Training | 1979–2018 | Full pretraining |
| Fine-tuning | 2007–2018 (Phase 2), 2019 (ArchesWeatherGen OOD) | Recent-past bias correction |
| Test | 2020 | Evaluation |

**Dataset size:** ~735 GB for the full ERA5 download.

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/train/`](recipes/train/) | Pretrain and fine-tune ArchesWeather / ArchesWeatherGen on ERA5 |
| [`recipes/inference/`](recipes/inference/) | Run deterministic and ensemble forecasts; evaluate and visualize |

## Installation

```bash
git clone https://github.com/silogen/ai-samples.git
cd ai4sciences/geoarches-training
```

Build the Docker image (includes all `geoarches` dependencies):

```bash
docker build -t pytorch_training_geoarches:latest .
```

Run the container:

```bash
docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --shm-size=16g \
    pytorch_training_geoarches bash
```

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** (192 GB HBM3). The 192 GB memory enables larger batch sizes than typical 24 GB GPU setups.

**Base image:** Provided Dockerfile (ROCm PyTorch); see the `ai4sciences/geoarches-training` subfolder for the exact version pinned.

**Precision / batch size on MI300X:**

| Precision | Max batch size (M-model) |
|---|---|
| 32-true | 5 |
| 16-mixed | 8 |
| bf16-mixed | 8 |

**Configuration management:** Hydra — override any parameter with `++key=value` syntax on the command line.

No AMD-specific code changes were required. No ROCm-specific optimizations were applied beyond the standard Dockerfile; the blog notes future optimization opportunities remain unexplored.

## References

- [AMD ROCm blog — Training ArchesWeather on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/geoarches-training/README.html)
- [ArchesWeather paper (arXiv)](https://arxiv.org/abs/2412.12971)
- [Hugging Face model card](https://huggingface.co/gcouairon/ArchesWeather)
- [geoarches documentation](https://geoarches.readthedocs.io)
