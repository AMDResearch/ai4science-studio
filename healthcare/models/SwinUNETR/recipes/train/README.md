# SwinUNETR — Training Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

> **Ready-to-run scripts:** see [`../../examples/`](../../examples/) for `docker_run.sh`, `run_train.sh`, and `sbatch_train_mi300x.sh`.

Train SwinUNETR for 3D lung tumor segmentation on the NSCLC-Radiomics dataset.

## Prerequisites

- Container: `rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0` (training) or `rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0` (inference)
- GPU: AMD Instinct with ROCm 6.4+ driver
- Runtime: Docker with GPU device access
- Data: NSCLC-Radiomics (auto-downloaded by MONAI)

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATA_DIR` | No | `/data` | NSCLC-Radiomics data root |
| `CKPT_DIR` | No | `/workspace/checkpoints` | Checkpoint / log directory |
| `MAX_EPOCHS` | No | `700` | Training epochs |
| `ROI_X` | No | `96` | Patch size X (up to 480 on MI300X) |
| `ROI_Y` | No | `96` | Patch size Y (up to 480 on MI300X) |
| `ROI_Z` | No | `96` | Patch size Z |
| `FEATURE_SIZE` | No | `48` | Encoder feature size |
| `NUM_WORKERS` | No | `64` | DataLoader workers |
| `AMP_DTYPE` | No | `float16` | AMP dtype (float16 recommended, bfloat16 underperforms) |

## Setup

```bash
git clone https://github.com/silogen/ai-samples.git
cd life-science/medical-imaging/swinunetr
```

Build and launch via Docker Compose:

```bash
docker compose up --build
```

**Base image:** `rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0`

## Key training parameters

| Parameter | Value |
|---|---|
| ROI dimensions (default) | 96×96×96 |
| ROI dimensions (optimized) | 480×480×96 |
| Batch size | 1 |
| `num_workers` | 64 |
| Max epochs | 700 |
| Feature size | 48 |

## DataLoader configuration

Configure PyTorch DataLoader for optimal data throughput:

```python
DataLoader(
    dataset,
    batch_size=1,
    num_workers=64,           # >32 workers eliminates data bottleneck
    persistent_workers=True,  # cache workers between epochs (~14 s/epoch saved)
    pin_memory=True,          # faster CPU→GPU transfer
)
```

## MIOpen auto-tuning

Set these environment variables before training for a **~3× overall speedup** (>5× forward/backward pass):

```bash
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=3
```

> On **ROCm 6.4+ / PyTorch 2.6.0+** these are set by default — no manual configuration needed.

## Mixed precision

Use **float16** (not bfloat16 — bfloat16 underperforms for this model):

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast(dtype=torch.float16):
    output = model(inputs)
loss = criterion(output, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Dataset access

MONAI downloads NSCLC-Radiomics automatically:

```python
from monai.apps import TciaDataset

dataset = TciaDataset(
    collection="NSCLC-Radiomics",
    section="training",
    transform=transforms,
    download=True,
)
```

## References

- [AMD ROCm blog — Training SwinUNETR](https://rocm.blogs.amd.com/artificial-intelligence/running-swinunetr-amd/README.html)
- [silogen/ai-samples recipe](https://github.com/silogen/ai-samples/tree/main/life-science/medical-imaging/swinunetr)
- [MONAI SwinUNETR upstream](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)
