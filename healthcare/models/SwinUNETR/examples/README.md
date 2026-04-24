# SwinUNETR examples

Ready-to-run scripts for [SwinUNETR](../README.md). Training and inference use
**different ROCm images** — select the correct mode when launching the container.

## Files

| Script | Purpose |
|--------|---------|
| [`docker_run.sh`](docker_run.sh) | Launch training or inference container (pass `train` or `infer`) |
| [`run_train.sh`](run_train.sh) | Train SwinUNETR on NSCLC-Radiomics with MIOpen auto-tuning (inside container) |
| [`run_inference.sh`](run_inference.sh) | Optimized inference with AMP + torch.compile max-autotune (inside container) |
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | SLURM driver for training on MI300X (ROCm 6.4) |
| [`sbatch_inference_amd.sh`](sbatch_inference_amd.sh) | SLURM driver for inference on MI300X (ROCm 7.0) |

## Quick start — Docker (local workstation / interactive node)

```bash
# Training (ROCm 6.4 image)
./docker_run.sh train
docker exec -it swinunetr-train bash
# Inside the container:
pip install "monai[all]" nibabel
bash /examples/run_train.sh

# Inference (ROCm 7.0 image, torch.compile max-autotune)
./docker_run.sh infer
docker exec -it swinunetr-infer bash
# Inside the container:
pip install "monai[all]" nibabel
CHECKPOINT=/workspace/checkpoints/model_final.pt bash /examples/run_inference.sh
```

The script auto-detects the AMD Container Toolkit; if unavailable it falls
back to passing `/dev/kfd` and `/dev/dri` directly.

## Quick start — bare metal

```bash
# Training (requires ROCm 6.4 + PyTorch 2.6)
pip install "monai[all]" nibabel
bash run_train.sh

# Inference (requires ROCm 7.0 + PyTorch 2.6 for torch.compile max-autotune)
pip install "monai[all]" nibabel
CHECKPOINT=/path/to/model.pth bash run_inference.sh
```

## SLURM submission

Build separate Apptainer/Singularity SIFs for training (ROCm 6.4) and inference
(ROCm 7.0), then edit `#SBATCH` directives and submit:

```bash
# Training
SU_SIF=/path/to/swinunetr-train.sif sbatch sbatch_train_amd.sh

# Large-ROI training (up to 480×480×96 on MI300X 192 GB)
SU_SIF=/path/to/swinunetr-train.sif \
SU_ROI_X=480 SU_ROI_Y=480 SU_ROI_Z=96 \
    sbatch sbatch_train_amd.sh

# Inference (with torch.compile max-autotune, ~2.9× total speedup)
SU_SIF=/path/to/swinunetr-infer.sif \
SU_CHECKPOINT=/path/to/model.pth \
    sbatch sbatch_inference_amd.sh
```

Omit `SU_SIF` to use the bare-metal environment.

## Key MI300X notes

- **Training image:** `rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0`
- **Inference image:** `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- **MIOpen auto-tuning** (`MIOPEN_FIND_MODE=1`, `MIOPEN_FIND_ENFORCE=3`): ~3× training speedup; default on ROCm 6.4+ / PyTorch 2.6+
- **ROI up to 480×480×96** on MI300X 192 GB (vs 96×96×96 on 24 GB GPUs)
- **AMP dtype:** use `float16` — `bfloat16` underperforms for this model
- **Inference:** `torch.compile(max-autotune)` adds ~35–39%; combined with AMP = **2.9× total speedup**
