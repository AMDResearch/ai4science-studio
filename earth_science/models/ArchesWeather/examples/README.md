# ArchesWeather examples

Ready-to-run scripts for [ArchesWeather](../README.md). All scripts use the
Docker image built from [silogen/ai-samples](https://github.com/silogen/ai-samples)
(`ai4sciences/geoarches-training`).

## Files

| Script | Purpose |
|--------|---------|
| [`docker_run.sh`](docker_run.sh) | Build the Docker image and launch an interactive container |
| [`run_inference.sh`](run_inference.sh) | Run inference and evaluation (inside container) |
| [`run_train.sh`](run_train.sh) | Pretrain or fine-tune ArchesWeather / ArchesWeatherGen (inside container) |
| [`sbatch_inference_amd.sh`](sbatch_inference_amd.sh) | SLURM driver for `run_inference.sh` on MI300X |
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | SLURM driver for `run_train.sh` on MI300X |

## Quick start — Docker (local workstation / interactive node)

```bash
# 1. Build the image and launch a container (auto-clones silogen/ai-samples)
./docker_run.sh

# 2. Attach to the running container
docker exec -it archesweather bash

# 3a. Run inference (evaluates on ERA5 test year 2020 by default)
bash /examples/run_inference.sh

# 3b. Run training (pretrain archesweather, 16-mixed precision, batch 8)
bash /examples/run_train.sh

# Stop and remove the container when done
docker rm -f archesweather
```

The script auto-detects the AMD Container Toolkit; if unavailable it falls
back to passing `/dev/kfd` and `/dev/dri` directly.

## Quick start — bare metal

```bash
# 1. Clone silogen/ai-samples and install per the geoarches-training Dockerfile
git clone --depth=1 https://github.com/silogen/ai-samples.git
# Follow the Dockerfile instructions to install geoarches and dependencies

# 2. Run inference
bash run_inference.sh

# 3. Or train
bash run_train.sh
```

## SLURM submission

Build an Apptainer/Singularity SIF from the silogen geoarches-training Dockerfile,
then edit the `#SBATCH` directives (`--partition`, `--account`) and submit:

```bash
# Inference (evaluate on test year 2020)
AW_SIF=/path/to/archesweather.sif sbatch sbatch_inference_amd.sh

# Training (pretrain, 16-mixed precision)
AW_SIF=/path/to/archesweather.sif sbatch sbatch_train_amd.sh

# Fine-tuning from a pretrained checkpoint
AW_PHASE=finetune AW_LOAD_FROM=/workspace/checkpoints/archesweather-seed0 \
    AW_SIF=/path/to/archesweather.sif sbatch sbatch_train_amd.sh
```

Omit `AW_SIF` to use the bare-metal environment.

## Key MI300X notes

- **Batch size 8** at `16-mixed` / `bf16-mixed` vs **5** at `32-true` (192 GB HBM)
- ERA5 dataset for full training is ~735 GB — see `recipes/train/README.md` for download
- No code changes required; no ROCm-specific optimizations applied (torch.compile / MIOpen tuning not yet validated)
