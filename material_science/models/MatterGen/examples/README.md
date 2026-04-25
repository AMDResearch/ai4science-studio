# MatterGen examples

Ready-to-run scripts for [MatterGen](../README.md). All scripts target AMD Instinct
GPUs via ROCm and use ROCm-compatible forks of `pytorch_scatter` / `pytorch_sparse`.

## Files

| Script | Purpose |
|--------|---------|
| [`docker_run.sh`](docker_run.sh) | Launch container, clone, and install MatterGen |
| [`run_inference.sh`](run_inference.sh) | Generate novel crystal structures (inside container) |
| [`run_train.sh`](run_train.sh) | Train MatterGen on the mp_20 dataset (inside container) |
| [`sbatch_inference_amd.sh`](sbatch_inference_amd.sh) | SLURM driver for generation on MI300X |
| [`sbatch_train_amd.sh`](sbatch_train_amd.sh) | SLURM driver for training on MI300X |

## Quick start — Docker (local workstation / interactive node)

```bash
# 1. Start container and install MatterGen (~5 min, installs ROCm forks)
./docker_run.sh

# 2. Attach to the running container
docker exec -it mattergen bash

# 3a. Generate structures (unconditional, 16 structures)
bash /workspace/run_inference.sh

# 3b. Train on mp_20 dataset (~15 hours / 900 epochs on MI300X)
bash /workspace/run_train.sh

# Stop and remove the container when done
docker rm -f mattergen
```

The script auto-detects the AMD Container Toolkit; if unavailable it falls
back to passing `/dev/kfd` and `/dev/dri` directly.

## Quick start — bare metal

```bash
# 1. Follow upstream setup from microsoft/mattergen (requires ROCm forks)
git clone https://github.com/microsoft/mattergen.git && cd mattergen
bash src/setup.bash   # installs ROCm forks of pytorch_scatter / pytorch_sparse

# 2. Generate structures
PRETRAINED_NAME=mattergen_base bash run_inference.sh

# 3. Or train
bash run_train.sh
```

## SLURM submission

Build an Apptainer/Singularity SIF from the rocm/pytorch base image with MatterGen
installed, then edit the `#SBATCH` directives (`--partition`, `--account`) and submit:

```bash
# Generate 16 structures (unconditional)
MG_SIF=/path/to/mattergen.sif sbatch sbatch_inference_amd.sh

# Property-conditioned generation (magnetic density)
MG_SIF=/path/to/mattergen.sif \
MG_PRETRAINED_NAME=dft_mag_density \
MG_PROPERTIES="{'dft_mag_density': 0.15}" \
    sbatch sbatch_inference_amd.sh

# Training (900 epochs, ~15 hours on MI300X)
MG_SIF=/path/to/mattergen.sif sbatch sbatch_train_amd.sh
```

Omit `MG_SIF` to use the bare-metal environment.

## Key MI300X notes

- **Base image:** `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1`
- **ROCm forks required:** `silogen/pytorch_scatter`, `silogen/pytorch_sparse` — upstream CUDA-only versions do not work
- Training ~15 hours / 900 epochs on a single MI300X
- AMD Container Toolkit runtime (`--runtime=amd`) preferred
