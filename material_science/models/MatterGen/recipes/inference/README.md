# MatterGen — Inference Recipe

Generate novel inorganic crystal structures using pretrained MatterGen checkpoints.

## Prerequisites

- AMD Instinct GPU (MI300X recommended) with ROCm kernel-mode driver (`amdgpu-dkms`)
- Docker with AMD Container Toolkit

## Setup

```bash
git clone https://github.com/microsoft/mattergen.git
cd mattergen
```

Launch the ROCm container:

```bash
docker run -d --runtime=amd -e AMD_VISIBLE_DEVICES=all --name mattergen \
    -v $(pwd):/workspace/ rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1 \
    tail -f /dev/null
docker exec -it mattergen bash
```

Install dependencies inside the container:

```bash
cd /workspace
bash /workspace/src/setup.bash
```

## Unconditional generation

Generate crystal structures with no property constraints:

```bash
export MODEL_NAME=mattergen_base
export RESULTS_PATH=/workspace/results/

mattergen-generate $RESULTS_PATH \
    --pretrained-name=$MODEL_NAME \
    --batch_size=16 \
    --num_batches=1
```

## Property-conditioned generation

Generate structures with a target magnetic density:

```bash
mattergen-generate $RESULTS_PATH \
    --pretrained-name=dft_mag_density \
    --batch_size=16 \
    --properties_to_condition_on="{'dft_mag_density': 0.15}" \
    --diffusion_guidance_factor=2.0
```

Generate structures conditioned on composition and thermodynamic stability:

```bash
mattergen-generate $RESULTS_PATH \
    --pretrained-name=chemical_system_energy_above_hull \
    --batch_size=16 \
    --properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}" \
    --diffusion_guidance_factor=2.0
```

## Evaluation with MatterSim

Relax generated structures and compute validity metrics using the MatterSim ML force field:

```bash
# Pull the Alexandria reference data (required for evaluation)
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""

mattergen-evaluate \
    --structures_path=$RESULTS_PATH \
    --relax=True \
    --structure_matcher='disordered' \
    --save_as="$RESULTS_PATH/metrics.json"
```

MatterSim variants:

| Variant | Speed | Accuracy |
|---|---|---|
| `MatterSim-v1.0.0-1M` | Faster | Lower |
| `MatterSim-v1.0.0-5M` | Slower | Higher |

## References

- [AMD ROCm blog post](https://rocm.blogs.amd.com/artificial-intelligence/mattergen/README.html)
- [MatterGen paper (Nature, 2025)](https://www.nature.com/articles/s41586-025-08628-5)
- [Hugging Face model card](https://huggingface.co/microsoft/mattergen)
