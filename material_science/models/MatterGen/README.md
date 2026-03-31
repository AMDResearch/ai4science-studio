# MatterGen

**Hugging Face:** [`microsoft/mattergen`](https://huggingface.co/microsoft/mattergen)
**Upstream code:** [`microsoft/mattergen`](https://github.com/microsoft/mattergen)
**Companion tool:** [`microsoft/mattersim`](https://github.com/microsoft/mattersim) (ML force field for structure relaxation)
**Paper:** [MatterGen: a generative model for inorganic materials design](https://www.nature.com/articles/s41586-025-08628-5) (Nature, 2025)
**License:** MIT

## What it does

MatterGen is a **diffusion-based generative model** for inorganic materials design developed by Microsoft Research. It generates novel crystal structures with user-specified properties — including composition, symmetry, magnetic density, electronic properties, and thermodynamic stability — rather than screening existing databases.

The model generates a joint distribution over **atom types, lattice parameters, and atomic coordinates**, enabling unconditional generation as well as fine-grained multi-property conditioning.

## Model variants

| Pretrained name | Conditioning |
|---|---|
| `mattergen_base` | Unconditional generation |
| `dft_mag_density` | Magnetic density (`dft_mag_density`) |
| `chemical_system_energy_above_hull` | Composition (`chemical_system`) + stability (`energy_above_hull`) |

Additional fine-tuned variants covering band gap, bulk modulus, and other DFT properties are available on the [Hugging Face model card](https://huggingface.co/microsoft/mattergen).

## Training data

| Dataset | License |
|---|---|
| Materials Project (MP v2022.10.28) | CC BY 4.0 |
| Alexandria dataset | CC BY 4.0 |

The default training split uses `mp_20` — Materials Project structures with ≤ 20 atoms per unit cell.

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/inference/`](recipes/inference/) | Generate novel crystal structures, conditioned or unconditional |
| [`recipes/train/`](recipes/train/) | Train or fine-tune MatterGen on new property datasets |

## Installation

```bash
git clone https://github.com/microsoft/mattergen.git
cd mattergen
bash src/setup.bash
```

> **Note:** The setup script installs ROCm-compatible forks of `pytorch_scatter` and `pytorch_sparse` (from `silogen/pytorch_scatter` and `silogen/pytorch_sparse`). These replace CUDA-only upstream packages and are required for AMD GPU support.

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** with ROCm 7.0 and the official ROCm PyTorch image:

```
rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
```

Launch container:

```bash
docker run -d --runtime=amd -e AMD_VISIBLE_DEVICES=all --name mattergen \
    -v $(pwd):/workspace/ rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1 \
    tail -f /dev/null
```

Training performance on a **single MI300X**:

| Configuration | Time |
|---|---|
| 900 epochs (default) | ~15 hours |

See [AMD ROCm blog post](https://rocm.blogs.amd.com/artificial-intelligence/mattergen/README.html) for the full walkthrough including training and evaluation.
