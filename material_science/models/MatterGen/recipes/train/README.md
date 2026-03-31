# MatterGen — Training Recipe

Train MatterGen from scratch or fine-tune a pretrained checkpoint on a new property dataset.

## Prerequisites

- AMD Instinct GPU (MI300X recommended; 192 GB HBM enables large batch sizes)
- Docker with AMD Container Toolkit
- ~50 GB disk space for `mp_20` dataset

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

Install dependencies:

```bash
cd /workspace
bash /workspace/src/setup.bash
```

## Prepare the dataset

```bash
# Pull the mp_20 data via Git LFS
git lfs pull -I data-release/mp-20/ --exclude=""

# Unzip and convert to model format
unzip data-release/mp-20/mp_20.zip -d datasets

csv-to-dataset \
    --csv-folder datasets/mp_20/ \
    --dataset-name mp_20 \
    --cache-folder datasets/cache
```

## Train

Default training run (900 epochs, ~15 hours on single MI300X):

```bash
mattergen-train data_module=mp_20 ~trainer.logger
```

Multi-GPU training uses PyTorch Lightning's DDP strategy automatically when multiple GPUs are available.

## Fine-tune on a new property

To fine-tune a pretrained checkpoint on a custom property dataset, override the data module and model config:

```bash
mattergen-train \
    data_module=<your_data_module> \
    model=<pretrained_variant> \
    ~trainer.logger
```

See the [MatterGen GitHub repo](https://github.com/microsoft/mattergen) for documentation on defining custom data modules and property conditioning.

## References

- [AMD ROCm blog post](https://rocm.blogs.amd.com/artificial-intelligence/mattergen/README.html)
- [MatterGen paper (Nature, 2025)](https://www.nature.com/articles/s41586-025-08628-5)
- [Hugging Face model card](https://huggingface.co/microsoft/mattergen)
