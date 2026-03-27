# ORBIT-2

**Hugging Face:** [`jychoi-hpc/ORBIT-2`](https://huggingface.co/jychoi-hpc/ORBIT-2)  
**Code:** [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2)  
**Paper:** [ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling](https://arxiv.org/abs/2505.04802) (arXiv:2505.04802)

ORBIT-2 is a scalable **vision foundation model for global weather and climate downscaling**, combining a Reslim (Residual Slim ViT) architecture with **TILES** (tile-wise sequence scaling) for long sequences and large-scale training. Checkpoints on Hugging Face include **pretrain** and **fine-tuned** weights (e.g. US regional and global precipitation/temperature variants). **Exact file names and folder layout change over time**—always check the [current Hub tree](https://huggingface.co/jychoi-hpc/ORBIT-2/tree/main) before scripting downloads.

## Using this model

Training, inference, visualization, and SLURM examples are maintained in the **GitHub repository**, not in AI4Science Studio. This repo only holds **pointers and runbook-style recipes**:

- [`recipes/README.md`](recipes/README.md) — index  
- [`recipes/inference-and-visualization.md`](recipes/inference-and-visualization.md) — checkpoints + visualization workflow (broad audience)  
- [`recipes/training-hpc.md`](recipes/training-hpc.md) — large-scale / Frontier-oriented training summary  
- [`recipes/data-access.md`](recipes/data-access.md) — Constellation / Globus / OLCF data context  
- [`recipes/compute-and-alternatives.md`](recipes/compute-and-alternatives.md) — Frontier vs AMD cloud / HPC Fund / other GPUs  

## Installation (reference)

Follow the upstream README for **conda** setup. It documents separate instructions for **AMD (ROCm)** and **NVIDIA (CUDA)** stacks (including versions for Frontier-class AMD GPUs and DGX-style NVIDIA GPUs). Do not pin versions here; copy from upstream when you implement.

## Datasets

The ORBIT-2 training release and methodology are described on the [Constellation dataset page](https://doi.ccs.ornl.gov/dataset/e4c2db1f-e88c-5ad0-bb96-59be0ef7c772) (DOI **10.13139/OLCF/2589526**). Underlying public sources include ERA5, PRISM, DAYMET, and IMERG—comply with each provider’s terms if you rebuild pipelines.

## License and attribution

Use the **Hugging Face model card** and **GitHub repository** license terms. If you use ORBIT-2 in research, cite the software and paper (see upstream README for BibTeX). Contacts listed on the Hub card include ORNL investigators (e.g. Dan Lu, Xiao Wang).

## On-disk folder name

This directory is named **`ORBIT-2`** for clarity. The canonical Hub id remains **`jychoi-hpc/ORBIT-2`**.
