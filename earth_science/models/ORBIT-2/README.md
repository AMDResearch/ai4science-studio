# ORBIT-2

**Hugging Face:** [`jychoi-hpc/ORBIT-2`](https://huggingface.co/jychoi-hpc/ORBIT-2)  
**Code:** [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2)  
**Paper:** [ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling](https://arxiv.org/abs/2505.04802) (arXiv:2505.04802)

ORBIT-2 is a scalable **vision foundation model for global weather and climate downscaling**, combining a Reslim (Residual Slim ViT) architecture with **TILES** (tile-wise sequence scaling) for long sequences and large-scale training. Checkpoints on Hugging Face include **pretrain** and **fine-tuned** weights (e.g. US regional and global precipitation/temperature variants). **Exact file names and folder layout change over time**—always check the [current Hub tree](https://huggingface.co/jychoi-hpc/ORBIT-2/tree/main) before scripting downloads.

## Validated on AMD Instinct (MI355X)

ORBIT-2 has been reproduced end-to-end on AMD Instinct **MI355X** (gfx950, 8 GPU/node) via the Studio recipes:

- **Inference / visualization** — single- and multi-GPU runs, containerized (ROCm PyTorch + Apptainer overlay), synthetic-data smoke test and real ERA5/PRISM paths. See [`recipes/inference/`](recipes/inference/).
- **Training** — multi-node training validated at **1, 2, 4, and 8 nodes** (8 GPU/node) with hybrid FSDP. Clean near-linear scaling at small node counts; per-epoch loss decreases monotonically as a crash/hang sanity gate (not an absolute-convergence claim). Full weak-/strong-scaling tables, methodology, and caveats are in [`recipes/train/`](recipes/train/).
- **Run model** — Apptainer overlay for Python deps + a shared scratch root (`AI4S_SHARED_DIR`); every knob is an `ORBIT2_*` environment variable with documented defaults (see [`examples/`](examples/)).

## Using this model

Training, inference, visualization, and SLURM examples are maintained in the **GitHub repository**, not in AI4Science Studio. This repo only holds **pointers and runbook-style recipes**:

- [`recipes/inference/`](recipes/inference/) — checkpoints + visualization workflow (broad audience)  
- [`recipes/train/`](recipes/train/) — large-scale / Frontier-oriented training summary  
- [`recipes/data/`](recipes/data/) — Constellation / Globus / OLCF data context and staging  
- [`recipes/compute/`](recipes/compute/) — Frontier, AMD cloud / HPC Fund, institutional clusters  
- [`recipes/perf-analysis/`](recipes/perf-analysis/) — multi-subagent bottleneck analysis (TraceLens + Omnistat) on AMD Instinct  
- [`recipes/perf-optimizer-loop/`](recipes/perf-optimizer-loop/) — iterative throughput optimizer loop + lever catalog  
- [`examples/`](examples/) — ready-to-run Docker, SLURM, and smoke-test scripts  

## Installation (reference)

Follow the upstream README for **conda** setup. The recipes here follow the **AMD Instinct** + **ROCm** path described there (PyTorch builds and versions as upstream states). Anything outside that path is best handled from the upstream install matrix. Do not pin versions here; copy from upstream when you implement.

## Datasets

The ORBIT-2 training release and methodology are described on the [Constellation dataset page](https://doi.ccs.ornl.gov/dataset/e4c2db1f-e88c-5ad0-bb96-59be0ef7c772) (DOI **10.13139/OLCF/2589526**). Underlying public sources include ERA5, PRISM, DAYMET, and IMERG—comply with each provider’s terms if you rebuild pipelines.

## License and attribution

Use the **Hugging Face model card** and **GitHub repository** license terms. If you use ORBIT-2 in research, cite the software and paper (see upstream README for BibTeX). Contacts listed on the Hub card include ORNL investigators (e.g. Dan Lu, Xiao Wang).

## On-disk folder name

This directory is named **`ORBIT-2`** for clarity. The canonical Hub id remains **`jychoi-hpc/ORBIT-2`**.
