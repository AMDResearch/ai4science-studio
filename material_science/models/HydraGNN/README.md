# HydraGNN (Predictive GFM 2024)

**Hugging Face:** [`mlupopa/HydraGNN_Predictive_GFM_2024`](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024)  
**Code:** [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) (branch **`Predictive_GFM_2024`**)  
**Dataset / artifacts (OLCF):** [HydraGNN_Predictive_GFM_2024 on Constellation](https://doi.ccs.ornl.gov/dataset/3a49c8df-83f7-5d32-84be-f81d289e7cdd) (DOI **10.13139/OLCF/2474799**)

This release is an **ensemble of fifteen predictive graph foundation models (GFMs)** for **atomistic materials** modeling, trained with [HydraGNN](https://github.com/ORNL/HydraGNN) using **multi-task** objectives for **system energy** (graph-level) and **atomic forces** (node-level). Training aggregates five open datasets totaling on the order of **154M** structures (see the [model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024) for scope, curation, and limitations—e.g. **no excited states**). The Hugging Face repo hosts **ADIOS**-format preprocessed data under `ADIOS_files/` and checkpoints under `Ensemble_of_models/`. **Exact file names and layout can change**—check the [current Hub tree](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024/tree/main) before scripting downloads.

## Using this model

Authoritative training, HPO, ensemble, and inference scripts live in **upstream HydraGNN** (branch above). This repo holds **pointers and runbook-style recipes**:

- [`recipes/README.md`](recipes/README.md) — index  
- [`recipes/inference-and-visualization.md`](recipes/inference-and-visualization.md) — checkpoints from Hub + inference / ensemble workflows  
- [`recipes/training-hpc.md`](recipes/training-hpc.md) — Frontier-scale pretraining / HPO summary  
- [`recipes/local-cluster-amd.md`](recipes/local-cluster-amd.md) — SLURM/PBS-style runs on **your** AMD Instinct cluster  
- [`recipes/data-access.md`](recipes/data-access.md) — Hub ADIOS bundles, Constellation, staging, and data context  
- [`recipes/compute-and-alternatives.md`](recipes/compute-and-alternatives.md) — OLCF, AMD cloud / HPC Fund, institutional clusters  

## Installation (reference)

Follow the upstream **HydraGNN** README and dependency files on branch **`Predictive_GFM_2024`** (for example `pip install -e .` and the provided `requirements*.txt` / install scripts). Do not pin versions here; copy from upstream when you implement.

## License and attribution

Use the **Hugging Face model card** and **HydraGNN** repository license terms (**BSD-3-Clause Clear** on the Hub card). Recommended citation (from the model card):

> M. Lupo Pasini, J. Y. Choi, K. Mehta, P. Zhang, D. Rogers, J. Bae, K. Ibrahim, A. Aji, K. W. Schulz, J. Polo, and P. Balaprakash, *HydraGNN_Predictive_GFM_2024 — Ensemble of predictive graph foundation models for group state atomistic materials modeling*, DOI [10.13139/OLCF/2474799](https://doi.org/10.13139/OLCF/2474799).

## On-disk folder name

This directory is named **`HydraGNN`** for clarity. The canonical Hub id remains **`mlupopa/HydraGNN_Predictive_GFM_2024`**.
