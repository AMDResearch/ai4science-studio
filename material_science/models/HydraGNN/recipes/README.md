# HydraGNN recipes

Studio runbooks for [HydraGNN (Predictive GFM 2024)](../README.md) (`mlupopa/HydraGNN_Predictive_GFM_2024` on Hugging Face). **Authoritative scripts and configs** live in [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) on branch **`Predictive_GFM_2024`**.

## Use cases

| Doc | Audience | Summary |
|-----|----------|---------|
| [inference-and-visualization.md](inference-and-visualization.md) | Most users | Download ensemble checkpoints from Hub, align `config.json` with weights, run inference via HydraGNN; optional ensemble averaging and UQ via `examples/ensemble_learning`. |
| [training-hpc.md](training-hpc.md) | OLCF / large HPC | Frontier-scale distributed pretraining, DeepHyper HPO, and `examples/multidataset_hpo`—only where you have a suitable allocation. |

## Data and compute (read before large runs)

| Doc | Summary |
|-----|---------|
| [data-access.md](data-access.md) | `ADIOS_files/` on Hub, full package via OLCF Data Constellation; large downloads; energy/force preprocessing summarized on the model card. |
| [compute-and-alternatives.md](compute-and-alternatives.md) | Frontier access is restricted; AMD cloud / HPC Fund and other GPUs; inference vs pretraining footprint. |
