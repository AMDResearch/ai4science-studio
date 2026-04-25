# HydraGNN recipes

Studio runbooks for [HydraGNN (Predictive GFM 2024)](../README.md) (`mlupopa/HydraGNN_Predictive_GFM_2024` on Hugging Face). **Authoritative scripts and configs** live in [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) on branch **`Predictive_GFM_2024`**.

## Use cases

| Doc | Audience | Summary |
|-----|----------|---------|
| [inference/README.md](inference/README.md) | Most users | Download ensemble checkpoints from Hub, align `config.json` with weights, run inference via HydraGNN; optional ensemble averaging and UQ via `examples/ensemble_learning`. |
| [train/README.md](train/README.md) | OLCF / large HPC | Frontier-scale distributed pretraining, DeepHyper HPO, and `examples/multidataset_hpo`—only where you have a suitable allocation. |

## Data and compute (read before large runs)

| Doc | Summary |
|-----|---------|
| [data/README.md](data/README.md) | `ADIOS_files/` on Hub (CLI staging), Constellation / Globus mirror, OLCF copy-out notes; large downloads; energy/force preprocessing summarized on the model card. |
| [compute/README.md](compute/README.md) | Frontier access is restricted; AMD cloud / HPC Fund, institutional AMD clusters; inference vs pretraining footprint. |
