# HydraGNN: large-scale training on HPC (Frontier-oriented)

This recipe summarizes the **exascale-oriented** workflow described on the [Hugging Face model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024) and implemented in [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) branch **`Predictive_GFM_2024`**. It is **not** something most users can run without a major HPC allocation and staged **ADIOS** datasets.

## Who can run this

- **OLCF** users with an approved project and compute on **Frontier** (or comparable systems with a supported PyTorch / ROCm or CUDA stack and sufficient scale).  
- **General users:** prefer Hub checkpoints and [inference-and-visualization.md](inference-and-visualization.md), or smaller experiments on hardware you control—see [compute-and-alternatives.md](compute-and-alternatives.md).

## High-level steps (from upstream)

1. **Environment** — Follow HydraGNN installation instructions on branch **`Predictive_GFM_2024`** for your target system (modules, PyTorch build, and optional distributed dependencies).
2. **Data** — Preprocessed training data in **ADIOS** form are described under `ADIOS_files/` on the Hub and in the Constellation release (see [data-access.md](data-access.md)). Paths in your configs must point at **your** staged files.
3. **HPO and pretraining** — The model card describes **hyperparameter optimization** with [DeepHyper](https://github.com/deephyper/deephyper), short HPO training epochs with early stopping, selection of fifteen trials, and **continued pretraining** up to a capped epoch count. Entry points live under **`examples/multidataset_hpo`** on the **`Predictive_GFM_2024`** branch.
4. **Distributed training** — Pretraining was performed with **distributed data parallelism (DDP)** at very large node counts (see the model card). Reproduce at smaller scale by reducing world size and batch settings per upstream guidance.
5. **Energy accounting** — The publication references **omnistat** for training energy measurement on large runs; use that only if your site supports it.

## Disclaimer

AI4Science Studio does **not** grant OLCF access, storage, or software support. For policies, queues, and project requests, use **official OLCF documentation** and the user assistance process linked from [compute-and-alternatives.md](compute-and-alternatives.md).
