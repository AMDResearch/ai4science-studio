# HydraGNN: training and jobs on your AMD Instinct cluster

This recipe is for users who have **SLURM**, **PBS**, or a similar scheduler on an **institutional or lab cluster** with **AMD Instinct** GPUs (for example MI210, MI250X, MI300 series). Published **Predictive GFM 2024** pretraining used **Frontier**; your site may ship a different **ROCm** / **PyTorch** combination—use **modules** or **containers** that match **your** GPU generation and follow [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) branch **`Predictive_GFM_2024`**.

## Prerequisites

1. **Stage ADIOS data and checkpoints** on shared paths; see [data-access.md](data-access.md).
2. **Clone** HydraGNN, check out **`Predictive_GFM_2024`**, and install (`pip install -e .` and requirement files) against a **PyTorch ROCm** build compatible with **your** hardware.
3. Optional: install **DeepHyper** if you run HPO workflows under **`examples/multidataset_hpo`**.

## Scale and parallelism

- Pretraining used **distributed data parallelism (DDP)** at very large scale; on your cluster, set **world size**, **nodes**, and **batch** settings to match **available GPUs** and memory per device (see model card and upstream examples).
- **HPO** is optional at smaller scale; entry points remain under **`examples/multidataset_hpo`** on the same branch.

## Job layout

- Adapt upstream job templates (if any) or wrap your **torchrun** / **srun** / **mpi** launch with your scheduler’s **account**, **partition**, **GPUs per node**, and **walltime**.
- Ensure every rank sees the same **JSON** configs and **ADIOS** paths on shared storage.

## Inference and ensemble work

For loading checkpoints, **`hydragnn.load_existing_model`**, **`run_prediction`**, and **`examples/ensemble_learning`**, see [inference-and-visualization.md](inference-and-visualization.md).

## Compared with Frontier (OLCF)

For OLCF-oriented context and omnistat-scale notes, see [training-hpc.md](training-hpc.md). This file is the **general** AMD Instinct cluster path.
