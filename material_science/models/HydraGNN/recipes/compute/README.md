# HydraGNN: compute access and alternatives

**Frontier** is an **OLCF** resource for approved projects—not a public cloud. Most readers will use **smaller AMD Instinct** systems (institutional clusters or cloud), **Hugging Face checkpoints**, and **inference or small fine-tuning** instead of full multisource pretraining.

## Frontier (OLCF)

**Requirements (conceptual):** OLCF user account, accepted project, allocation, and compliance with OLCF policies. The published **Predictive GFM 2024** pretraining used **large-scale DDP** on Frontier-class hardware (see the [model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024)).

**Where to start (official):**

- [OLCF user documentation](https://docs.olcf.ornl.gov/index.html)
- [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- Project and allocation processes are described in OLCF's current **proposals / allocations** pages (follow links from the docs site).

AI4Science Studio cannot approve or provision OLCF access.

## AMD Instinct clusters (institutional or cloud)

For **your own SLURM (or similar) cluster** with AMD Instinct GPUs, follow the HydraGNN install path on branch **`Predictive_GFM_2024`** for **ROCm** and **PyTorch** as documented there, and scale **down** parallelism and batch size relative to Frontier-scale jobs. See also [../train/README.md](../train/README.md).

## AMD GPU options outside Frontier

These are **starting points**; eligibility, capacity, and pricing change over time—read the live pages.

| Resource | What it is (summary) | Link |
|----------|----------------------|------|
| **AMD Developer Cloud** | Cloud access to AMD Instinct GPUs (e.g. MI300-class) with ROCm-oriented images; complimentary credits / pay-as-you-go options. | [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) |
| **AI & HPC Fund** | Program for research/academic institutions to apply for accelerated computing resources. | [AMD AI & HPC Fund](https://www.amd.com/en/corporate/hpc-fund.html) |
| **Cloud hub** | Index of AI/HPC cloud access options AMD lists. | [AI & HPC cloud access](https://www.amd.com/en/developer/resources/cloud-access.html) |

## Reality check

- **Inference** (including **ensemble** forward passes) typically needs **far less** capacity than **multidataset pretraining** and HPO.
- Match your job to **hardware you can actually schedule**; do not assume Frontier-scale node counts.
