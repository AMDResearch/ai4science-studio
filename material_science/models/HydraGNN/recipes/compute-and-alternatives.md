# HydraGNN: compute access and alternatives

**Frontier** is an **OLCF** resource for approved projects—not a public cloud. Most readers will use **smaller AMD or NVIDIA systems**, **Hugging Face checkpoints**, and **inference or small fine-tuning** instead of full multisource pretraining.

## Frontier (OLCF)

**Requirements (conceptual):** OLCF user account, accepted project, allocation, and compliance with OLCF policies. The published **Predictive GFM 2024** pretraining used **large-scale DDP** on Frontier-class hardware (see the [model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024)).

**Where to start (official):**

- [OLCF user documentation](https://docs.olcf.ornl.gov/index.html)  
- [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)  
- Project and allocation processes are described in OLCF’s current **proposals / allocations** pages (follow links from the docs site).

AI4Science Studio cannot approve or provision OLCF access.

## AMD GPU options outside Frontier

These are **starting points**; eligibility, capacity, and pricing change over time—read the live pages.

| Resource | What it is (summary) | Link |
|----------|----------------------|------|
| **AMD Developer Cloud** | Cloud access to AMD Instinct GPUs (e.g. MI300-class) with ROCm-oriented images; complimentary credits / pay-as-you-go options described on AMD’s site. | [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) |
| **Getting started** | Step-by-step for provisioning an environment (VM, JupyterLab, containers). | [How to get started on the AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html) |
| **AI & HPC Fund** | Program for research/academic institutions to apply for accelerated computing resources and related support (review cycles, terms on AMD’s page). | [AMD AI & HPC Fund](https://www.amd.com/en/corporate/hpc-fund.html) |
| **Application** | Form linked from the fund page for requests. | [HPC Fund application](https://www.amd.com/en/forms/registration/amd-hpc-fund-research-accelerator.html) |
| **Cloud hub** | Index of AI/HPC cloud access options AMD lists. | [AI & HPC cloud access](https://www.amd.com/en/developer/resources/cloud-access.html) |

After you have a node, follow the **HydraGNN** install path on branch **`Predictive_GFM_2024`** for **ROCm** or **CUDA** as documented there, and scale **down** parallelism and batch size relative to Frontier-scale jobs.

## NVIDIA GPUs

HydraGNN is **PyTorch-based**; use a **CUDA** build of PyTorch and dependencies compatible with the upstream README on **`Predictive_GFM_2024`** if your hardware is NVIDIA-based.

## Reality check

- **Inference** (including **ensemble** forward passes) typically needs **far less** capacity than **multidataset pretraining** and HPO.  
- Match your job to **hardware you can actually schedule**; do not assume Frontier-scale node counts.
