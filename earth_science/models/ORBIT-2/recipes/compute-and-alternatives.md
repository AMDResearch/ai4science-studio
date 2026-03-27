# ORBIT-2: compute access and alternatives

**Frontier** is an **OLCF** resource for approved projects—not a public cloud. Most readers will use **smaller AMD Instinct** systems (institutional clusters or cloud), Hugging Face **checkpoints**, and the **inference** path instead of full exascale training.

## Frontier (OLCF)

**Requirements (conceptual):** OLCF user account, accepted project, allocation, and compliance with OLCF policies. Training examples in ORBIT-2 assume **SLURM** and the **Frontier** software stack (ROCm modules, networking libraries for distributed training, etc.).

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

After you have a node, follow the **AMD ROCm** installation path in the ORBIT-2 GitHub README (ROCm + PyTorch versions stated there) and scale **down** parallelism relative to Frontier.

For **your own SLURM (or similar) cluster** with AMD Instinct GPUs, see [local-cluster-amd.md](local-cluster-amd.md).

## Reality check

- **Inference and visualization** with released checkpoints often need **far fewer** GPUs than exascale **pretraining**.  
- Match your job to **hardware you can actually schedule**; do not assume Frontier-scale node counts.
