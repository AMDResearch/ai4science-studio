# ORBIT-2 recipes

Studio runbooks for [ORBIT-2](../README.md) (`jychoi-hpc/ORBIT-2` on Hugging Face). **Authoritative scripts and configs** live in [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2).

## Use cases

| Doc | Audience | Summary |
|-----|----------|---------|
| [inference-and-visualization.md](inference-and-visualization.md) | Most users | Download checkpoints from Hub, wire into upstream `visualize.py` / visualization job scripts, run downscaling-style inference. |
| [training-hpc.md](training-hpc.md) | OLCF / large HPC | Frontier-style `sbatch`, configs under `configs/`, parallelism and TILES—only where you have an allocation. |

## Data and compute (read before large runs)

| Doc | Summary |
|-----|---------|
| [data-access.md](data-access.md) | Public dataset via Constellation + Globus; OLCF Orion concepts; config paths `low_res_dir` / `high_res_dir` are yours to fill. |
| [compute-and-alternatives.md](compute-and-alternatives.md) | Frontier access is restricted; AMD Developer Cloud, AI & HPC Fund, and other options; NVIDIA path from upstream. |
