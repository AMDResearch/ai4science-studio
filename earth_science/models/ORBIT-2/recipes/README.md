# ORBIT-2 recipes

Studio runbooks for [ORBIT-2](../README.md) (`jychoi-hpc/ORBIT-2` on Hugging Face). **Authoritative scripts and configs** live in [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2).

## Use cases

| Doc | Audience | Summary |
|-----|----------|---------|
| [inference-and-visualization.md](inference-and-visualization.md) | Most users | Download checkpoints from Hub, wire into upstream `visualize.py` / visualization job scripts, run downscaling-style inference. |
| [local-cluster-amd.md](local-cluster-amd.md) | Institutional AMD HPC | ROCm + PyTorch on **your** Instinct cluster; adapt upstream launch scripts, parallelism, and data paths—see also [data-access.md](data-access.md). |
| [training-hpc.md](training-hpc.md) | OLCF / large HPC | Frontier-style `sbatch`, configs under `configs/`, parallelism and TILES—only where you have an allocation. |

## Data and compute (read before large runs)

| Doc | Summary |
|-----|---------|
| [data-access.md](data-access.md) | Public dataset via Constellation + Globus; staging on shared cluster FS; OLCF Orion and copy-out notes; config paths `low_res_dir` / `high_res_dir` are yours to fill. |
| [compute-and-alternatives.md](compute-and-alternatives.md) | Frontier access is restricted; AMD Developer Cloud, AI & HPC Fund, institutional AMD clusters ([local-cluster-amd.md](local-cluster-amd.md)). |
