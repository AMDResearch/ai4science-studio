# ORBIT-2 recipes

Studio runbooks for [ORBIT-2](../README.md) (`jychoi-hpc/ORBIT-2` on Hugging Face). **Authoritative scripts and configs** live in [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2).

## Use cases

| Doc | Audience | Summary |
|-----|----------|---------|
| [inference/README.md](inference/README.md) | Most users | Download checkpoints from Hub, wire into upstream `visualize.py` / visualization job scripts, run downscaling-style inference. |
| [train/README.md](train/README.md) | OLCF / large HPC | Frontier-style `sbatch`, configs under `configs/`, parallelism and TILES—only where you have an allocation. |

## Data and compute (read before large runs)

| Doc | Summary |
|-----|---------|
| [data/README.md](data/README.md) | Public dataset via Constellation + Globus; staging on shared cluster FS; OLCF Orion and copy-out notes; config paths `low_res_dir` / `high_res_dir` are yours to fill. |
| [compute/README.md](compute/README.md) | Frontier access is restricted; AMD Developer Cloud, AI & HPC Fund, institutional AMD clusters. |

## Performance engineering (AMD Instinct)

| Doc | Summary |
|-----|---------|
| [perf-analysis/README.md](perf-analysis/README.md) | Multi-subagent bottleneck analysis (TraceLens + Omnistat) after a perf run; FOM contract, parallelism defaults, landmines. Baselines: [one-node-gpu-baseline.md](perf-analysis/one-node-gpu-baseline.md), [BASELINE_LOCKIN.md](perf-analysis/BASELINE_LOCKIN.md). |
| [perf-optimizer-loop/README.md](perf-optimizer-loop/README.md) | Iterative sysopt loop (one lever/iter, accept/revert on throughput). Levers tried/blocked in [lever_catalog.yaml](perf-optimizer-loop/lever_catalog.yaml); GEMM-time finding in [gemm-attribution.md](perf-optimizer-loop/gemm-attribution.md); HBM staging in [STAGING_ERA5_FOR_HBM.md](perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md). |
