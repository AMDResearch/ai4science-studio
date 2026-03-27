# ORBIT-2: training and jobs on your AMD Instinct cluster

This recipe is for users who have **SLURM**, **PBS**, or a similar scheduler on an **institutional or lab cluster** with **AMD Instinct** GPUs (for example MI210, MI250X, MI300 series). **Frontier** at OLCF is only one reference system; your center’s **ROCm** and **PyTorch** builds may target a different GPU generation—use **site modules** or **containers** your admins provide and align with the [ORBIT-2 upstream README](https://github.com/XiaoWang-Github/ORBIT-2).

## Prerequisites

1. **Stage data** on shared filesystem paths; see [data-access.md](data-access.md).
2. **Clone** [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2) and build a **conda** (or equivalent) environment with a **PyTorch ROCm** build that matches **your** GPU architecture and ROCm major version—follow upstream install lines for **AMD**; treat Frontier-class examples as a reference, not a fixed GPU pin.
3. Install dependencies upstream lists (`xformers`, `mpi4py`, `pip install -e .`, etc.) for the ROCm stack your site supports.

## Configuration

- In YAML, set **`trainer.gpu_type: "amd"`**.
- Set **`parallelism`** (`fsdp`, `simple_ddp`, `tensor_par`, `seq_par`) so the product matches **your** job’s **total GPU count** (see comments in upstream configs). Scale **down** from Frontier-scale examples proportionally.
- For large fields, enable **`tiling.do_tiling`** and tune `div` / `overlap` per upstream guidance.

## Job scripts

- Start from **`examples/launch_intermediate.sh`** (or other `examples/launch_*.sh` upstream provides).
- Replace **`#SBATCH`** (or PBS directives) with your **account**, **partition**, **time**, **GPUs per node**, and **node count**.
- Point **`conda`** or **module** loads at paths valid on **your** cluster.
- Keep **`low_res_dir`**, **`high_res_dir`**, and related fields consistent with [data-access.md](data-access.md).
- **AI4Science Studio helpers** (this repo, same model folder): under [`../examples/`](../examples/) use **`preflight_orbit2.py`** on a login node (after `conda activate`) to verify paths, `climate_learn`, and optional config/checkpoint; submit **`sbatch_infer_mi2508x.sh`** for an AMD HPC Fund–style **`mi2508x`** partition template, or call **`run_visualize.py`** from your own `srun` line (it forwards to upstream `visualize.py` with the correct `cwd` and `PYTHONPATH`). Set **`ORBIT2_ROOT`** to your upstream clone.

## Visualization and inference

Use the same environment and checkpoint wiring as [inference-and-visualization.md](inference-and-visualization.md); run `visualize.py` (or the launch script’s inner command) on a **GPU node** or interactively with the same ROCm stack.

## Compared with Frontier (OLCF)

For OLCF-specific modules, Orion paths, and exascale-oriented launch examples, see [training-hpc.md](training-hpc.md). This file is the **general** AMD Instinct cluster path.
