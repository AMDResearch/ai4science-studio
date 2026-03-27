# ORBIT-2: inference and visualization

This is the most common **AI4Science Studio** entry point: use **published checkpoints** from Hugging Face with the **official code** to run visualization / inference-style workflows.

## Prerequisites

1. Clone [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2) and follow its **Installation** section, using the upstream steps for **AMD Instinct** with **ROCm**.
2. Download the checkpoint(s) and matching YAML you need from [`jychoi-hpc/ORBIT-2`](https://huggingface.co/jychoi-hpc/ORBIT-2) (see the live file tree under `pretrain/`, `us-finetune/`, `global-finetune/`, etc.).

## Point the code at a checkpoint

The upstream README describes two ways to select a checkpoint for visualization:

1. **Config file** — set `trainer.pretrain` (or the field upstream documents for your config) to the local path of the `.ckpt` file.
2. **CLI override** — pass `--checkpoint /path/to/model.ckpt` to `visualize.py` when that matches the version of the repo you use.

Always align the **YAML config** with the **checkpoint** (matching model scale and task). Upstream lists example pairs for US and global fine-tuned models; re-read the current GitHub README for exact names.

## Running visualization

- **Frontier (OLCF):** edit `examples/launch_visualize.sh` (allocation account, conda path, config path), then `sbatch launch_visualize.sh`. See the upstream “Tutorial Example → Frontier → Step 4: Visualize Results”.
- **Other clusters / interactive nodes:** run the same `visualize.py` invocation the launch script uses, with your scheduler or `mpirun` layout as appropriate for that system.
- **Studio examples (AI4Science Studio checkout):** in [`../examples/`](../examples/), **`run_visualize.py`** runs upstream `visualize.py` with `ORBIT2_ROOT` set to your clone; **`preflight_orbit2.py`** checks the environment without SLURM; **`sbatch_infer_mi2508x.sh`** is a sample SLURM job for an **`mi2508x`**-style partition. Details: [local-cluster-amd.md](local-cluster-amd.md).

Outputs typically include low-resolution inputs, high-resolution predictions, and comparison plots as described upstream.

## Inference latency (expectations)

The ORBIT-2 paper reports that **after training**, inference can be very fast relative to dynamical downscaling: for example, **milliseconds per global sample** for a small-parameter model and **sub-second** scale for a 10B-parameter model on an 8-GPU setup for a stated ERA5→ERA5 benchmark (see paper tables). Your hardware and batching will differ—treat these as **order-of-magnitude** guidance, not guarantees.

## See also

- [data-access.md](data-access.md) — if you need paired low/high-res data beyond bundled demos.
- [compute-and-alternatives.md](compute-and-alternatives.md) — if you lack Frontier-scale resources.
