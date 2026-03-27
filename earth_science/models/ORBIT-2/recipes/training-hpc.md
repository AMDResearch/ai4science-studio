# ORBIT-2: large-scale training on HPC (Frontier-oriented)

This recipe summarizes the **exascale-oriented** workflow documented in [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2). It is **not** something most users can run without a major HPC allocation.

## Who can run this

- **Oak Ridge Leadership Computing Facility (OLCF)** users with an approved project and **compute hours on Frontier** (or a similarly sized **AMD** system with a comparable **PyTorch + ROCm** stack).  
- **General public / casual cloud users:** you typically do **not** have this path. Use Hugging Face checkpoints and [inference-and-visualization.md](inference-and-visualization.md), or smaller-scale experiments on hardware you control—see [compute-and-alternatives.md](compute-and-alternatives.md). For training or visualization on **institutional AMD clusters**, see [local-cluster-amd.md](local-cluster-amd.md).

## High-level steps (from upstream)

1. **Environment** — On Frontier-class AMD nodes, follow the **Frontier** installation block in the ORBIT-2 README (Python 3.11 conda env, PyTorch ROCm build, `xformers`, `mpi4py`, `pip install -e .`).
2. **Choose a config** — Under `configs/`, pick a size (e.g. `interm_8m.yaml`, `interm_117m.yaml`, `interm_1b.yaml`, `interm_10b.yaml`). Larger models need more nodes/GPUs.
3. **GPU and parallelism** — In YAML, set `trainer.gpu_type: "amd"`. Set `parallelism` fields (`fsdp`, `simple_ddp`, `tensor_par`, `seq_par`) so the product matches your **total GPU count** (see upstream comments).
4. **TILES** — For very large images, enable `tiling.do_tiling` and tune `div` / `overlap` per upstream guidance (patch divisibility matters).
5. **Data paths** — Populate `low_res_dir`, `high_res_dir`, `spatial_resolution`, and variable dictionaries to point at **your** staged dataset. Paths on Orion are **project- and allocation-specific**; see [data-access.md](data-access.md).
6. **Submit training** — Edit `examples/launch_intermediate.sh` (`#SBATCH -A`, conda activation, config path), then `sbatch launch_intermediate.sh`. Monitor `flash-{JOBID}.out` as documented upstream.

## Fine-tuning from a pretrained checkpoint

Use `pretrain:` / `checkpoint:` in the YAML as described in the upstream “Hyperparameter Configuration” section when continuing or fine-tuning.

## Disclaimer

AI4Science Studio does **not** grant OLCF access, storage, or software support. For policies, queues, and project requests, use **official OLCF documentation** and the user assistance process linked from [compute-and-alternatives.md](compute-and-alternatives.md).
