# ORBIT-2: large-scale training on HPC (Frontier-oriented)

This recipe summarizes the **exascale-oriented** workflow documented in [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2). It is **not** something most users can run without a major HPC allocation.

## Who can run this

- **Oak Ridge Leadership Computing Facility (OLCF)** users with an approved project and **compute hours on Frontier** (or a similarly sized **AMD** system with a comparable **PyTorch + ROCm** stack).
- **General public / casual cloud users:** you typically do **not** have this path. Use Hugging Face checkpoints and [../inference/README.md](../inference/README.md), or smaller-scale experiments on hardware you control—see [../compute/README.md](../compute/README.md). For training or visualization on **institutional AMD clusters**, see [../compute/README.md](../compute/README.md).

## High-level steps (from upstream)

1. **Environment** — On Frontier-class AMD nodes, follow the **Frontier** installation block in the ORBIT-2 README (Python 3.11 conda env, PyTorch ROCm build, `xformers`, `mpi4py`, `pip install -e .`).
2. **Choose a config** — Under `configs/`, pick a size (e.g. `interm_8m.yaml`, `interm_117m.yaml`, `interm_1b.yaml`, `interm_10b.yaml`). Larger models need more nodes/GPUs. Those upstream configs set **`trainer.data_type: bfloat16`**; Bayes-CAST-style trees often ship ERA5 configs such as **`edm_8m_era5.yaml`** with the same dtype — Studio templates (`interm_8m_era5.yaml`) mirror that via `ORBIT2_DATA_TYPE` / `render_orbit2_config.py`.
3. **GPU and parallelism** — In YAML, set `trainer.gpu_type: "amd"`. Set `parallelism` fields (`fsdp`, `simple_ddp`, `tensor_par`, `seq_par`) so the product matches your **total GPU count** (see upstream comments).
4. **TILES** — For very large images, enable `tiling.do_tiling` and tune `div` / `overlap` per upstream guidance (patch divisibility matters).
5. **Data paths** — Populate `low_res_dir`, `high_res_dir`, `spatial_resolution`, and variable dictionaries to point at **your** staged dataset. Paths on Orion are **project- and allocation-specific**; see [../data/README.md](../data/README.md).
6. **Submit training** — Edit `examples/launch_intermediate.sh` (`#SBATCH -A`, conda activation, config path), then `sbatch launch_intermediate.sh`. Monitor `flash-{JOBID}.out` as documented upstream.

## Weak-scaling results — EDM, ERA5 1.0°, MI355X

The table below is a **weak-scaling** sweep of the Bayes-CAST **EDM** model on AMD **MI355X** nodes (8 GPU/node). It is **weak** scaling: the **per-rank batch is fixed at 1024**, so the global batch grows with GPU count and per-GPU work stays constant. Parallelism is **HSDP** (`fsdp=8` within a node, `simple_ddp=N` across nodes), precision **bf16**, dataset **ERA5 1.0°** (`spatial_resolution=111`). "Steady s/batch" is the mean batch wall time over epochs 2+ (warmup epochs 0–1 and the first batch of each epoch excluded). **Efficiency = t₁ₙₒ𝒹ₑ / t_Nₙₒ𝒹ₑ** — a weak-scaling metric that reflects communication-overlap quality / RCCL bandwidth (≈1.0 is ideal).

| Nodes | GPUs | per-rank batch | global batch | steady s/batch | samples/s | efficiency | steady batches | trust |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 8  | 1024 | 8192  | 2.639 | 3103.7  | 1.00 | 57 | ✅ |
| 2 | 16 | 1024 | 16384 | 2.626 | 6240.2  | 1.01 | 27 | ✅ |
| 4 | 32 | 1024 | 32768 | 2.021 | 16212.7 | 1.31 | 9  | ⚠️ |
| 8 | 64 | 1024 | 65536 | 2.060 | 31820.6 | 1.28 | 9  | ⚠️ |

> **Read this before quoting the table.** Clean weak scaling is demonstrated **only at 1→2 nodes**: steady step time is essentially flat (2.639 s → 2.626 s, efficiency ≈ 1.01), confirming RCCL/IB communication is fully overlapped. The **4- and 8-node efficiencies (>1.0) are a measurement artifact, NOT super-linear speedup.** At per-rank batch 1024 the global batch reaches 32768 / 65536, which is large relative to the staged ERA5 1.0° tree, so only **~3 batches/epoch** run at 4N/8N (`steady batches` column collapses from 57 → 9). Steady step time over so few batches is noise-dominated, and `throughput = global_batch / step_time` inflates mechanically. The loss curves corroborate this: 4N/8N final loss (0.90 / 0.97) is **worse** than 1N/2N (0.81 / 0.76) because each rank sees too few steps to train. To get trustworthy 4/8-node weak-scaling numbers, the dataset must be enlarged so big global batches still yield many steps per epoch — that is **Phase 2** (stage more ERA5, see [`STAGING_ERA5_FOR_HBM.md`](../perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md)).

- **Loss sanity:** all four pass the monotonic-decrease check; note this only verifies loss goes down each epoch, not absolute convergence (see caveat above).
- **Artifacts:** [`collate_scaling_study.py`](../../examples/collate_scaling_study.py) writes `scaling_study.{md,csv,json}` under `$ORBIT2_OUTPUT_DIR`.
- **Reproduce:** [`run_scaling_study.sh`](../../examples/run_scaling_study.sh) `--nodes 1` then `--nodes 2,4,8` with `ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml`, `ORBIT2_BATCH_SIZE=1024`, `ORBIT2_DATA_TYPE=bfloat16`, `ORBIT2_MAX_EPOCH=6`, `ORBIT2_MAX_BATCHES=20`.

## Fine-tuning from a pretrained checkpoint

Use `pretrain:` / `checkpoint:` in the YAML as described in the upstream "Hyperparameter Configuration" section when continuing or fine-tuning.

## Disclaimer

AI4Science Studio does **not** grant OLCF access, storage, or software support. For policies, queues, and project requests, use **official OLCF documentation** and the user assistance process linked from [../compute/README.md](../compute/README.md).
