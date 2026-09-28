# ORBIT-2: inference and visualization

This is the most common **AI4Science Studio** entry point: use **published checkpoints** from Hugging Face with the **official code** to run visualization / inference-style workflows.

## Prerequisites

- Container: `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`
- Code: clone [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2) and follow its Installation section (AMD Instinct + ROCm path)
- Weights: download checkpoint(s) and matching YAML from [`jychoi-hpc/ORBIT-2`](https://huggingface.co/jychoi-hpc/ORBIT-2) (see the live file tree under `pretrain/`, `us-finetune/`, `global-finetune/`, etc.)

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ORBIT2_ROOT` | Yes | -- | Path to cloned ORBIT-2 repository |
| `ORBIT2_SIF` | No | -- | Apptainer SIF path (bare-metal if unset) |
| `ORBIT2_OVERLAY` | No | -- | Pre-built overlay (build with `build_overlay_amd.sh`) |
| `ORBIT2_CHECKPOINT` | No | -- | `.ckpt` path; auto-downloaded from HF in synthetic mode |
| `ORBIT2_CONFIG` | No | `interm_8m_ft.yaml` | Config YAML basename for real-data mode |
| `ORBIT2_USE_SYNTHETIC` | No | `0` | Set `1` for synthetic data smoke test |
| `ORBIT2_MASTER_PORT` | No | `29500` | Distributed master port |

## Point the code at a checkpoint

The upstream README describes two ways to select a checkpoint for visualization:

1. **Config file** — set `trainer.pretrain` (or the field upstream documents for your config) to the local path of the `.ckpt` file.
2. **CLI override** — pass `--checkpoint /path/to/model.ckpt` to `visualize.py` when that matches the version of the repo you use.

Always align the **YAML config** with the **checkpoint** (matching model scale and task). Upstream lists example pairs for US and global fine-tuned models; re-read the current GitHub README for exact names.

## Running visualization

- **Frontier (OLCF):** edit `examples/launch_visualize.sh` (allocation account, conda path, config path), then `sbatch launch_visualize.sh`. See the upstream "Tutorial Example → Frontier → Step 4: Visualize Results".
- **Other clusters / interactive nodes:** run the same `visualize.py` invocation the launch script uses, with your scheduler or `mpirun` layout as appropriate for that system.
- **Studio examples (AI4Science Studio checkout):** in [`../../examples/`](../../examples/), **`run_visualize.py`** runs upstream `visualize.py` with `ORBIT2_ROOT` set to your clone; **`make_synthetic_data.py`** generates a ~2 MB synthetic dataset so you can smoke-test the full pipeline without ERA5/PRISM data; **`preflight_orbit2.py`** checks the environment without SLURM; **`sbatch_infer_amd.sh`** is a SLURM job that works on any AMD Instinct partition (MI250X, MI300X, MI350X) and supports synthetic data via `ORBIT2_USE_SYNTHETIC=1`; **`build_overlay_amd.sh`** pre-bakes pip deps into a persistent ext3 overlay (~15 min once, skipped on every subsequent job); **`docker_run.sh`** launches a single-node container for interactive inference. See [../compute/README.md](../compute/README.md).

### Quick smoke-test (no real data needed)

```bash
export ORBIT2_ROOT=/path/to/ORBIT-2-clone
export ORBIT2_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
export ORBIT2_OVERLAY=/path/to/orbit2-overlay.img   # optional; build with build_overlay_amd.sh
export ORBIT2_USE_SYNTHETIC=1   # auto-generates data + downloads smallest HF checkpoint
sbatch ../../examples/sbatch_infer_amd.sh
```

Confirmed working: MI250X, MI300X, MI350X (rocm7.2.2 image). Covers 1-GPU and 8-GPU distributed runs via `srun --mpi=pmix`.

### ORNL Lux (OLCF, MI355X)

On **ORNL Lux**, use the same `sbatch_infer_amd.sh` flow with Lustre staging under `$AI4S_SHARED_DIR` (see `.cluster-config.yaml`). Differences from Vultr Lux:

- **No `/opt/ompi`** — the script sources HydraGNN's `lux_mpi_env.sh`, which loads Lux's system `openmpi/5.0.10` module and forces ob1/TCP. It always launches with `srun --mpi=pmix`; do **not** bypass MPI for single-GPU jobs.
- **Pre-stage on the login node** — SIF, overlay, ORBIT-2 clone, and HF checkpoint (`intermediate_8m.ckpt`) on Lustre; compute nodes have no egress.
- **Bind-mount Lustre weights** — when `ORBIT2_CHECKPOINT` points at `$AI4S_SHARED_DIR/models/ORBIT-2/weights/...`, the sbatch script bind-mounts the ORBIT-2 tree so the container sees the same absolute path.
- **Synthetic smoke** — `ORBIT2_USE_SYNTHETIC=1` with pre-downloaded checkpoint; expect low PSNR/SSIM on random data (pipeline validation only).

```bash
export AI4S_SHARED_DIR=/path/to/shared
export ORBIT2_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/code/ORBIT-2
export ORBIT2_SIF=$AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
export ORBIT2_OVERLAY=$AI4S_SHARED_DIR/models/ORBIT-2/overlays/orbit2-overlay.img
export ORBIT2_CHECKPOINT=$AI4S_SHARED_DIR/models/ORBIT-2/weights/pretrain/intermediate_8m.ckpt
export ORBIT2_USE_SYNTHETIC=1
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT --gres=gpu:1 --time=01:00:00 \
  --output=$AI4S_SHARED_DIR/models/ORBIT-2/logs/orbit2-vis-%j.out \
  earth_science/models/ORBIT-2/examples/sbatch_infer_amd.sh
```

Outputs typically include low-resolution inputs, high-resolution predictions, and comparison plots as described upstream.

## Inference latency (expectations)

The ORBIT-2 paper reports that **after training**, inference can be very fast relative to dynamical downscaling: for example, **milliseconds per global sample** for a small-parameter model and **sub-second** scale for a 10B-parameter model on an 8-GPU setup for a stated ERA5→ERA5 benchmark (see paper tables). Your hardware and batching will differ—treat these as **order-of-magnitude** guidance, not guarantees.

## See also

- [../data/README.md](../data/README.md) — if you need paired low/high-res data beyond bundled demos.
- [../compute/README.md](../compute/README.md) — if you lack Frontier-scale resources.
