# Run ORBIT-2 inference/visualization on an AMD cluster

Guide the user through running ORBIT-2 end-to-end: upstream repo clone, SIF setup, overlay build, synthetic data smoke-test, and SLURM job submission. Supports 1-GPU and multi-GPU distributed runs.

## Step 1 — Clone the upstream repo

ORBIT-2 code is not vendored in this Studio repo. Check if `ORBIT2_ROOT` points to an existing clone:

```bash
git clone https://github.com/XiaoWang-Github/ORBIT-2.git /path/to/ORBIT-2
export ORBIT2_ROOT=/path/to/ORBIT-2
```

## Step 2 — SIF image

Ask the user for `ORBIT2_SIF`. If not set or file doesn't exist:

```bash
apptainer pull docker://rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
export ORBIT2_SIF=/path/to/rocm_pytorch_rocm7.2.2_pt2.10_u24.sif
```

## Step 3 — Overlay (recommended, one-time ~15 min)

Ask if the user has a pre-built overlay at `ORBIT2_OVERLAY`. If not, offer to build it:

```bash
export ORBIT2_SIF=<path>
sbatch earth_science/models/ORBIT-2/examples/build_overlay_amd.sh
# Wait for completion; output path printed in build log (default: orbit2-overlay.img next to SIF)
export ORBIT2_OVERLAY=<path>
```

Without an overlay each job runs a ~15 min pip install (pytorch-lightning, xformers, mpi4py, wandb, etc.).

## Step 4 — Data mode

Ask the user which data mode to use:

**A) Synthetic data (recommended for smoke-testing — no real data needed):**
```bash
export ORBIT2_USE_SYNTHETIC=1
# Checkpoint is downloaded automatically from jychoi-hpc/ORBIT-2 on HuggingFace if ORBIT2_CHECKPOINT is unset
```

**B) Real data (ERA5/PRISM):**
```bash
export ORBIT2_CONFIG=interm_8m_ft.yaml          # or another config YAML
export ORBIT2_CHECKPOINT=/path/to/model.ckpt
# Ensure low_res_dir / high_res_dir in the YAML point to staged data
```

Note: real data requires ERA5/PRISM staged on the cluster — this is a known blocker on many systems.

## Step 5 — GPU count

Ask the user: 1 GPU (quick smoke-test) or 8 GPU (full distributed)?

**1 GPU** — edit the SBATCH header in `sbatch_infer_amd.sh`:
```
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
```

**8 GPU** — use the defaults in the script (`--gres=gpu:8 --ntasks-per-node=8`).

Multi-node: increase `--nodes` and `--ntasks` proportionally; the `srun` command is unchanged.

## Step 6 — Partition and account

Remind the user to set `--partition` and `--account` in the SBATCH header (placeholders: `YOUR_PARTITION_HERE` / `YOUR_PROJECT_HERE`).

## Submission

```bash
export ORBIT2_ROOT=<path>
export ORBIT2_SIF=<path>
export ORBIT2_OVERLAY=<path>          # omit if no overlay
export ORBIT2_USE_SYNTHETIC=1         # or set ORBIT2_CONFIG + ORBIT2_CHECKPOINT
sbatch earth_science/models/ORBIT-2/examples/sbatch_infer_amd.sh
```

## Monitoring and output

- `squeue -j <job_id>` to check status
- `tail -f orbit2-vis-<job_id>.out` to follow the log
- On success: look for `PSNR` and `SSIM` metrics in the log, plus visualization outputs in the working directory

## Expected results (synthetic data smoke-test)

| Config | GPUs | Wall time | PSNR | SSIM |
|---|---|---|---|---|
| 1-GPU synthetic | 1 | ~1–2 min | 14.35 | 0.039 |
| 8-GPU synthetic | 8 | ~2 min | 14.35 | 0.039 |

## Arguments

$ARGUMENTS
