# GP-MoLFormer examples

Ready-to-run scripts for [GP-MoLFormer](../README.md). These wrap the upstream
[`IBM/gp-molformer`](https://github.com/IBM/gp-molformer) with AMD/ROCm
container launch and SLURM/Apptainer support.

> **Research / engineering use only.** Not for clinical or diagnostic use.

## Files

| Script | Purpose |
|--------|---------|
| [`preflight_gpmolformer.py`](preflight_gpmolformer.py) | Check your environment before submitting jobs |
| [`run_generation.sh`](run_generation.sh) | Unconditional or scaffold-constrained molecule generation |
| [`run_pairtune.sh`](run_pairtune.sh) | Pair-tuning for property optimization |
| [`docker_run.sh`](docker_run.sh) | Docker launcher (clones gp-molformer, installs deps) |
| [`sbatch_inference_amd.sh`](sbatch_inference_amd.sh) | SLURM driver for generation on AMD Instinct |

## Quick start — Docker

```bash
# Set up the container
./docker_run.sh

# Unconditional generation (1000 molecules)
docker exec gp-molformer bash /workspace/run_generation.sh

# Scaffold-constrained generation
docker exec -e SCAFFOLD="c1ccccc1" gp-molformer bash /workspace/run_generation.sh
```

## Quick start — SLURM

```bash
export GPMOL_SIF=/path/to/rocm_pytorch.sif
sbatch sbatch_inference_amd.sh                        # unconditional
SCAFFOLD="c1ccccc1" sbatch sbatch_inference_amd.sh    # scaffold mode
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GPMOL_SIF` | Yes (Apptainer) | -- | Path to Apptainer SIF file |
| `SCAFFOLD` | No | -- | SMILES fragment (empty = unconditional) |
| `NUM_BATCHES` | No | `1` | Batches of 1000 molecules |
| `OUTPUT_FILE` | No | `/workspace/generated.csv` | Output CSV path |
