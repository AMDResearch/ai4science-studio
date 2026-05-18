# GP-MoLFormer — Inference Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

Generate novel drug-like molecules unconditionally or constrained to a scaffold fragment.

## Prerequisites

- Container: `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1`
- GPU: AMD Instinct with ROCm 7.0+ driver (covers MI250X, MI300X, MI350X)
- Runtime: Apptainer (SLURM clusters) or Docker with AMD Container Toolkit
- Weights: auto-downloaded from Hugging Face on first run

> The `fast_transformers` dependency is unavailable on ROCm — the code falls back to standard PyTorch transformers automatically, with no user action needed.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GPMOL_SIF` | Yes (Apptainer) | -- | Path to Apptainer SIF file |
| `GPMOL_WORK_DIR` | No | examples dir | Host dir bound to `/workspace` |
| `SCAFFOLD` | No | -- | SMILES fragment for constrained generation (empty = unconditional) |
| `NUM_BATCHES` | No | `1` | Batches of 1000 molecules |
| `OUTPUT_FILE` | No | `/workspace/generated.csv` | Output CSV path |

## Option A — SLURM cluster (Apptainer)

```bash
export GPMOL_SIF=${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
sbatch examples/sbatch_inference_amd.sh                          # unconditional
SCAFFOLD="c1ccccc1" sbatch examples/sbatch_inference_amd.sh      # scaffold mode
```

On first run the script clones `IBM/gp-molformer` and installs deps inside the container (requires internet access from compute nodes). Subsequent runs reuse the clone from `GPMOL_WORK_DIR`.

Key env vars: `GPMOL_SIF` (required), `GPMOL_WORK_DIR` (default: examples dir), `SCAFFOLD`, `NUM_BATCHES` (default: 1 = 1000 molecules), `OUTPUT_FILE`.

## Option B — Docker interactive (single node)

### Setup

```bash
bash examples/docker_run.sh
```

This script will:
1. Pull `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1`
2. Clone `IBM/gp-molformer` inside the container
3. Apply `pairtune_training.patch` (if present in `examples/`)
4. Install Python dependencies

## Unconditional generation

```bash
# Docker
docker exec gp-molformer bash /workspace/run_generation.sh

# Apptainer (SLURM)
sbatch examples/sbatch_inference_amd.sh
```

Generates 1000 SMILES strings and writes them to `generated.csv`. The script also prints validity and uniqueness stats.

To generate more:

```bash
docker exec -e NUM_BATCHES=5 gp-molformer bash /workspace/run_generation.sh
# or
NUM_BATCHES=5 sbatch examples/sbatch_inference_amd.sh
```

## Scaffold-constrained generation

Provide a SMILES fragment; the model completes full molecules preserving that scaffold:

```bash
docker exec -e SCAFFOLD="c1cccc" gp-molformer bash /workspace/run_generation.sh
# or
SCAFFOLD="c1cccc" sbatch examples/sbatch_inference_amd.sh
```

Any valid SMILES fragment works as a scaffold, e.g.:
- `c1cccc` — benzene ring fragment
- `C1CCCCC1` — cyclohexane
- `c1ccc(cc1)` — para-substituted benzene

## Output format

The output CSV contains one canonical SMILES per row. Read with pandas or RDKit:

```python
import pandas as pd
from rdkit import Chem

df = pd.read_csv("generated.csv", header=None, names=["smiles"])
df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
df_valid = df.dropna(subset=["mol"])
print(f"Valid: {len(df_valid)} / {len(df)}")
```

## Reference benchmark (MI300X)

| Task | Output |
|---|---|
| Unconditional (1 batch = 1000 molecules) | ~649 unique valid SMILES |
| Scaffold `c1cccc` (1 batch) | ~653 valid molecules |

## References

- [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/gp-molformer/README.html)
- [GP-MoLFormer paper](https://arxiv.org/abs/2302.07432)
- [Hugging Face model card](https://huggingface.co/ibm-research/GP-MoLFormer-Uniq)
