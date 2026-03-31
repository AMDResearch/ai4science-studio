# GP-MoLFormer — Inference Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

Generate novel drug-like molecules unconditionally or constrained to a scaffold fragment.

## Prerequisites

- AMD Instinct GPU with ROCm 7.0+ driver
- Docker with AMD Container Toolkit (or bare-device passthrough)

## Setup

```bash
bash examples/docker_run.sh
```

This script will:
1. Pull `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1`
2. Clone `IBM/gp-molformer` inside the container
3. Apply `pairtune_training.patch` (if present in `examples/`)
4. Install Python dependencies

> The `fast_transformers` dependency is unavailable on ROCm — the code falls back to standard PyTorch transformers automatically, with no user action needed.

## Unconditional generation

```bash
docker exec gp-molformer bash /workspace/run_generation.sh
```

Generates 1000 SMILES strings and writes them to `/workspace/generated.csv`. The script also prints validity and uniqueness stats.

To generate more:

```bash
docker exec -e NUM_BATCHES=5 gp-molformer bash /workspace/run_generation.sh
```

## Scaffold-constrained generation

Provide a SMILES fragment; the model completes full molecules preserving that scaffold:

```bash
docker exec -e SCAFFOLD="c1cccc" gp-molformer bash /workspace/run_generation.sh
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
