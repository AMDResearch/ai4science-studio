# GP-MoLFormer

> **Research / engineering use only.** This recipe is intended for computational chemistry and drug discovery research workflows. It does not constitute medical advice and must not be used with patient-identifiable data or PHI.

**Hugging Face:** [`ibm-research/GP-MoLFormer-Uniq`](https://huggingface.co/ibm-research/GP-MoLFormer-Uniq)
**Upstream code:** [`IBM/gp-molformer`](https://github.com/IBM/gp-molformer)
**Paper:** [Generative Pre-trained Transformer for De Novo Drug Design and Molecular Property Optimization](https://arxiv.org/abs/2302.07432)
**License:** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)

## What it does

GP-MoLFormer is a **decoder-only transformer for SMILES-based molecular generation** developed by IBM Research. Pre-trained on 1.1 billion SMILES strings from PubChem and ZINC, it supports:

| Mode | Description |
|---|---|
| Unconditional generation | Sample novel drug-like molecules from the learned SMILES distribution |
| Scaffold-constrained generation | Complete molecules around a fixed SMILES fragment (scaffold) |
| Pair-tuning (fine-tuning) | Fine-tune with RL-like paired optimization toward a target molecular property |

**Architecture:** Decoder-only transformer with Linear Attention and Rotary Positional Embeddings (RoPE).

**Available properties for pair-tuning:** QED (drug-likeness), penalized logP, DRD2 binding affinity.

## AMD / ROCm notes

Validated on **AMD Instinct MI300X** with ROCm 7.0 and PyTorch 2.7.1.

- Runs with the AMD Container Toolkit (`--runtime=amd`)
- The optional `fast_transformers` dependency is unavailable on ROCm — the code automatically falls back to the standard PyTorch transformer, which works correctly
- A `pairtune_training.patch` is required for pair-tuning (fixes compatibility with newer `transformers` versions); applied automatically by `docker_run.sh`

## Recipes

| Recipe | Summary |
|---|---|
| [`recipes/inference/`](recipes/inference/) | Unconditional and scaffold-constrained molecule generation |
| [`recipes/finetune/`](recipes/finetune/) | Pair-tuning toward a target molecular property (QED, logP, DRD2) |

## Quick start

```bash
# 1. Launch the container (clones repo and applies patch automatically)
bash examples/docker_run.sh

# 2. Inside the container — generate 1000 molecules
bash /examples/run_generation.sh

# 3. Or scaffold-constrained generation around a fragment
SCAFFOLD="c1cccc" bash /examples/run_generation.sh

# 4. Or pair-tune toward drug-likeness (QED)
bash /examples/run_pairtune.sh
```

## References

- [AMD ROCm blog — GP-MoLFormer on AMD MI300X](https://rocm.blogs.amd.com/artificial-intelligence/gp-molformer/README.html)
- [GP-MoLFormer paper](https://arxiv.org/abs/2302.07432)
- [Hugging Face model card](https://huggingface.co/ibm-research/GP-MoLFormer-Uniq)
