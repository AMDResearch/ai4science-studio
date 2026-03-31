# GP-MoLFormer — Pair-Tuning Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

Fine-tune GP-MoLFormer toward a target molecular property using the pair-tuning approach — a lightweight RL-like optimization that adapts the model without full retraining.

## Prerequisites

- AMD Instinct GPU with ROCm 7.0+ driver
- Docker with AMD Container Toolkit
- Container set up via `examples/docker_run.sh` (includes applying `pairtune_training.patch`)

## What is pair-tuning?

Pair-tuning creates pairs of molecules — one with higher property score, one with lower — and trains the model to assign higher probability to the better molecule. This guides generation toward the target property without requiring reward model training.

## Run pair-tuning

```bash
docker exec gp-molformer bash /workspace/run_pairtune.sh
```

Default: optimize for QED (drug-likeness) for 100 epochs.

Customize with environment variables:

```bash
docker exec \
    -e PROPERTY=logp \
    -e NUM_EPOCHS=200 \
    -e BATCH_SIZE=1200 \
    gp-molformer bash /workspace/run_pairtune.sh
```

| Variable | Default | Options |
|---|---|---|
| `PROPERTY` | `qed` | `qed`, `logp`, `drd2` |
| `NUM_EPOCHS` | `100` | Any integer |
| `EVAL_EPOCHS` | `10` | Evaluation frequency |
| `BATCH_SIZE` | `1200` | Reduce if OOM |

## Available properties

| Property | Description |
|---|---|
| `qed` | Quantitative Estimate of Drug-likeness (0–1, higher is more drug-like) |
| `logp` | Penalized logP (octanol-water partition coefficient, penalized for ring count and SA score) |
| `drd2` | Predicted DRD2 dopamine receptor binding affinity |

## Output

Adapter weights are saved to `models/pairtune/<property>/`:

```
models/pairtune/qed/checkpoint-<N>/
    adapter_model.bin
    adapter_config.json
```

The adapter is a lightweight add-on to the base checkpoint — not a full model copy.

## Patch requirement

The `pairtune_training.patch` is required for pair-tuning to work with recent `transformers` versions. It is applied automatically by `docker_run.sh`. If you set up the container manually, apply it first:

```bash
cd /workspace/gp-molformer
git apply /workspace/pairtune_training.patch
```

The patch file is available in the [AMD blog silogen/ai-samples source](https://github.com/silogen/ai-samples).

## References

- [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/gp-molformer/README.html)
- [GP-MoLFormer paper](https://arxiv.org/abs/2302.07432)
- [IBM/gp-molformer GitHub](https://github.com/IBM/gp-molformer)
