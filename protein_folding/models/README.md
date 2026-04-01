# Models (Protein folding)

> **Coming soon.** No models have been added to this domain yet. Contributions welcome — see the [repo README](../../README.md) and [`../../_template/`](../../_template/) for how to add one.

Each subdirectory under `models/` is one **Hugging Face model** (or family) for protein structure, folding, protein language models, etc.

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore).

## Adding a new model

1. Create `models/<your-slug>/README.md` with HF model id, license, and upstream links (see [`../../_template/`](../../_template/) for layout).
2. Add recipes under `models/<your-slug>/recipes/`.

Respect dataset and model licenses; note any restrictions on redistribution or commercial use in the model `README.md`.
