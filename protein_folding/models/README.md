# Models (Protein folding)

Each subdirectory under `models/` is one **Hugging Face model** (or family) for protein structure, folding, protein language models, etc.

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore).

## Adding a new model

1. Create `models/<your-slug>/README.md` with HF model id, license, and upstream links (see [`../earth_science/models/_template/`](../../earth_science/models/_template/) for layout).
2. Add recipes under `models/<your-slug>/recipes/`.

Respect dataset and model licenses; note any restrictions on redistribution or commercial use in the model `README.md`.
