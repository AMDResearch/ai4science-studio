# Add a new model to AI4Science Studio

Add a new Hugging Face model to this repository following all repo conventions.

## Steps

1. **Identify domain** — pick from `earth_science/`, `material_science/`, `protein_folding/`, or `healthcare/`.

2. **Derive the slug**:
   - HF id `org/model` → directory name `org__model` (replace `/` with `__`).
   - If a well-known public name is used instead, document the canonical HF id inside the model's `README.md`.

3. **Create the model folder** by copying `earth_science/models/_template/` to `<domain>/models/<slug>/`.

4. **Fill in `README.md`** with:
   - Hugging Face model id
   - Task description
   - License (SPDX id or direct link)
   - Upstream code repo and paper links

5. **Add recipes** under `<slug>/recipes/`. Use one subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.).

6. **Domain-specific checks**:
   - `earth_science/`: state spatial/temporal resolution, coordinate conventions, data sources (ERA5, satellite products, etc.).
   - `material_science/`: state input representations (graphs, SMILES, crystals) and unit conventions.
   - `protein_folding/`: surface license restrictions; avoid implying clinical/diagnostic use.
   - `healthcare/`: add research/engineering-only disclaimer; no PHI; copy intended-use and limitations from the model card.

7. **Do not commit** large checkpoints, datasets, `.env` files, or tokens — document how users obtain them instead.

## Model to add

$ARGUMENTS
