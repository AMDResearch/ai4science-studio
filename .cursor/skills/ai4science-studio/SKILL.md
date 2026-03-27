---
name: ai4science-studio
description: Applies when working in the AI4Science Studio repository. Describes domain layout, model slug rules, where recipes live, and safety expectations for open science recipes.
---

# AI4Science Studio (repository)

## Repository map

- **Domains:** `earth_science/` (includes climate and weather), `material_science/`, `protein_folding/`, `healthcare/`.
- **Models:** `<domain>/models/<model-slug>/` with a `README.md` per model and `recipes/` for that model only.
- **Index:** Root [`README.md`](../../../README.md) and [`recipes/README.md`](../../../recipes/README.md).

## Model slug rule

Hugging Face id `org/model` → directory name `org__model` (replace `/` with double underscore). Document the canonical HF id in the model’s `README.md`.

## Conventions

- Prefer **linking** Hugging Face model cards and upstream GitHub repos instead of vendoring large codebases.
- Do **not** add secrets, API keys, `.env` files with credentials, or PHI to the repo.
- Large artifacts (checkpoints, datasets) belong in `.gitignore` patterns; recipes should explain how to obtain or generate them.

## When adding content

1. Pick the correct **domain** folder.
2. Create or update `models/<model-slug>/README.md` (license, HF id, upstream).
3. Place scripts or runbooks under `models/<model-slug>/recipes/`.
4. Copy structure from [`earth_science/models/_template/`](../../../earth_science/models/_template/) when starting a new model folder.
5. For **HPC-oriented** models, consider `recipes/local-cluster-amd.md` (institutional **AMD Instinct** + SLURM/PBS-style notes) and **`data-access.md`** sections on **staging** data (public **Globus** or **Hugging Face CLI** vs copy from **OLCF**/collaborator when public endpoints are not usable).
6. Write recipes around **AMD Instinct** with **PyTorch ROCm** when that matches what upstream documents for supported paths (`gpu_type: "amd"` where configs expose it). Point readers to upstream for install options this repo does not spell out.

