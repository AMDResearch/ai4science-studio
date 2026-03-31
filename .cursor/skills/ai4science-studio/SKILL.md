---
name: ai4science-studio
description: Applies when working in the AI4Science Studio repository. Describes domain layout, model slug rules, where recipes live, and safety expectations for open science recipes.
---

# AI4Science Studio (repository)

## Repository map

- **Domains:** `earth_science/` (includes climate and weather), `material_science/`, `protein_folding/`, `healthcare/`.
- **Models:** `<domain>/models/<model-slug>/` with a `README.md` per model and `recipes/` for that model only.
- **Index:** Root [`README.md`](../../../README.md).

## Model slug rule

Hugging Face id `org/model` → directory name `org__model` (replace `/` with double underscore). Document the canonical HF id in the model’s `README.md`.

## Conventions

- Prefer **linking** Hugging Face model cards and upstream GitHub repos instead of vendoring large codebases.
- Do **not** add secrets, API keys, `.env` files with credentials, or PHI to the repo.
- Large artifacts (checkpoints, datasets) belong in `.gitignore` patterns; recipes should explain how to obtain or generate them.

## When adding content

1. Pick the correct **domain** folder.
2. Create or update `models/<model-slug>/README.md` (license, HF id, upstream). If weights are not on Hugging Face (GCS, Google Drive, GitHub releases), set the HF id field to `N/A` and add an "Obtaining model weights" section with a fetch snippet.
3. Place runbooks under `models/<model-slug>/recipes/`. One subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.), each with its own `README.md` and a callout box at the top linking to `examples/`.
4. Place ready-to-run scripts under `models/<model-slug>/examples/`:
   - `docker_run.sh` — auto-detects AMD Container Toolkit (`docker info | grep -qi amd`) vs device passthrough (`/dev/kfd` + all `/dev/dri/renderD*`); checks for existing container and exits with attach hint; auto-clones upstream repo if absent.
   - `run_<task>.sh` / `run_<task>.py` — all key params overridable via env vars with sensible defaults; prints config summary before running; exits with clear error when required inputs are missing.
   - `preflight_<slug>.py` — smoke-test verifying GPU access and imports.
   - `sbatch_<task>_mi300x.sh` — SLURM batch script; use `--rocm` (not `--nv`) for AMD/Apptainer GPU passthrough.
   - All scripts must be `chmod +x`.
5. Copy structure from [`_template/`](../../../_template/) when starting a new model folder.
6. For **HPC-oriented** models, consider `recipes/local-cluster-amd.md` (institutional **AMD Instinct** + SLURM/PBS-style notes) and **`data-access.md`** sections on **staging** data (public **Globus** or **Hugging Face CLI** vs copy from **OLCF**/collaborator when public endpoints are not usable).
7. Write recipes around **AMD Instinct** with **PyTorch ROCm** when that matches what upstream documents for supported paths (`gpu_type: "amd"` where configs expose it). Point readers to upstream for install options this repo does not spell out.

