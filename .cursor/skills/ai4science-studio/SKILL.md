---
name: ai4science-studio
description: Applies when working in the AI4Science Studio repository. Use for domain layout, model slug rules, make check, and where AMD/ROCm HPC lessons live (reference/rocm.md).
---

# AI4Science Studio (repository)

## Repository map

- **Domains:** `earth_science/` (includes climate and weather), `material_science/`, `protein_folding/`, `healthcare/`, `physics_simulation/`.
- **Models:** `<domain>/models/<model-slug>/` with a `README.md` per model and `recipes/` for that model only.
- **Model index:** Root [`models.yaml`](../../../models.yaml) — **generated** list of all models. Read this first for discovery. Do not edit it by hand.
- **Per-model manifest:** `<domain>/models/<model-slug>/model.yaml` — source of truth (HF id, license, recipes, env vars, hardware). Schema: [`schemas/model.schema.json`](../../../schemas/model.schema.json).
- **Human index:** Root [`README.md`](../../../README.md).

## Model slug rule

Hugging Face id `org/model` → directory name `org__model` (replace `/` with double underscore). Document the canonical HF id in the model's `README.md`. Public on-disk names (e.g. `ORBIT-2`, `HydraGNN`) are acceptable when the domain's `models/README.md` allows it.

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
   - `preflight_<slug>.py` — smoke-test verifying GPU access and imports. Must accept `--dry-run` (skip GPU/imports; used by `make check` on CPU). `--dry-run` must print that it skipped those checks; keep the live GPU/import checks in the same file.
   - `sbatch_<task>_amd.sh` — SLURM + Apptainer batch script; use `--rocm` (not `--nv`) for AMD GPU passthrough.
   - `sbatch_<task>_docker.sh` — SLURM + Docker batch script; uses device passthrough or `--runtime=amd`.
   - `build_overlay_amd.sh` — (Apptainer only, HPC models with heavy pip deps) builds a persistent ext3 overlay. Run once per cluster; reuse with `--overlay <path>:ro`.
   - All scripts must be `chmod +x`.
5. Create a `model.yaml` in the model folder with structured metadata (name, hf_id, license, task, summary, recipes, env_vars). See existing `model.yaml` files for the schema. `summary` is the short one-liner used in the generated index.
6. Run `make fix` to regenerate root `models.yaml`, then `make check` (`bash -n`, error-severity `shellcheck`, `container_image` syntax — no image pull). Do not hand-edit the index.
7. Copy structure from [`_template/`](../../../_template/) when starting a new model folder.
8. For **HPC-oriented** models, consider `recipes/local-cluster-amd.md` and **`data-access.md`** sections on data staging.
9. Write recipes around **AMD Instinct** with **PyTorch ROCm** when that matches upstream supported paths.
10. **GPU arch naming:** Scripts are named `_amd.sh` / `_docker.sh`, not `_mi300x.sh`. The same `rocm7.2.x` image covers MI250X (gfx90a), MI300X (gfx942), and MI350X (gfx950). Only the `ROCM_WHL_TAG` env var and the image tag need to change for a different ROCm generation.

---

## HPC / ROCm details

Read [reference/rocm.md](reference/rocm.md) before writing or debugging `sbatch_*`, overlays, `--no-deps` installs, env-var clobbering, or Omnistat/TraceLens. Domain skills must **link** that file instead of recopying overlay recipes.

Cross-runtime rule: a fix in Apptainer must be checked in Docker (and vice versa). Capture the lesson in `reference/rocm.md` (repo-wide) or the domain `reference/` file (model-specific).
