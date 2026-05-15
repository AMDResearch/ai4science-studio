# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

AI4Science Studio is an **agent-first** recipe collection for open AI-for-science models. The primary interface is an AI coding agent (Cursor, Claude Code, etc.) that reads machine-readable metadata to discover, configure, and run models. There is no build system, compiled code, or test suite—the repo is Markdown files, YAML manifests, and (optionally) scripts and notebooks in recipe folders. Upstream code lives in external GitHub repos; upstream model weights live on Hugging Face.

## Agent entry points (read these first)

- **`models.yaml`** (repo root) — index of all models; read this to discover what's available.
- **`<model>/model.yaml`** — per-model manifest with HF id, license, recipes, env vars, and hardware.
- **`.cursor/skills/`** — agent skills for Cursor; `.claude/commands/` — slash commands for Claude Code.

## Directory layout

```
models.yaml                                 # repo-wide model index (agent entry point)
<domain>/models/<model-slug>/model.yaml     # machine-readable model manifest
<domain>/models/<model-slug>/README.md      # HF id, license, upstream links
<domain>/models/<model-slug>/recipes/       # how-to docs, one subfolder per task
<domain>/models/<model-slug>/examples/      # ready-to-run scripts
```

**Domains:** `earth_science/`, `material_science/`, `protein_folding/`, `healthcare/`, `physics_simulation/`

## Model slug rule

Hugging Face id `org/model` → directory name `org__model` (replace `/` with `__`). Some models use a public name on disk instead; in that case, document the canonical HF id in the model's `README.md`. Example: `earth_science/models/ORBIT-2/` maps to HF id `jychoi-hpc/ORBIT-2`.

## Adding a new model

1. Pick the correct domain folder.
2. Copy `_template/` to `<domain>/models/<model-slug>/`.
3. Fill in `README.md`: Hugging Face model id (or `N/A` with alternate source), task, license (SPDX id or link), upstream code/paper.
4. Place how-to docs under `<model-slug>/recipes/`. Prefer one subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.), each with its own `README.md`.
5. Place ready-to-run scripts under `<model-slug>/examples/`: `docker_run.sh`, `run_<task>.sh`/`.py`, `preflight_<slug>.py`, and `sbatch_<task>_amd.sh`. All scripts must be `chmod +x`. For HPC models with heavy pip deps, also add `build_overlay_amd.sh`.
6. Create a `model.yaml` in the model folder with structured metadata (name, hf_id, license, task, recipes, env_vars).
7. Add the model to the root `models.yaml` index.
8. Do not commit large checkpoints or datasets—document how to obtain them instead.

## Conventions

- **Link, don't vendor.** Prefer linking Hugging Face model cards and upstream GitHub repos rather than copying large codebases into this repo.
- **Non-HF weights.** If weights are not on Hugging Face (e.g. GCS bucket, Google Drive, GitHub releases), set the HF id field to `N/A` and add an "Obtaining model weights" section with a fetch snippet.
- **AMD/ROCm notes** are optional and go inside individual recipes only where a maintainer has actually validated them. They are not a substitute for upstream documentation.
- **Example scripts** follow a standard pattern: `docker_run.sh` auto-detects AMD Container Toolkit vs device passthrough and checks for an existing container; `run_*.sh`/`.py` expose all key params as env vars with defaults; SLURM scripts use `--rocm` (not `--nv`) for AMD/Apptainer.
- **Healthcare & Life Sciences (HCLS) recipes** must include a research/engineering-only disclaimer and must not reference patient-identifiable data or PHI.
- Large artifacts (checkpoints, datasets, `.env` files) are in `.gitignore`; do not add them.

## Organic lesson capture and propagation

Every bug fix, workaround, or pattern discovery is a lesson. Do not treat skill/rule/recipe updates as a separate task — fold them into the fix itself.

When you fix anything in a model's scripts, do **all** of the following in the same pass:

1. **Fix the immediate script** that failed.
2. **Propagate to sibling models.** Scan `*/models/*/examples/` for the same pattern and fix them now.
3. **Propagate across runtimes.** If the fix was in an Apptainer script, check the Docker equivalent (and vice versa). Common cross-runtime issues: `SCRIPT_DIR` resolution, read-only FS handling, dep lists/version pins, env-var clobbering, torch protection after `pip install`.
4. **Update the relevant skill/doc.** Add the lesson to the right file:
   - Repo-wide patterns → `.cursor/skills/ai4science-studio/SKILL.md`
   - Domain-specific patterns → `.cursor/skills/ai4science-<domain>/SKILL.md`
5. **Create or update a rule** (`.cursor/rules/`) if the lesson is a recurring process mistake, not just a one-off technical fix.

**Litmus test:** would a fresh agent session working on a different model make the same mistake? If yes, the fix is incomplete — the lesson must be discoverable in a skill, rule, or `CLAUDE.md` before the task is done.

## Agent Skills

- **Cursor**: `.cursor/skills/` — domain conventions (earth science, healthcare, material science, protein folding, physics simulation, run-models, discover)
- **Claude Code**: `.claude/commands/` — slash commands for add-model, add-recipe, check-model, list-models, audit-models, init-cluster, and run-* for each model
