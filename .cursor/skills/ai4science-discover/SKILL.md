---
name: ai4science-discover
description: How to answer user questions about what models are available, filter by domain/task/license, and compare models in AI4Science Studio.
---

# Discovering models in AI4Science Studio

Use this skill when a user asks what models are available, wants to compare models, or asks questions like "what can I do with this repo?"

## Quick discovery

Read `models.yaml` at the repo root. It lists every model with:
- `slug` — folder name
- `domain` — earth_science, material_science, healthcare, physics_simulation, protein_folding
- `hf_id` — Hugging Face model id (or N/A)
- `license` — SPDX id
- `task` — one-line description
- `tasks_available` — list of recipe tasks (inference, train, finetune, ensemble)
- `path` — relative path to model folder

## Answering "what models are available?"

Parse `models.yaml` and present a table:

| Model | Domain | Task | Available recipes |
|-------|--------|------|-------------------|
| StormCast | Earth science | Weather prediction | inference, ensemble |
| ... | ... | ... | ... |

## Filtering

When the user asks for a subset:
- **By domain:** filter on the `domain` field
- **By task type:** filter on `tasks_available` (e.g. "which models support fine-tuning?" → filter for `finetune` in `tasks_available`)
- **By license:** filter on `license` (e.g. "which models are MIT licensed?")
- **By HF availability:** filter on `hf_id != N/A` for models with HF weights

## Deep-dive on a model

When the user wants details on a specific model:
1. Read `<path>/model.yaml` for structured metadata
2. Read `<path>/README.md` for prose description
3. Read `<path>/examples/README.md` for available scripts
4. List `<path>/recipes/` subdirectories for available recipes

## Comparing models

When the user asks to compare two or more models:
1. Read each model's `model.yaml`
2. Present a comparison table with columns: name, domain, HF id, license, tasks, container image, validated hardware, VRAM

## Checking readiness

When asked "which models are ready to run?":
1. For each model in `models.yaml`, check:
   - Does `examples/docker_run.sh` exist?
   - Does `examples/preflight_*.py` exist?
   - Does at least one `recipes/*/README.md` exist?
   - Does `examples/sbatch_*_amd.sh` exist?
2. Report a readiness checklist per model
