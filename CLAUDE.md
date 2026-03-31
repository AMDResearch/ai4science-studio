# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

AI4Science Studio is a documentation-first recipe collection for open AI-for-science models. There is no build system, compiled code, or test suite—the repo is entirely Markdown files and (optionally) scripts and notebooks in recipe folders. Upstream code lives in external GitHub repos; upstream model weights live on Hugging Face.

## Directory layout

```
<domain>/models/<model-slug>/README.md    # HF id, license, upstream links
<domain>/models/<model-slug>/recipes/     # scripts, notebooks, how-to docs
```

**Domains:** `earth_science/`, `material_science/`, `protein_folding/`, `healthcare/`

## Model slug rule

Hugging Face id `org/model` → directory name `org__model` (replace `/` with `__`). Some models use a public name on disk instead; in that case, document the canonical HF id in the model's `README.md`. Example: `earth_science/models/ORBIT-2/` maps to HF id `jychoi-hpc/ORBIT-2`.

## Adding a new model

1. Pick the correct domain folder.
2. Copy `earth_science/models/_template/` to `<domain>/models/<model-slug>/`.
3. Fill in `README.md`: Hugging Face model id, task, license (SPDX id or link), upstream code/paper.
4. Place scripts or runbooks under `<model-slug>/recipes/`. Prefer one subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.).
5. Do not commit large checkpoints or datasets—document how to obtain them instead.

## Conventions

- **Link, don't vendor.** Prefer linking Hugging Face model cards and upstream GitHub repos rather than copying large codebases into this repo.
- **AMD/ROCm notes** are optional and go inside individual recipes only where a maintainer has actually validated them. They are not a substitute for upstream documentation.
- **Healthcare recipes** must include a research/engineering-only disclaimer and must not reference patient-identifiable data or PHI.
- Large artifacts (checkpoints, datasets, `.env` files) are in `.gitignore`; do not add them.

## Cursor Agent Skills

Domain-specific conventions for AI coding agents are defined in `.cursor/skills/`. Each `SKILL.md` covers one scope: general repo rules (`ai4science-studio`), Hugging Face recipe workflows (`ai4science-huggingface-recipes`), and per-domain cautions for earth science, healthcare, material science, and protein folding.

## Claude Code Skills

Slash commands for Claude Code live in `.claude/commands/`:

| Command | Purpose |
|---|---|
| `/add-model` | Walk through adding a new HF model (slug, folder, README, recipes) |
| `/add-recipe` | Add or improve a recipe for an existing model |
| `/check-model` | Audit a model folder for completeness and convention compliance |

Invoke with an argument, e.g. `/add-model jychoi-hpc/ORBIT-2 → earth_science`.
