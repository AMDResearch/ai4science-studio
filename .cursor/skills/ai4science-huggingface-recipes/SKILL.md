---
name: ai4science-huggingface-recipes
description: Guides creation and editing of Hugging Face–centric recipes in AI4Science Studio—discovery, task alignment, scripting, documentation, and attribution.
---

# Hugging Face recipes workflow

Use this skill when adding or refactoring **train / fine-tune / inference / eval** material for models hosted on Hugging Face inside this repo.

## Steps

1. **Identify the model** on Hugging Face (model card, license, intended use, dependencies). Record the full id `org/model` in the model folder `README.md`.
2. **Map to a slug** for the filesystem: `org__model` under the correct domain’s `models/` directory.
3. **Align the task** with what the model card and upstream code support (inference-only vs fine-tuning vs training from scratch).
4. **Reuse upstream** scripts or minimal wrappers: prefer calling official examples with pinned versions rather than reimplementing full training stacks unless necessary.
5. **Document** Python/PyTorch (or other) versions, optional ROCm/CUDA notes if validated, and how to run from the repo root or from `recipes/` subfolders.
6. **Attribute** authors, papers, and license in the model `README.md` and in recipe comments where helpful.

## Recipe folder tips

- Use subfolders under `recipes/` for distinct tasks, e.g. `recipes/inference/`, `recipes/finetune/`.
- Keep entrypoints **small** and **documented**; link to Hugging Face Spaces or Colab only as supplements, not replacements for reproducible commands.

## Avoid

- Committing `token` values or private Hub tokens.
- Checking in large `*.bin`, `*.safetensors`, or full datasets when `.gitignore` already excludes them—point users to Hub or documented download steps instead.
