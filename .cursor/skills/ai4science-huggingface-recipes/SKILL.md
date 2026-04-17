---
name: ai4science-huggingface-recipes
description: Guides creation and editing of Hugging Face–centric recipes in AI4Science Studio—discovery, task alignment, scripting, documentation, and attribution.
---

# Hugging Face recipes workflow

Use this skill when adding or refactoring **train / fine-tune / inference / eval** material for models hosted on Hugging Face inside this repo.

## Steps

1. **Identify the model** on Hugging Face (model card, license, intended use, dependencies). Record the full id `org/model` in the model folder `README.md`.
2. **Map to a folder name** under the correct domain’s `models/` directory. **Default:** replace `/` with `__` in the Hub id (`org__model`). **Alternative:** a **public model name** on disk (e.g. `HydraGNN`, `ORBIT-2`) is acceptable when that domain’s `models/README.md` allows it and the model `README.md` states the full **`org/model`** id at the top—do not rename those folders to `org__model` unless the user asks.
3. **Align the task** with what the model card and upstream code support (inference-only vs fine-tuning vs training from scratch).
4. **Reuse upstream** scripts or minimal wrappers: prefer calling official examples with pinned versions rather than reimplementing full training stacks unless necessary.
5. **Document** Python/PyTorch (or other) versions, **ROCm** builds and versions for **AMD Instinct** when validated, and how to run from the repo root or from `recipes/` subfolders.
6. **Attribute** authors, papers, and license in the model `README.md` and in recipe comments where helpful.

## Recipe folder tips

- Use subfolders under `recipes/` for distinct tasks, e.g. `recipes/inference/`, `recipes/finetune/`.
- Keep entrypoints **small** and **documented**; link to Hugging Face Spaces or Colab only as supplements, not replacements for reproducible commands.

## Avoid

- Committing `token` values or private Hub tokens.
- Checking in large `*.bin`, `*.safetensors`, or full datasets when `.gitignore` already excludes them—point users to Hub or documented download steps instead.
- **Hardcoding HF repo filenames** without verifying: model repos ship different names than recipes assume (e.g. `walrus.pt` not `model.pt`). Use `list_repo_files()` to discover actual names, filter by extension, pick smallest or first match.
- Assuming a `_mi300x.sh` SLURM script name — use `_amd.sh` (covers MI250X, MI300X, MI350X with the same `rocm7.2.x` image).

## HF checkpoint download pattern (robust)

```python
from huggingface_hub import list_repo_files, hf_hub_download
import glob, os

repo = "org/model"
cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "model")
os.makedirs(cache, exist_ok=True)

# Reuse any existing download
existing = glob.glob(os.path.join(cache, "**/*.ckpt"), recursive=True)
if existing:
    chosen = sorted(existing)[0]
else:
    files = [f for f in list_repo_files(repo) if f.endswith(".ckpt")]
    chosen_name = sorted(files)[0]  # or filter by size keyword
    chosen = hf_hub_download(repo_id=repo, filename=chosen_name, local_dir=cache)
print(chosen)
```
