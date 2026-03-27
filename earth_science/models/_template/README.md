# Model template (copy this folder)

Rename the parent folder from `_template` to your **model slug** (see [`../README.md`](../README.md)).

## Fields to fill in

| Field | Example |
|-------|---------|
| **Hugging Face model id** | `org/model-name` |
| **Task** | e.g. downscaling, segmentation, forecasting |
| **License** | SPDX id or link to model card |
| **Upstream code / paper** | GitHub, arXiv, etc. |

## Recipes

Put training, fine-tuning, inference, or evaluation scripts (or step-by-step docs) under `recipes/`. Prefer one subfolder per task, e.g. `recipes/inference/`, `recipes/finetune/`.

Do not commit large checkpoints or datasets; document how to obtain them instead.
