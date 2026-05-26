# Model template (copy this folder)

Rename the parent folder from `_template` to your **model slug** (see the domain's `models/README.md` for naming rules).

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

## Acknowledgements and citation

Every model entry must include attribution. Fill in this section before opening a PR:

```
- **Upstream repo:** [org/repo](https://github.com/org/repo)
- **Paper:** Author et al., *Title*, Venue Year, https://doi.org/...  (or arXiv:XXXX.XXXXX)
- **Cite as:** paste BibTeX or DOI-based citation from the upstream repo/model card
- **ROCm blog (if applicable):** [Title](https://rocm.blogs.amd.com/...) — Author Name (AMD Silo AI)
- **Collaboration (if applicable):** e.g. AstraZeneca × AMD, ORNL × AMD
```

Also add an entry to [`ACKNOWLEDGEMENTS.md`](../../ACKNOWLEDGEMENTS.md) at the repo root following the existing per-model format.
