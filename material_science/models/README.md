# Models (Material science)

Each subdirectory under `models/` is one **Hugging Face model** (or family) for material science, chemistry, surrogates for simulation, etc.

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore):

| Hugging Face id | Folder name |
|-----------------|-------------|
| `org/crystal-property-model` | `org__crystal-property-model` |

Some folders use a **public model name** on disk (e.g. `HydraGNN`) instead of `org__model`. In those cases the **canonical Hugging Face id** must be stated clearly at the top of that model’s `README.md`.

## Adding a new model

1. Create `models/<your-slug>/README.md` with HF model id, license, and upstream links (use [`../../_template/`](../../_template/) as a structural reference, or duplicate that folder here).
2. Add recipes under `models/<your-slug>/recipes/`.
