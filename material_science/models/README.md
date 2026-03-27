# Models (Material science)

Each subdirectory under `models/` is one **Hugging Face model** (or family) for material science, chemistry, surrogates for simulation, etc.

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore):

| Hugging Face id | Folder name |
|-----------------|-------------|
| `org/crystal-property-model` | `org__crystal-property-model` |

## Adding a new model

1. Create `models/<your-slug>/README.md` with HF model id, license, and upstream links (use [`../earth_science/models/_template/`](../../earth_science/models/_template/) as a structural reference, or duplicate that folder here).
2. Add recipes under `models/<your-slug>/recipes/`.
