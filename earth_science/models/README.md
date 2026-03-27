# Models (Earth science)

Each subdirectory under `models/` is one **Hugging Face model** (or a closely related model family) used for Earth-system ML—including **climate**, **weather**, and broader **geospatial / remote sensing** work.

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore):

| Hugging Face id | Folder name |
|-----------------|-------------|
| `microsoft/phi-2` | `microsoft__phi-2` |
| `my-org/my-climate-model` | `my-org__my-climate-model` |

Use lowercase unless the org name on Hugging Face is case-sensitive; match the Hub id when in doubt.

## Adding a new model

1. Copy `models/_template/` to `models/<your-slug>/`.
2. Edit `README.md` with the real HF model id, license, and links.
3. Add recipes under `models/<your-slug>/recipes/`.

## Template

- [`_template/`](_template/) — copy-paste starting point with stub `README.md` and empty `recipes/`.
