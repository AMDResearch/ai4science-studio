# Models (Earth science)

Each subdirectory under `models/` is one **Hugging Face model** (or a closely related model family) used for Earth-system ML—including **climate**, **weather**, and broader **geospatial / remote sensing** work.

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore):

| Hugging Face id | Folder name |
|-----------------|-------------|
| `microsoft/phi-2` | `microsoft__phi-2` |
| `my-org/my-climate-model` | `my-org__my-climate-model` |

Use lowercase unless the org name on Hugging Face is case-sensitive; match the Hub id when in doubt.

Some folders use a **public model name** on disk (e.g. `ORBIT-2`) instead of `org__model`. In those cases the **canonical Hugging Face id** must be stated clearly at the top of that model’s `README.md`.

## Adding a new model

1. Copy [`../../_template/`](../../_template/) to `models/<your-folder>/` (or pick a public model name; document the Hub id in `README.md`).
2. Edit `README.md` with the real HF model id, license, and links.
3. Add recipes under `models/<your-folder>/recipes/`.

## Template

- [`../../_template/`](../../_template/) — copy-paste starting point with stub `README.md` and empty `recipes/`.
