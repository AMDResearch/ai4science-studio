# Models (Healthcare & Life Sciences)

Each subdirectory under `models/` is one **Hugging Face model** (or family) for healthcare and life sciences ML (imaging, clinical text, genomics helpers, drug discovery, molecular design, etc.).

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore).

## Adding a new model

1. Create `models/<your-slug>/README.md` with HF model id, license, intended use, and upstream links (see [`../../_template/`](../../_template/) for layout).
2. Add recipes under `models/<your-slug>/recipes/`.

Do not commit protected health information (PHI) or patient-identifiable data. Recipes should use public benchmarks or synthetic examples unless run in a compliant environment.
