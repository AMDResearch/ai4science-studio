# Material science

This domain holds recipes for **material science** and related open models on Hugging Face: properties prediction, generative chemistry, surrogate models for simulation, and similar workflows.

## Model directories

All models live under [`models/`](models/). Each subfolder is one HF model or family:

- Example: [`models/HydraGNN/`](models/HydraGNN/) — **mlupopa/HydraGNN_Predictive_GFM_2024** on Hugging Face (atomistic graph foundation models); see its [`README.md`](models/HydraGNN/README.md).
- [`models/README.md`](models/README.md) — folder naming (`org__model` or public model name) and how to add a model.

For a file layout reference when adding a new model, use [`../_template/`](../_template/).

## Recipes

Per-model recipes live in `models/<model-slug>/recipes/`.

## Contributing

1. Create `models/<model-slug>/README.md` with HF id, license, and upstream links.
2. Add recipes under `models/<model-slug>/recipes/`.
3. Avoid committing large structure files or proprietary datasets; document download or generation steps instead.
