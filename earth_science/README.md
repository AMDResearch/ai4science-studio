# Earth science

This domain holds recipes for **Earth-system** machine learning on Hugging Face and related open models—including **climate**, **weather** (forecasting, nowcasting, downscaling), **geospatial** and **remote sensing** applications, and other planetary-surface or atmosphere–ocean modeling when it fits this scope.

There are **no** separate top-level `climate/` or `weather/` trees; both belong here.

## Model directories

All models live under [`models/`](models/). Each subfolder (except `_template`) is one HF model or family:

- [`models/README.md`](models/README.md) — slug naming (`org__model`) and how to add a model.
- [`models/_template/`](models/_template/) — copy this when creating a new model folder.

## Recipes

Per-model recipes live in `models/<model-slug>/recipes/`. See the top-level recipes index: [`../recipes/README.md`](../recipes/README.md).

## Contributing

1. Choose or create `models/<model-slug>/` using the template.
2. Document the Hugging Face model id, task, data assumptions, and license.
3. Add minimal scripts or runbooks under `recipes/`.
