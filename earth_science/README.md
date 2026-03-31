# Earth science

This domain holds recipes for **Earth-system** machine learning on Hugging Face and related open models—including **climate**, **weather** (forecasting, nowcasting, downscaling), **geospatial** and **remote sensing** applications, and other planetary-surface or atmosphere–ocean modeling when it fits this scope.

There are **no** separate top-level `climate/` or `weather/` trees; both belong here.

## Model directories

All models live under [`models/`](models/). Each subfolder is one HF model or family:

- Example: [`models/ORBIT-2/`](models/ORBIT-2/) — **jychoi-hpc/ORBIT-2** on Hugging Face (weather and climate downscaling); see its [`README.md`](models/ORBIT-2/README.md).
- [`models/README.md`](models/README.md) — folder naming (`org__model` or public model name) and how to add a model.
- [`../_template/`](../_template/) — copy this when creating a new model folder.

## Recipes

Per-model recipes live in `models/<model-folder>/recipes/`. See the top-level recipes index: [`../recipes/README.md`](../recipes/README.md).

## Contributing

1. Choose or create `models/<model-folder>/` using the template.
2. Document the Hugging Face model id, task, data assumptions, and license.
3. Add minimal scripts or runbooks under `recipes/`.
