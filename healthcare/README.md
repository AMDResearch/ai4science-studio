# Healthcare & Life Sciences (HCLS)

This domain holds recipes for **healthcare and life sciences** open models on Hugging Face (imaging, clinical NLP, genomics helpers, drug discovery, molecular design, etc.). Content here is for **research and engineering** workflows only.

## Model directories

All models live under [`models/`](models/):

- [`models/README.md`](models/README.md) — slug naming and how to add a model.

Layout reference: [`../_template/`](../_template/).

## Recipes

Per-model recipes live in `models/<model-slug>/recipes/`.

## Disclaimers

- Recipes are **not** medical advice, diagnosis, or treatment.
- Do **not** commit **PHI**, patient-identifiable data, or internal hospital artifacts. Use public benchmarks, toy examples, or synthetic data in this repository.

## Contributing

1. Create `models/<model-slug>/README.md` with HF id, license, **intended use** from the model card, and disclaimers where appropriate.
2. Add recipes under `models/<model-slug>/recipes/` that avoid real patient data.
3. Prefer documenting evaluation on public datasets rather than private clinical pipelines.
