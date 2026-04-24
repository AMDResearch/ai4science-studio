# Models (Physics simulation)

Each subdirectory under `models/` is one **model or framework** for physics simulation ML — including surrogate models, neural operators, and foundation models for continuum dynamics, fluid dynamics, turbulence, plasma physics, and related multiphysics systems.

## Current models

| Folder | HF id | Task |
|--------|-------|------|
| [`MATEY/`](MATEY/) | N/A (train from scratch) | Multiscale adaptive transformer for spatiotemporal physical systems |
| [`Walrus/`](Walrus/) | `polymathic-ai/walrus` | 1.3B cross-domain continuum dynamics foundation model |

## Slug convention

Map the Hugging Face id `org/model` to a single directory name by replacing `/` with `__` (double underscore).

Some folders use a **public model name** on disk (e.g. `MATEY`, `Walrus`) instead of `org__model`. In those cases the **canonical Hugging Face id** must be stated clearly at the top of that model's `README.md`.

## Adding a new model

1. Create `models/<your-slug>/README.md` with HF model id (or `N/A` with alternate source), license, and upstream links (see [`../../_template/`](../../_template/) for layout).
2. Add recipes under `models/<your-slug>/recipes/`.
3. Add ready-to-run scripts under `models/<your-slug>/examples/`.
