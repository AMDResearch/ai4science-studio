# Physics simulation

This domain holds recipes for **physics simulation** machine learning models — including surrogate models, neural operators, and foundation models for continuum dynamics, fluid dynamics, turbulence, plasma physics, and related multiphysics systems.

## Model directories

All models live under [`models/`](models/). Each subfolder is one model or framework:

- [`models/MATEY/`](models/MATEY/) — **ORNL MATEY**: multiscale adaptive transformer framework for spatiotemporal physical systems
- [`models/Walrus/`](models/Walrus/) — **polymathic-ai/walrus**: 1.3B-parameter cross-domain foundation model for continuum dynamics

## Recipes

Per-model recipes live in `models/<model-folder>/recipes/`.

## Contributing

1. Choose or create `models/<model-folder>/` using `_template/`.
2. Document the Hugging Face model id (or `N/A` with alternate source), task, data assumptions, and license.
3. Add minimal scripts or runbooks under `recipes/`.
