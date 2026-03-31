# Add or update a recipe for an existing model

Create or improve a recipe (inference, fine-tune, eval, etc.) for a model already in AI4Science Studio.

## Steps

1. **Locate the model folder** — `<domain>/models/<slug>/recipes/`.

2. **Choose or create a subfolder** for the task:
   - `recipes/inference/`
   - `recipes/finetune/`
   - `recipes/eval/`
   - other task-specific names as needed

3. **Write the recipe**:
   - Keep entrypoints small and well-documented.
   - Pin Python/PyTorch (or other framework) versions.
   - Include optional ROCm/CUDA notes only when a maintainer has actually validated them.
   - Reuse upstream scripts with minimal wrappers; avoid reimplementing full stacks.

4. **Attribution** — credit upstream authors, papers, and license in the model `README.md` and in recipe comments.

5. **Do not commit** secrets, Hub tokens, large binaries, or datasets — point users to documented download steps.

## Domain-specific checks

- `earth_science/`: document spatial/temporal resolution and data sources.
- `material_science/`: document input representations and unit conventions.
- `protein_folding/`: use public or synthetic structures; no confidential sequences.
- `healthcare/`: include research/engineering-only disclaimer; no PHI.

## Recipe to add

$ARGUMENTS
