# Review a model folder for completeness and convention compliance

Audit an existing model entry in AI4Science Studio and report any issues.

## Checklist

### Structure
- [ ] Folder is under the correct domain (`earth_science/`, `material_science/`, `protein_folding/`, `healthcare/`)
- [ ] Slug follows the `org__model` naming rule (or public name with canonical HF id documented)
- [ ] `README.md` exists at `<domain>/models/<slug>/README.md`
- [ ] `recipes/` subfolder exists with at least one task subfolder

### README.md content
- [ ] Hugging Face model id present
- [ ] Task clearly described
- [ ] License (SPDX id or link) present
- [ ] Upstream code repo linked
- [ ] Paper linked (if one exists)

### Domain-specific
- `earth_science/`: spatial/temporal resolution stated in relevant recipes
- `material_science/`: input representations and unit conventions documented
- `protein_folding/`: license restrictions surfaced; no clinical/diagnostic implications
- `healthcare/`: research/engineering-only disclaimer present; no PHI; intended use and limitations from the model card included

### Safety
- [ ] No API keys, tokens, or `.env` credentials committed
- [ ] No large binaries (`.bin`, `.safetensors`, datasets) tracked in git
- [ ] PHI check passes (especially for `healthcare/`)

## Model folder to review

$ARGUMENTS
