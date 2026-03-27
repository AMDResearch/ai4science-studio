---
name: ai4science-material-science
description: Applies to material science and chemistry ML recipes under material_science/models/ in AI4Science Studio.
---

# Material science domain

## Scope

The `material_science/` domain covers **materials**, **chemistry**, and related ML: property prediction, generative models, surrogates for DFT or MD-scale workflows when exposed as Hugging Face models, and similar tasks.

## Layout

- Models: `material_science/models/<model-slug>/`
- Slug rules: `material_science/models/README.md`
- Structural template reference: `earth_science/models/_template/`

## Agent guidance

- Keep recipes **explicit** about input representations (graphs, crystals, SMILES, etc.) and any **unit conventions**.
- Prefer citing **benchmarks** and **baseline** numbers from literature or model cards when adding evaluation snippets.
- Do not commit large **proprietary structure databases**; link to public sources or describe how users supply their own data.

## Safety and licensing

- Chemistry and materials models may have **use-based** restrictions; mirror the model card’s limitations in the local `README.md`.
