---
name: ai4science-material-science
description: Applies to material science and chemistry ML recipes under material_science/models/ in AI4Science Studio.
---

# Material science domain

## Scope

The `material_science/` domain covers **materials**, **chemistry**, and related ML: property prediction, generative models, surrogates for DFT or MD-scale workflows when exposed as Hugging Face models, and similar tasks.

## Layout

- Models: `material_science/models/<model-slug>/`
- Slug rules: `material_science/models/README.md` (default `org__model`; public on-disk names such as `HydraGNN` are allowed when the model `README.md` states the canonical Hub id)
- Structural template reference: `_template/` (repo root)
- Example: [`material_science/models/HydraGNN/`](../../../material_science/models/HydraGNN/) — atomistic **graph** foundation models (**HydraGNN**), Hub id **`mlupopa/HydraGNN_Predictive_GFM_2024`**

## Agent guidance

- Keep recipes **explicit** about input representations (graphs, crystals, SMILES, etc.) and any **unit conventions** (including **energy** vs **force** targets and graph vs node outputs where relevant).
- Prefer citing **benchmarks** and **baseline** numbers from literature or model cards when adding evaluation snippets.
- Do not commit large **proprietary structure databases**; link to public sources or describe how users supply their own data.
- **Institutional AMD** clusters and **staging** large artifacts (**Hugging Face CLI**, **Globus** / Constellation mirrors, OLCF copy-out when applicable) belong in `data-access.md`–style runbooks for **ADIOS** or similar scientific I/O when models use them.

## Safety and licensing

- Chemistry and materials models may have **use-based** restrictions; mirror the model card’s limitations in the local `README.md`.
