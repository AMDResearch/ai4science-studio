---
name: ai4science-protein-folding
description: Applies to protein structure, folding, and protein LM recipes under protein_folding/models/ in AI4Science Studio.
---

# Protein folding domain

## Scope

The `protein_folding/` domain covers **protein structure prediction**, **folding-related** models, **protein language models**, and adjacent open models on Hugging Face used for structural biology workflows.

## Layout

- Models: `protein_folding/models/<model-slug>/`
- Slug rules: `protein_folding/models/README.md`
- Structural template reference: `_template/` (repo root)

## Agent guidance

- Respect **model and dataset licenses** (research-only, non-commercial, etc.) and surface them in the model `README.md`.
- Avoid implying **clinical** or **diagnostic** use unless the upstream model is explicitly cleared for that (most are not).
- Recipes should use **public structures** or **synthetic** examples for demonstrations; do not embed confidential sequence data.

## Ethics

- When discussing **pathogens** or **dual-use** contexts, stay aligned with upstream policy and general biosafety norms; do not help circumvent stated restrictions.
