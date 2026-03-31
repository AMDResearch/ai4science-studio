---
name: ai4science-healthcare
description: Applies to healthcare-related ML recipes under healthcare/models/ in AI4Science Studio, with privacy and non-clinical-use guardrails.
---

# Healthcare domain

## Scope

The `healthcare/` domain holds recipes for **healthcare-oriented** open models (imaging, clinical NLP, genomics helpers, etc.) on Hugging Face. All content is for **research and engineering** support, not patient care decisions.

## Layout

- Models: `healthcare/models/<model-slug>/`
- Slug rules: `healthcare/models/README.md`
- Structural template reference: `_template/` (repo root)

## Mandatory guardrails

- **No PHI:** Do not add patient-identifiable information, hospital identifiers, free-text notes from real charts, or internal EHR exports to the repository.
- **Not medical advice:** Documentation and scripts must not present outputs as diagnosis, treatment, or clinical guidance.
- **Intended use:** Copy or summarize the model card’s **intended use** and **limitations** into the model `README.md`.

## Agent guidance

- Prefer **public benchmarks** (e.g. de-identified challenge datasets) in examples.
- When discussing evaluation, distinguish **offline metrics** from **regulatory** or **deployment** readiness.
- If a user asks for real-patient pipelines, redirect to **institutional compliance** (IRB, BAAs, local policy) outside this repo’s scope.
