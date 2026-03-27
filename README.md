# AI4Science Studio

Open playground for **recipes** around AI-for-science models: training, fine-tuning, inference, evaluation, and related workflows. Models are expected to be **open** and discoverable on [Hugging Face](https://huggingface.co/); upstream code and tests may live on Hugging Face or in separate GitHub repositories.

Optional notes for **AMD** hardware (for example **ROCm** with PyTorch) can be added in individual recipes where maintainers have validated them—nothing here is a substitute for upstream documentation.

## Layout

Recipes are organized by **science domain**, then **model**, then **task-specific files**:

```text
<domain>/models/<model-slug>/README.md   # HF id, license, links
<domain>/models/<model-slug>/recipes/    # scripts, notebooks, or how-to docs
```

**Domains**

| Domain | Path | Scope |
|--------|------|--------|
| Earth science | [`earth_science/`](earth_science/) | Climate, weather, Earth-system and geospatial ML |
| Material science | [`material_science/`](material_science/) | Materials, chemistry, simulation surrogates |
| Protein folding | [`protein_folding/`](protein_folding/) | Structure, folding, protein LMs |
| Healthcare | [`healthcare/`](healthcare/) | Healthcare-related ML (see disclaimers below) |

**Model slug:** Hugging Face id `org/model` → directory `org__model` (replace `/` with `__`). Details: each domain’s [`models/README.md`](earth_science/models/README.md).

**Getting started:** Copy [`earth_science/models/_template/`](earth_science/models/_template/) when adding a new model folder.

## Recipes index

See [`recipes/README.md`](recipes/README.md) for how the tree fits together and links to each domain.

## Agent skills (Cursor)

This repository includes **Cursor Agent Skills** under [`.cursor/skills/`](.cursor/skills/). They describe repo conventions, Hugging Face recipe workflows, and domain-specific cautions so coding agents stay aligned with this project.

## Disclaimers

- Each model remains under its **upstream license**; check the model card on Hugging Face before use.
- **Healthcare** content is for research and engineering only—not medical advice, diagnosis, or treatment. Do not use patient-identifiable or PHI in examples committed to this repo.
- This studio is a **community-style recipe collection** unless otherwise stated by your organization.

## Contributing a recipe

1. Pick the domain and create `models/<model-slug>/` if it does not exist (use the template under `earth_science/models/_template/`).
2. Document the Hugging Face model id and license in that folder’s `README.md`.
3. Add minimal, reproducible steps or scripts under `recipes/` and avoid committing secrets or large binary artifacts (see [`.gitignore`](.gitignore)).
