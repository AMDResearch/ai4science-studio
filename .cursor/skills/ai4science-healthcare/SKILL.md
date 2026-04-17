---
name: ai4science-healthcare
description: Applies to healthcare and life sciences ML recipes under healthcare/models/ in AI4Science Studio, with privacy and non-clinical-use guardrails.
---

# Healthcare & Life Sciences (HCLS) domain

## Scope

The `healthcare/` domain holds recipes for **healthcare and life sciences** open models (imaging, clinical NLP, genomics helpers, drug discovery, molecular design, etc.) on Hugging Face. All content is for **research and engineering** support, not patient care decisions.

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

## AMD/HPC patterns for healthcare models

### GP-MoLFormer

- **Apptainer SLURM path:** `sbatch_inference_amd.sh` clones `IBM/gp-molformer` inside the container on first run, installs `requirements.txt` via pip, then calls `run_generation.sh`. Internet access from compute nodes required for first run. Subsequent runs reuse the clone from `GPMOL_WORK_DIR`.
- **No overlay needed:** deps are lightweight; runtime install (~1 min) is acceptable. No torch-overwrite risk because gp-molformer’s requirements don’t pin torch.
- **`fast_transformers` unavailable on ROCm:** the code falls back to standard PyTorch transformers automatically; no user action needed.
- **Key env vars:** `GPMOL_SIF` (required), `GPMOL_WORK_DIR` (default: examples dir), `SCAFFOLD` (empty = unconditional), `NUM_BATCHES` (default 1 = 1000 molecules), `OUTPUT_FILE`.
- **Validated result:** 996/1000 valid molecules (99.6%) unconditional generation, 41s wall time on MI300X.

### General healthcare model pattern

- Use `sbatch_<task>_amd.sh` naming (not `_mi300x.sh`); same script covers MI250X, MI300X, MI350X.
- Always include the research/engineering-only disclaimer both in the recipe README and in the sbatch script header comment.
