# Add a new model to AI4Science Studio

Add a new model to this repository following all repo conventions.

## Steps

1. **Identify domain** — pick from `earth_science/`, `material_science/`, `protein_folding/`, or `healthcare/`.

2. **Derive the slug**:
   - HF id `org/model` → directory name `org__model` (replace `/` with `__`).
   - If a well-known public name is used instead (e.g. `StormCast`, `NeuralGCM`), use that name and document the canonical HF id inside the model's `README.md`.
   - If weights are not on Hugging Face (e.g. GCS bucket, GitHub release, Google Drive), set the HF id field to `N/A` and add an "Obtaining model weights" section with a fetch snippet.

3. **Create the model folder** by copying `earth_science/models/_template/` to `<domain>/models/<slug>/`.

4. **Fill in `README.md`** with:
   - Hugging Face model id (or `N/A` with alternate source documented)
   - Task description
   - License (SPDX id or direct link)
   - Upstream code repo and paper links

5. **Add recipes** under `<slug>/recipes/`. Use one subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.). Each recipe subfolder needs a `README.md`.

6. **Add an `examples/` directory** with ready-to-run scripts:
   - `docker_run.sh` — launches the container; auto-detects AMD Container Toolkit vs device passthrough; checks for existing container before launching; auto-clones the upstream repo if absent.
   - `run_inference.sh` / `run_inference.py` — runs the model; all key params overridable via env vars with sensible defaults; prints a config summary before running.
   - `preflight_<slug>.py` — smoke-test that verifies GPU access and imports before a full run.
   - `sbatch_inference_mi300x.sh` — SLURM batch script (use `--rocm` for AMD/Apptainer GPU passthrough, not `--nv`).
   - Add `run_ensemble.py`, `run_finetune.sh`, etc. as applicable.
   - All scripts must be `chmod +x`.

7. **AMD/ROCm notes** — include only what has actually been validated:
   - Note the base Docker image and ROCm version in `docker_run.sh`.
   - If ROCm-specific packages are needed (e.g. ROCm forks of scatter/sparse), document the substitution in the script.
   - If `torch.compile` or other AMD optimizations have been tested, note results in the recipe README.

8. **Domain-specific checks**:
   - `earth_science/`: state spatial/temporal resolution, coordinate conventions, data sources (ERA5, satellite, etc.).
   - `material_science/`: state input representations (graphs, SMILES, crystals) and unit conventions.
   - `protein_folding/`: surface license restrictions; avoid implying clinical/diagnostic use.
   - `healthcare/`: add research/engineering-only disclaimer; no PHI; copy intended-use and limitations from the model card.

9. **Do not commit** large checkpoints, datasets, `.env` files, or tokens — document how users obtain them instead.

## Model to add

$ARGUMENTS
