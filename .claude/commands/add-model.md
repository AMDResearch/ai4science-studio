# Add a new model to AI4Science Studio

Add a new model to this repository following all repo conventions.

## Steps

1. **Identify domain** — pick from `earth_science/`, `material_science/`, `protein_folding/`, `healthcare/`, or `physics_simulation/`. If the model clearly belongs to a new domain, create the domain folder with a `README.md` first.

2. **Derive the slug**:
   - HF id `org/model` → directory name `org__model` (replace `/` with `__`).
   - If a well-known public name is used instead (e.g. `StormCast`, `NeuralGCM`), use that name and document the canonical HF id inside the model's `README.md`.
   - If weights are not on Hugging Face (e.g. GCS bucket, GitHub release, Google Drive), set the HF id field to `N/A` and add an "Obtaining model weights" section with a fetch snippet.

3. **Create the model folder** by copying `_template/` to `<domain>/models/<slug>/`.

4. **Fill in `README.md`** with:
   - Hugging Face model id (or `N/A` with alternate source documented)
   - Task description
   - License (SPDX id or direct link)
   - Upstream code repo and paper links
   - **Acknowledgements and citation section** (required — see template):
     - Upstream repo URL
     - Paper citation (author, title, venue, DOI or arXiv id)
     - BibTeX or DOI-based "cite as" block copied from the upstream repo or model card
     - ROCm blog URL and author names (AMD Silo AI) if a blog post exists
     - Any named collaboration (e.g. AstraZeneca × AMD, ORNL × AMD)

5. **Add recipes** under `<slug>/recipes/`. Use one subfolder per task (`recipes/inference/`, `recipes/finetune/`, etc.). Each recipe subfolder needs a `README.md`.

6. **Add an `examples/` directory** with ready-to-run scripts:
   - `docker_run.sh` — launches the container; auto-detects AMD Container Toolkit vs device passthrough; checks for existing container before launching; auto-clones the upstream repo if absent.
   - `run_inference.sh` / `run_inference.py` — runs the model; all key params overridable via env vars with sensible defaults; prints a config summary before running.
   - `preflight_<slug>.py` — smoke-test that verifies GPU access and imports before a full run.
   - `sbatch_inference_amd.sh` — SLURM batch script (use `--rocm` for AMD/Apptainer GPU passthrough, not `--nv`). Named `_amd.sh` (not `_mi300x.sh`) — the same script covers MI250X, MI300X, and MI350X with a `rocm7.2.x` image.
   - `build_overlay_amd.sh` — (for HPC models with heavy pip deps, e.g. >1 GB of packages) builds a persistent Apptainer ext3 overlay pre-loaded with pip deps; mount `:ro` in inference jobs to skip the install phase.
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
   - `healthcare/` (Healthcare & Life Sciences): add research/engineering-only disclaimer; no PHI; copy intended-use and limitations from the model card.
   - `physics_simulation/`: state physical domain (fluid dynamics, plasma, etc.), dataset format (HDF5, NetCDF), and HPC/multi-node requirements.

9. **Update `ACKNOWLEDGEMENTS.md`** at the repo root — add a per-model entry under the appropriate domain section following the existing format. Include paper citation, upstream repo, ROCm blog + author (if applicable), and any collaboration callout.

10. **Do not commit** large checkpoints, datasets, `.env` files, or tokens — document how users obtain them instead.

11. **Git workflow** — always branch, never commit directly to `main`:
    1. `git fetch origin && git checkout -b <your-username>/<model-slug> origin/main`
    2. Create all files, set scripts `chmod +x`.
    3. `git add <domain>/models/<slug>/` and commit.
    4. `git push -u origin <your-username>/<model-slug>`
    5. Open a PR with `gh pr create` targeting `main`.
    6. After the PR is merged: `git checkout main && git pull origin main && git branch -D <your-username>/<model-slug>`.

## Model to add

$ARGUMENTS
