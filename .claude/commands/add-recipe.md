# Add or update a recipe for an existing model

Create or improve a recipe (inference, fine-tune, eval, etc.) for a model already in AI4Science Studio.

## Steps

1. **Locate the model folder** — `<domain>/models/<slug>/`.

2. **Choose or create a recipe subfolder** under `recipes/`:
   - `recipes/inference/`
   - `recipes/finetune/`
   - `recipes/eval/`
   - other task-specific names as needed
   - Each subfolder needs a `README.md` with a callout box at the top linking to `examples/`.

3. **Write or update example scripts** under `<slug>/examples/`:

   | Script | Purpose |
   |--------|---------|
   | `docker_run.sh` | Launch the container; auto-detect AMD Container Toolkit (`docker info \| grep -qi amd`) vs device passthrough (`/dev/kfd` + all `/dev/dri/renderD*` nodes); check for existing container and exit with attach hint rather than error; auto-clone upstream repo if absent. |
   | `run_<task>.sh` / `run_<task>.py` | Run the model; all key params overridable via env vars with sensible defaults; print config summary before running; print error + fix instructions when required inputs are missing, then exit 1. |
   | `preflight_<slug>.py` | Smoke-test: verify GPU access, imports, and a tiny forward pass. |
   | `sbatch_<task>_mi300x.sh` | SLURM batch script; use `--rocm` (not `--nv`) for AMD/Apptainer GPU passthrough. |

   - All scripts must be `chmod +x`.
   - Use a `tl_config.toml.template` alongside run scripts for config-file-driven models.

4. **Recipe README content**:
   - Add a callout box at the top pointing to `examples/`.
   - Pin Python/framework versions.
   - Include ROCm/CUDA notes only for validated results; state what was tested and the outcome (e.g. speedup %).
   - Reuse upstream scripts with minimal wrappers; avoid reimplementing full stacks.

5. **Attribution** — credit upstream authors, papers, and license in the model `README.md` and in recipe comments.

6. **Do not commit** secrets, Hub tokens, large binaries, or datasets — point users to documented download steps.

## Domain-specific checks

- `earth_science/`: document spatial/temporal resolution and data sources.
- `material_science/`: document input representations and unit conventions.
- `protein_folding/`: use public or synthetic structures; no confidential sequences.
- `healthcare/`: include research/engineering-only disclaimer; no PHI.

## Recipe to add

$ARGUMENTS
