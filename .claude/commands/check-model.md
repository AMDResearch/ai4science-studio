# Review a model folder for completeness and convention compliance

Audit an existing model entry in AI4Science Studio and report any issues.

## Checklist

### Structure
- [ ] Folder is under the correct domain (`earth_science/`, `material_science/`, `protein_folding/`, `healthcare/`)
- [ ] Slug follows the `org__model` naming rule, or is a public name with canonical HF id documented in `README.md`
- [ ] `README.md` exists at `<domain>/models/<slug>/README.md`
- [ ] `recipes/` subfolder exists with at least one task subfolder, each containing its own `README.md`
- [ ] `examples/` directory exists with at least `docker_run.sh` and one `run_*.sh`/`run_*.py`

### README.md content
- [ ] Hugging Face model id present (or `N/A` with alternate weight source documented)
- [ ] Task clearly described
- [ ] License (SPDX id or link) present
- [ ] Upstream code repo linked
- [ ] Paper linked (if one exists)
- [ ] If weights are not on HF: "Obtaining model weights" section with fetch snippet

### examples/ scripts
- [ ] `docker_run.sh`: auto-detects AMD Container Toolkit vs device passthrough; checks for existing container; auto-clones upstream repo if absent
- [ ] `run_*.sh` / `run_*.py`: key params overridable via env vars; prints config summary; exits with clear error when required inputs are missing
- [ ] `preflight_<slug>.py`: verifies GPU access and imports
- [ ] `sbatch_*_mi300x.sh`: uses `--rocm` (not `--nv`) for AMD/Apptainer GPU passthrough
- [ ] All scripts are `chmod +x`

### Domain-specific
- `earth_science/`: spatial/temporal resolution stated in relevant recipes
- `material_science/`: input representations and unit conventions documented
- `protein_folding/`: license restrictions surfaced; no clinical/diagnostic implications
- `healthcare/`: research/engineering-only disclaimer present; no PHI; intended use and limitations from the model card included

### AMD/ROCm notes
- [ ] Docker image + ROCm version stated in `docker_run.sh`
- [ ] Any ROCm-specific package substitutions documented in the script
- [ ] Validated optimizations (torch.compile, MIOpen tuning, etc.) noted with measured results in the recipe README; unvalidated claims absent

### Safety
- [ ] No API keys, tokens, or `.env` credentials committed
- [ ] No large binaries (`.bin`, `.safetensors`, datasets) tracked in git
- [ ] PHI check passes (especially for `healthcare/`)

## Model folder to review

$ARGUMENTS
