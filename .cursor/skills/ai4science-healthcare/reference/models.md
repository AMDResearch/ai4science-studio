# Healthcare model HPC notes (GP-MoLFormer)

## AMD/HPC patterns for healthcare models

### GP-MoLFormer

- **Apptainer SLURM path:** `sbatch_inference_amd.sh` clones `IBM/gp-molformer` inside the container on first run, installs deps via `pip install --target /opt/gpmol-pkgs`, then calls `run_generation.sh`. Internet access from compute nodes required for first run. Subsequent runs reuse the clone from `GPMOL_WORK_DIR`.
- **Upstream uses `environment.yml` (conda), not `requirements.txt`.** The sbatch script translates conda deps to pip equivalents. Key mappings: `pytorch` is already in SIF (skip), `rdkit` becomes `rdkit-pypi`, `pytorch-cuda` is skipped (ROCm).
- **No overlay needed:** deps are lightweight; runtime install (~1 min) is acceptable. Torch is stripped from `--target` dir after install to keep the SIF's ROCm torch.
- **`fast_transformers` unavailable on ROCm:** the code falls back to standard PyTorch transformers automatically; no user action needed.
- **Key env vars:** `GPMOL_SIF` (required), `GPMOL_WORK_DIR` (default: examples dir), `SCAFFOLD` (empty = unconditional), `NUM_BATCHES` (default 1 = 1000 molecules), `OUTPUT_FILE`.
- **ROCm version compat:** The ROCm 7.0 SIF (`py3.10`) fails silently on hosts with newer `amdgpu` drivers (e.g. kernel module `6.16.6` on MI300A clusters) — `torch.cuda.is_available()` returns `False` and generation falls back to CPU (~10-20x slower). Use the ROCm 7.2.2 SIF (`py3.12`) on such clusters, or set `HSA_OVERRIDE_GFX_VERSION=9.4.2` as a workaround. The sbatch script includes a GPU detection check that warns when this happens.
- **`rdkit-pypi` has no py3.12 wheel:** When using the ROCm 7.2.2 SIF (py3.12), the RDKit validity check is gracefully skipped. Molecules are still generated correctly.
- **Validated results:**
  - MI300X: 996/1000 valid (99.6%), ~41s wall time (ROCm 7.0 SIF)
  - MI300A: 1000 molecules generated, ~18s wall time (ROCm 7.2.2 SIF)

### General healthcare model pattern

- Use `sbatch_<task>_amd.sh` naming (not `_mi300x.sh`); same script covers MI250X, MI300X, MI350X.
- Always include the research/engineering-only disclaimer both in the recipe README and in the sbatch script header comment.
