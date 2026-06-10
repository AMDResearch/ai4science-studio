---
name: ai4science-earth-science
description: Applies to Earth-system ML in AI4Science Studio—climate, weather, geospatial data, and model folders under earth_science/models/. Includes validated AMD/ROCm HPC patterns for earth science models.
---

# Earth science domain

## Scope

The `earth_science/` domain covers **climate**, **weather**, and broader **Earth-system** machine learning: gridded fields, forecasting, downscaling, remote sensing, geospatial tensors, reanalysis-style inputs, etc. Do **not** create separate top-level climate or weather trees.

## Layout

- Models: `earth_science/models/<model-slug>/`
- Template: `_template/` (repo root)
- Conventions: `earth_science/models/README.md`

## Agent guidance

- Place new Earth-related HF models under **`earth_science/models/`** unless the model clearly fits another domain better (e.g. pure protein LM → `protein_folding/`).
- Recipes should state **spatial/temporal resolution**, **coordinate conventions** if relevant, and **data sources** (ERA5, satellite products, etc.) without bundling large raw archives in git.
- When suggesting AMD-specific notes, keep them **optional** and tied to tested stack versions (e.g. PyTorch + ROCm).
- **Institutional AMD** clusters and **data staging** (**Globus**, Constellation DOI pages, Hugging Face Hub CLI) onto shared filesystems are in-scope for recipe text; align guidance with **gridded** / reanalysis-style datasets and citation requirements.
- For models whose **authoritative code** lives on GitHub (e.g. **ORBIT-2** under `earth_science/models/ORBIT-2/`), the Studio may add **`examples/`** with **thin** Python or SLURM scripts that **delegate** to upstream entry points (same `cwd` / `PYTHONPATH` / scheduler layout upstream expects). Do **not** copy large upstream training or distributed inference files into Studio—wrappers plus recipe links stay maintainable.
- **Distributed inference** on cluster jobs must match upstream assumptions (e.g. SLURM task count vs YAML **parallelism** product). Site-specific **partition** and **account** names (e.g. HPCFund-style queues) belong in comments or placeholders, not hard-coded secrets.

## Validated AMD HPC patterns for earth science models

### StormCast (earth2studio)

- **No local data needed:** earth2studio DataSource classes fetch HRRR/GFS live from NOAA HTTPS archives. Compute nodes need outbound HTTPS to NOAA — no pre-staging required.
- **Overlay size:** 4 GB ext3; ~1.7 GB content (earth2studio[stormcast] + cartopy, stripped of torch)
- **ZarrBackend pitfall:** `run_inference.py` raises an error if the output zarr already exists. Always `rm -rf <output>.zarr` before each run in the sbatch script.
- **No CUDA packages:** install earth2studio with `--no-deps` then add deps manually; physicsnemo and timm also need `--no-deps` (they declare torch, which would pull CUDA torch).

### ORBIT-2 (pytorch-lightning + xformers + MPI)

- **Synthetic data generator:** `make_synthetic_data.py` generates a ~2 MB synthetic dataset (low_res + high_res NetCDF shards) so the full pipeline can be smoke-tested without ERA5/PRISM data. Use `ORBIT2_USE_SYNTHETIC=1` in the sbatch script.
- **Overlay size:** 7 GB ext3; ~2 GB content (pytorch-lightning, xformers, mpi4py, wandb, tensorboard, etc.)
- **pytorch-lightning must be installed with `--no-deps`** — it declares `torch` as a dep; pip resolves CUDA torch from PyPI which silently replaces the ROCm torch in PYTHONPATH.
- **xformers for ROCm:** install from the PyTorch ROCm wheel index, not PyPI. PyPI xformers links against `libcudart.so.12`. Use `--index-url https://download.pytorch.org/whl/${ROCM_WHL_TAG}` with `--no-deps`.
- **xformers.components shim:** ORBIT-2 imports `xformers.components.attention.core.scaled_dot_product_attention`. This subpackage was removed in xformers ≥0.0.28. Write a three-file shim after install:
  ```python
  (xf / "components" / "__init__.py").write_text("")
  (xf / "components" / "attention" / "__init__.py").write_text("")
  (xf / "components" / "attention" / "core.py").write_text(
      "import torch.nn.functional as F\n\n"
      "def scaled_dot_product_attention(q, k, v, att_mask=None, dropout=0.0):\n"
      "    return F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask, dropout_p=dropout)\n"
  )
  ```
- **Distributed launch:** `srun --mpi=pmix apptainer exec ... --env HOSTNAME="$MASTER_ADDR"` — see studio SKILL.md for full pattern. Confirmed working for 1-GPU and 8-GPU.
- **Data blocker:** ORBIT-2 real training data lives on Frontier Lustre (`/lustre/orion/<YOUR_PROJECT>/...`) — not publicly accessible. Synthetic data covers the smoke-test path.
- **Training on institutional clusters:** `sbatch_train_amd.sh` + `run_orbit2_train.py` wrap `intermediate_downscaling.py` with Apptainer overlay, `srun --mpi=pmix`, and `interm_8m_lux.yaml`. Staged Constellation/Globus 10.0_arcmin NPZ data can be used in **same-dir mode** (`low_res` = `high_res`, `superres_mag: 1`) for perf/scaling timing only.
- **Training landmines (2026-06):** stub `gptl4py` in `run_orbit2_train.py`; overlay may lack `xformers.ops` → auto `FusedAttn.DEFAULT`; `max_epochs` must be ≥ 2; batch cap patches module before `main()` not via `runpy.run_path`.
- **Perf/scaling:** `sbatch_train_perf_amd.sh`, `run_scaling_study.sh`, artifacts under `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/` and `outputs/scaling-*`. See [recipes/perf-analysis/HANDOFF.md](../../../earth_science/models/ORBIT-2/recipes/perf-analysis/HANDOFF.md).
- **Multi-node scheduling:** Before large submits, check partition state (`sinfo`, site dashboards). Prefer **idle, healthy** nodes; use `--nodelist` / `--exclude` only per your site's policy. Do not stack two exclusive full-node GPU jobs on the same host.

### ArchesWeather (geoarches)

- **Not on PyPI:** install with `pip install "git+https://github.com/INRIA/geoarches.git"`.
- **Data blocker:** `encode_dataset` requires ERA5 zarr (~35 GB). Not live-fetchable — must be staged on cluster before this model can run end-to-end.
- **CLI syntax:** `geoarches.inference.encode_dataset` uses argparse (`--uids`, `--input-path`, `--output-path`), not Hydra `++` overrides.
- **Environment:** installs cleanly; checkpoint downloads from HF (`gcouairon/ArchesWeather`). Only the data is blocked.

### Aurora, GenCast, PanguWeather (ai-models framework)

- **Shared recipe repo:** all three use `silogen/ai-samples` (`ai4sciences/ai-weather-forecasting`).
- **Two Docker images:** Aurora uses `pytorchweather:latest` (PyTorch); GenCast and PanguWeather share `jaxweather:latest` (JAX).
- **CDS credentials required:** all three fetch ERA5 initial conditions from the Copernicus Climate Data Store. Users must configure `env_file` with their `CDSAPI_KEY`.
- **PanguWeather ONNX patch:** one-line patch adds the ROCm execution provider — handled automatically by the Docker image.
- **GenCast ensemble sizing:** `NUM_ENSEMBLE_MEMBERS` must be a multiple of the GPU count.
- **Aurora memory:** 0.1° resolution (~6× more grid points than 0.25°) demands the MI300X's full 192 GB HBM3.

### NeuralGCM (JAX + GCS)

- **No CDS/HF needed:** weights from GCS (`gs://neuralgcm/models/`) and ERA5 from public ARCO-ERA5 Zarr — both anonymous, no auth required.
- **JAX for ROCm:** requires a custom JAX ROCm wheel (`jaxlib-0.6.0`) from `ROCm/rocm-jax` GitHub releases, plus `jax-rocm7-pjrt` and `jax-rocm7-plugin`.
- **Docker image:** `rocm/dev-ubuntu-22.04:7.0.2-complete` — deps installed at container start by `docker_run.sh`.
- **`libdw1`:** required JAX runtime dependency on Ubuntu 22.04; installed via `apt install -y libdw1`.
- **Performance:** 0.7° ~110s, 1.4° ~48s, 2.8° ~33s for a 4-day forecast on MI300X.

## Overlay pip install pitfalls (common to all earth science models)

**`pip --target` ignores the SIF's installed packages** — it resolves everything fresh. When torch is a transitive dep, pip downloads a fresh ROCm torch wheel (~6 GB), filling the overlay before other packages install.

**Constraints file approach does NOT work:**
- The venv's torch has a local version string: `torch==2.10.0+rocm7.2.2.gitXXX`
- pip cannot match this against `>=2.5.0` from `--target` mode → `ResolutionImpossible`
- Bare package names in constraints files (e.g. `torchvision` with no `==X.Y`) are also invalid pip constraint syntax → same error

**Correct pattern — NFS staging + strip + copy:** see studio SKILL.md for full code. Use `--extra-index-url https://download.pytorch.org/whl/<ROCM_WHL_TAG>` (no constraint); ROCm torch has no `nvidia-*` CUDA co-package deps so the entire CUDA chain is never resolved.

## Validated dep quirks (earth science models)

- **`tensordict`** (used by StormCast via earth2studio) requires **`pyvers`** at runtime. `--no-deps` for tensordict is needed to avoid pulling CUDA torch, but `pyvers` must be installed separately alongside the other runtime deps.
- **StormCast no-overlay path**: uses the install-to-tmp + bind-mount pattern (see studio SKILL.md). Packages go to `/opt/stormcast-pkgs` via `--target` + bind-mount. `PYTHONPATH` must include this path in the inference `bash -c` block.
- **ORBIT-2 no-overlay path**: same pattern, packages at `/opt/orbit2-pkgs`. All three `apptainer exec` calls (checkpoint download, synthetic data gen, srun inference) must include `"${PKGDIR_BIND[@]}"`.

## Typical pitfalls

- Mixing incompatible projection or time semantics — document CRS and time axis assumptions.
- Omitted license for underlying **datasets** — mention dataset terms alongside the model license when recipes depend on them.
- Hardcoding HF repo filenames without verifying: use `list_repo_files()` to discover actual checkpoint names.
