---
name: ai4science-earth-science
description: Applies to Earth-system ML in AI4Science Studio—climate, weather, geospatial data, and model folders under earth_science/models/.
---

# Earth science domain

## Scope

The `earth_science/` domain covers **climate**, **weather**, and broader **Earth-system** machine learning: gridded fields, forecasting, downscaling, remote sensing, geospatial tensors, reanalysis-style inputs, etc. Do **not** create separate top-level climate or weather trees.

## Layout

- Models: `earth_science/models/<model-slug>/`
- Template: `earth_science/models/_template/`
- Conventions: `earth_science/models/README.md`

## Agent guidance

- Place new Earth-related HF models under **`earth_science/models/`** unless the model clearly fits another domain better (e.g. pure protein LM → `protein_folding/`).
- Recipes should state **spatial/temporal resolution**, **coordinate conventions** if relevant, and **data sources** (ERA5, satellite products, etc.) without bundling large raw archives in git.
- When suggesting AMD-specific notes, keep them **optional** and tied to tested stack versions (e.g. PyTorch + ROCm).
- **Institutional AMD** clusters and **data staging** (**Globus**, Constellation DOI pages, Hugging Face Hub CLI) onto shared filesystems are in-scope for recipe text; align guidance with **gridded** / reanalysis-style datasets and citation requirements.
- For models whose **authoritative code** lives on GitHub (e.g. **ORBIT-2** under `earth_science/models/ORBIT-2/`), the Studio may add **`examples/`** with **thin** Python or SLURM scripts that **delegate** to upstream entry points (same `cwd` / `PYTHONPATH` / scheduler layout upstream expects). Do **not** copy large upstream training or distributed inference files into Studio—wrappers plus recipe links stay maintainable.
- **Distributed inference** on cluster jobs must match upstream assumptions (e.g. SLURM task count vs YAML **parallelism** product). Site-specific **partition** and **account** names (e.g. HPCFund-style queues) belong in comments or placeholders, not hard-coded secrets.

## Typical pitfalls

- Mixing incompatible projection or time semantics—document CRS and time axis assumptions.
- Omitted license for underlying **datasets**—mention dataset terms alongside the model license when recipes depend on them.
