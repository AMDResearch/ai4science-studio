# HydraGNN: data access

Training artifacts for the **Predictive GFM 2024** ensemble include **ADIOS** datasets and **checkpoint** trees. How you obtain them depends on whether you use the **Hugging Face Hub**, **OLCF Data Constellation**, or your own pipelines.

## Hugging Face Hub (checkpoints and ADIOS mirrors)

The model repo [`mlupopa/HydraGNN_Predictive_GFM_2024`](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024) includes:

- **`ADIOS_files/`** — Subdirectories of preprocessed datasets in **ADIOS** format (names and contents are described on the [model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024); typical names reference sources such as ANI1x, MPTrj, OC20, OC22, QM7-X). These files can be **very large**; plan bandwidth and disk before bulk download.
- **`Ensemble_of_models/`** — Per-trial folders with **`config.json`** and **`*.pk`** checkpoints for each selected GFM.

Always confirm paths against the [live Hub tree](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024/tree/main).

## OLCF Data Constellation (registered package)

The full data and model parameter package is also registered with a **DOI** on **ORNL Data Constellation**:

- **Landing page:** [HydraGNN_Predictive_GFM_2024 dataset](https://doi.ccs.ornl.gov/dataset/3a49c8df-83f7-5d32-84be-f81d289e7cdd)
- **DOI:** `10.13139/OLCF/2474799`

Use that page for citation, provenance, and any **Globus** or site-specific download instructions Constellation provides.

## Staging data on a cluster you control

Place **ADIOS** trees and **checkpoints** on a **shared** filesystem (project or scratch) that compute nodes can read; HydraGNN JSON configs must point at those paths.

### When the release is public (recommended)

**From Hugging Face** (typical for many sites):

- Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and authenticate if the repo requires it.
- Download into a directory on shared storage, for example:
  - `huggingface-cli download mlupopa/HydraGNN_Predictive_GFM_2024 --local-dir /path/on/shared/fs/HydraGNN_Predictive_GFM_2024`
- To limit what is fetched, add **`--include`** / **`--exclude`** glob patterns per the CLI help for your installed `huggingface_hub` version (for example patterns under `ADIOS_files/` or `Ensemble_of_models/`).
- **`ADIOS_files/`** can be **very large**; verify free space, use **resume-friendly** transfers, and consider optional **`hf_transfer`**-accelerated downloads if your admins allow it (see the [Hub download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)).

**From Constellation** (Globus mirror of the registered package):

- Use the [HydraGNN Constellation page](https://doi.ccs.ornl.gov/dataset/3a49c8df-83f7-5d32-84be-f81d289e7cdd) as the source of truth for Globus flow: institutional or personal Globus endpoint → shared cluster path.

### When you cannot use public endpoints but have OLCF or a collaborator who does

- Use **Globus** and [OLCF data guidance](https://docs.olcf.ornl.gov/data/index.html) for endpoint-to-endpoint moves from ORNL where permitted.
- If a copy already lives under **`/lustre/orion/...`** for your project and policy allows **copying out**, use **rsync** or **scp** with PI/support approval—paths are **allocation-specific**, not documented here.

## Methodological context (high level)

The model card states that structures were curated (for example filtering by force magnitude), energies were **re-aligned** across datasets using a **linear regression** on elemental composition, and the aggregated training set **excludes excited states**. **Units and target definitions** for energy and forces are defined by the upstream DFT pipelines and HydraGNN configuration—consult the card and **`Predictive_GFM_2024`** branch docs before comparing numbers across tools.

## Config reminder

Whatever the source, paths in HydraGNN JSON configs must match **your** filesystem layout. Keep dataset provenance, license (**BSD-3-Clause Clear** on the Hub card), and citation in your project documentation.
