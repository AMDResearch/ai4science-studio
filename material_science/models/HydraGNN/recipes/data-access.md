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

## Methodological context (high level)

The model card states that structures were curated (for example filtering by force magnitude), energies were **re-aligned** across datasets using a **linear regression** on elemental composition, and the aggregated training set **excludes excited states**. **Units and target definitions** for energy and forces are defined by the upstream DFT pipelines and HydraGNN configuration—consult the card and **`Predictive_GFM_2024`** branch docs before comparing numbers across tools.

## Config reminder

Whatever the source, paths in HydraGNN JSON configs must match **your** filesystem layout. Keep dataset provenance, license (**BSD-3-Clause Clear** on the Hub card), and citation in your project documentation.
