# ORBIT-2: data access

ORBIT-2 training uses **paired low-resolution and high-resolution** fields. How you obtain them depends on whether you are **inside OLCF** or **outside**.

## Public ORBIT-2 dataset (recommended for most users)

The curated ORBIT-2 dataset release is registered with a **DOI** and described on the ORNL **Constellation** catalog:

- **Landing page:** [ORBIT-2 dataset on Constellation](https://doi.ccs.ornl.gov/dataset/e4c2db1f-e88c-5ad0-bb96-59be0ef7c772)  
- **DOI:** `10.13139/OLCF/2589526`

That page describes the release (integration of ERA5, PRISM, DAYMET, IMERG with preprocessing aligned to WeatherBench2-style practice) and provides a **Globus**-based **“Download Dataset”** flow.

**Use the official link on that page** to open Globus with the correct collection and path. Do **not** rely on copied endpoint IDs or internal OLCF paths from third-party notes—they can change; the Constellation page is the source of truth.

Data are documented as **NumPy** `.npz` / `.npy` style artifacts suitable for standard Python scientific stacks.

## If you run on OLCF (Frontier / Orion)

Frontier compute nodes use the **Orion** Lustre file system. OLCF documents **project-scoped** areas (scratch, proj-shared, world-shared) under `/lustre/orion/...` with quotas, purge policies, and performance guidance.

**Authoritative references:**

- [OLCF data storage and transfers](https://docs.olcf.ornl.gov/data/index.html)  
- [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)

**Important:** The ORBIT-2 YAML configs expect you to set **`low_res_dir`** and **`high_res_dir`** (and related metadata) to directories **you** control. There is **no single universal path** in this Studio repo for “the” ORBIT dataset on Orion—your team stages data under your allocation. Coordinate with your OLCF project PI or support if you need shared project space.

## If you are not on OLCF

Assume you **cannot** see internal Orion paths. Use:

1. The **Constellation / Globus** release above, or  
2. Your own preprocessing from **ERA5**, **PRISM**, **DAYMET**, **IMERG**, etc., respecting each provider’s **license and citation** requirements, using the ORBIT-2 paper and WeatherBench2 references as methodological guides.

## Config reminder

Whatever the source, paths in your config must match **your** filesystem layout and variable naming. Keep dataset provenance and terms in your project documentation alongside the model license.
