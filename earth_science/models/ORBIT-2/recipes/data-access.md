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

## Staging data on a cluster you control

Use a **shared project or scratch filesystem** visible from both login and compute nodes—not node-local temp only—so jobs can read `low_res_dir` / `high_res_dir` consistently.

### When the release is public (recommended)

1. Open the [Constellation landing page](https://doi.ccs.ornl.gov/dataset/e4c2db1f-e88c-5ad0-bb96-59be0ef7c772) and use the official **Globus “Download Dataset”** link so you get the current collection and path (do not hard-code stale endpoint IDs).
2. Activate or use a **Globus endpoint** your site provides (or a personal endpoint on a machine that can reach the cluster filesystem), then run a **Globus transfer** into a directory on your cluster’s Lustre, GPFS, or equivalent.
3. Point your ORBIT-2 YAML at those directories once the transfer completes.

**Checkpoints** for inference or fine-tuning are on Hugging Face ([`jychoi-hpc/ORBIT-2`](https://huggingface.co/jychoi-hpc/ORBIT-2)); use the Hub for weights and configs, not as a substitute for the full curated **training** NPZ-style dataset tree unless your workflow explicitly uses only Hub artifacts.

### When you cannot pull from the public catalog but have OLCF or a collaborator who does

- Prefer **Globus** transfers from the **same ORNL collections** linked from Constellation, following [OLCF data storage and transfers](https://docs.olcf.ornl.gov/data/index.html), if policy allows **endpoint-to-endpoint** movement to your institution.
- If your project already has a copy under **`/lustre/orion/...`** and **policy and quotas** allow it, you may **rsync** or **scp** from that space to your cluster via a login host—coordinate with your **OLCF PI or support**; there is no universal path in this repo.

## Config reminder

Whatever the source, paths in your config must match **your** filesystem layout and variable naming. Keep dataset provenance and terms in your project documentation alongside the model license.
