---
name: ai4science-material-science
description: Applies to material science and chemistry ML recipes under material_science/models/ in AI4Science Studio.
---

# Material science domain

## Scope

The `material_science/` domain covers **materials**, **chemistry**, and related ML: property prediction, generative models, surrogates for DFT or MD-scale workflows when exposed as Hugging Face models, and similar tasks.

## Layout

- Models: `material_science/models/<model-slug>/`
- Slug rules: `material_science/models/README.md` (default `org__model`; public on-disk names such as `HydraGNN` are allowed when the model `README.md` states the canonical Hub id)
- Structural template reference: `_template/` (repo root)
- Example: [`material_science/models/HydraGNN/`](../../../material_science/models/HydraGNN/) — atomistic **graph** foundation models (**HydraGNN**), Hub id **`mlupopa/HydraGNN_Predictive_GFM_2024`**

## Agent guidance

- Keep recipes **explicit** about input representations (graphs, crystals, SMILES, etc.) and any **unit conventions** (including **energy** vs **force** targets and graph vs node outputs where relevant).
- Prefer citing **benchmarks** and **baseline** numbers from literature or model cards when adding evaluation snippets.
- Do not commit large **proprietary structure databases**; link to public sources or describe how users supply their own data.
- **Institutional AMD** clusters and **staging** large artifacts (**Hugging Face CLI**, **Globus** / Constellation mirrors, OLCF copy-out when applicable) belong in `data-access.md`-style runbooks for **ADIOS** or similar scientific I/O when models use them.

## HydraGNN multi-node training pattern

HydraGNN uses MPI-based distributed training with ADIOS datasets. Key patterns for AMD clusters:

- **Launch:** `srun --mpi=pmix apptainer exec --rocm --overlay <overlay>:ro $SIF bash <rank_script>`. The `--mpi=pmix` is required because `mpi4py` calls `MPI_Init` inside the container.
- **Non-DDStore path (phase 1):** Use `--multi --multi_model_list=<datasets>` without `--ddstore`. Each rank opens ADIOS files directly via `AdiosMultiDataset`. Simpler and sufficient for moderate scales (1-8 nodes).
- **DDStore path (phase 2):** Add `--ddstore` flag plus env vars `HYDRAGNN_AGGR_BACKEND=mpi`, `HYDRAGNN_DDSTORE_METHOD=1`, `HYDRAGNN_CUSTOM_DATALOADER=1`. Required for large-scale (32+ nodes) where per-rank ADIOS I/O becomes a bottleneck.
- **Dataset convention:** ADIOS datasets are expected at `./dataset/<name>-v2.bp` relative to the training script. Symlink from shared storage rather than copying.
- **Config file:** The upstream `gfm_mlip.json` is hardware-agnostic and works on any cluster. Use CLI flags (`--batch_size`, `--num_epoch`, `--precision`) to override parameters at runtime without modifying the file.
- **Env vars inside container:** Set `OMP_NUM_THREADS` to match `--cpus-per-task`, `MIOPEN_DISABLE_CACHE=1`, `MIOPEN_USER_DB_PATH=$SCRATCH_LOCAL/<jobid>/miopen` (node-local fast storage from `.cluster-config.yaml`), `HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1`.

## Safety and licensing

- Chemistry and materials models may have **use-based** restrictions; mirror the model card's limitations in the local `README.md`.
