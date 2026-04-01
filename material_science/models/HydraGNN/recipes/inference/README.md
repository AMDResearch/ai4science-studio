# HydraGNN: inference and ensemble workflows

This is the usual **AI4Science Studio** entry point: use **published checkpoints** from Hugging Face with **HydraGNN** on branch **`Predictive_GFM_2024`**.

## Prerequisites

1. Clone [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) and check out branch **`Predictive_GFM_2024`**.
2. Install the package and dependencies per the upstream README (for example `pip install -e .` and the repository's requirement files).
3. Download the **ensemble** artifacts you need from [`mlupopa/HydraGNN_Predictive_GFM_2024`](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024). Under `Ensemble_of_models/`, each trial folder (e.g. `gfm_0.229`) typically contains a **`config.json`** and checkpoint files named like **`gfm_0.<ID>_epoch_<N>.pk`** (see the live Hub tree for exact names).

## Align config with weights

The **architecture and data settings** in `config.json` must match the checkpoint you load. Use the **`config.json` from the same Hub subfolder** as the `.pk` file you choose.

## Single-model load and prediction

HydraGNN documents high-level APIs such as **`hydragnn.load_existing_model`** and **`hydragnn.run_prediction`** with a JSON configuration (see the [HydraGNN README](https://github.com/ORNL/HydraGNN) on your checked-out branch). Point the configuration at your local paths for the downloaded **`config.json`** and **checkpoint** as required by that version of the code.

Your **input graphs** must match the featurization and task definition expected by the pretrained GFMs (atomistic structure, node features, and targets as described upstream). **Energy and force units** follow the training pipeline documented on the model card and in HydraGNN—do not assume arbitrary unit conversion without checking upstream.

## Ensemble averaging and uncertainty

For **ensemble averaging** and **epistemic uncertainty quantification** across the fifteen GFMs, use the scripts under **`examples/ensemble_learning`** on branch **`Predictive_GFM_2024`**, as referenced on the [model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024).

## Hub layout caveat

Folder and file names on the Hub **can change**. Always verify paths against the [current file tree](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024/tree/main) before automating downloads.

## See also

- [../data/README.md](../data/README.md) — ADIOS training data and Constellation package.
- [../compute/README.md](../compute/README.md) — hardware expectations for multi-model inference vs training.
