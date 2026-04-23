# HydraGNN: inference and ensemble workflows

> **Ready-to-run scripts:** see [`../../examples/`](../../examples/) for `docker_run.sh` and `run_inference.sh`.

This is the usual **AI4Science Studio** entry point: use **published checkpoints** from Hugging Face with **HydraGNN** on branch **`Predictive_GFM_2024`**.

## Prerequisites

- Container: `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1`
- Code: clone [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) branch **`Predictive_GFM_2024`**; install with `pip install -e .`
- Weights: download from [`mlupopa/HydraGNN_Predictive_GFM_2024`](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024) -- under `Ensemble_of_models/`, each trial folder has a `config.json` and `.pk` checkpoint

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HG_CHECKPOINT` | Yes | -- | Path to `.pk` checkpoint file |
| `HG_CONFIG` | Yes | -- | Matching `config.json` for the checkpoint |
| `HG_OUTPUT_DIR` | No | `/workspace/results` | Prediction output directory |

## Quick Start

```bash
cd material_science/models/HydraGNN/examples
./docker_run.sh inference
```

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
