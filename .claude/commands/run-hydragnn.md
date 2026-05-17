# Run HydraGNN ensemble inference on an AMD cluster

Guide the user through running HydraGNN predictive GFM inference on AMD GPUs.

## Step 0 — Cluster config check

Check if `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml` exists. If neither exists, run the `/init-cluster` flow first. If a config exists, read it and pre-fill container runtime and SLURM partition/account from saved values.

## Step 1 — Questionnaire

**Q0. Task**
- **Inference** — Load a checkpoint and run predictions
- **Training** — Smaller-scale training (the full GFM pretraining is Frontier-scale)

**Q1. (Inference) Checkpoint and config**
First, auto-discover: run `find <paths.projects> -maxdepth 5 -name "*.pk" 2>/dev/null` and `find <paths.projects> -maxdepth 5 -name "config.json" 2>/dev/null` (substituting `paths.projects` from cluster config) to check for existing checkpoints on shared storage. Present any results to the user. If nothing is found, ask:
- Do you have a `.pk` checkpoint and matching `config.json` from the HF Hub [`mlupopa/HydraGNN_Predictive_GFM_2024`](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024)?
- Options: **Yes, provide paths** / **No, download for me** / **Auto-discovered (use found path)**

**Q2. Output directory**
Where to write predictions? Default: `<paths.projects>/hydragnn-results` (read `paths.projects` from cluster config, never use $HOME for large outputs).

---

## Step 2 — Download weights (if needed)

```bash
pip install huggingface-hub
huggingface-cli download mlupopa/HydraGNN_Predictive_GFM_2024 \
    --include "Ensemble_of_models/gfm_0.229/*" \
    --local-dir ./hydragnn-weights
```

## Step 3 — Launch

### Docker
```bash
cd material_science/models/HydraGNN/examples
HG_CHECKPOINT=/path/to/gfm_0.229_epoch_100.pk \
HG_CONFIG=/path/to/config.json \
./docker_run.sh inference
```

### Manual
```bash
export HG_CHECKPOINT=/path/to/checkpoint.pk
export HG_CONFIG=/path/to/config.json
bash material_science/models/HydraGNN/examples/run_inference.sh
```

## Expected results

The script loads the model, reports success, and provides a Python API entry point for building input graphs and calling `hydragnn.run_prediction()`.

## Arguments

$ARGUMENTS
