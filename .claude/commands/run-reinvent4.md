# Run REINVENT4 transfer learning on an AMD cluster

Guide the user through running REINVENT4 transfer learning for molecular design on AMD GPUs.

> **Research / engineering use only.** Not for clinical or diagnostic use.

## Step 1 — Questionnaire

**Q0. TOML config**
Do you have a transfer learning TOML config file ready? If not, I will help you create one from the template at `healthcare/models/REINVENT4/examples/tl_config.toml.template`.

**Q1. Input SMILES file**
Path to your focused SMILES file for transfer learning (one SMILES per line)?

**Q2. Number of epochs**
Default: 10. How many?

**Q3. Output directory**
Where to write the fine-tuned model and logs?

---

## Step 2 — Setup

### Docker
```bash
cd healthcare/models/REINVENT4/examples
./docker_run.sh
```

This clones REINVENT4, installs deps, and starts the container.

## Step 3 — Run

```bash
docker exec reinvent4 bash /workspace/run_tl.sh
```

Or with custom config:
```bash
docker exec -e CONFIG_FILE=/workspace/my_config.toml reinvent4 bash /workspace/run_tl.sh
```

## Step 4 — Monitor

Check `RESULTS_DIR/tl_run.log` for training progress.

## Expected results

| Task | Notes |
|---|---|
| Transfer learning (10 epochs) | Fine-tuned LSTM prior biased toward target chemical space |

Output: fine-tuned `.model` file in the results directory.

## Arguments

$ARGUMENTS
