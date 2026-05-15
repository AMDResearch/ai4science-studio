# Run SemlaFlow 3D molecular generation on an AMD cluster

Guide the user through generating 3D molecular structures with SemlaFlow on AMD GPUs.

> **Research / engineering use only.** Not for clinical or diagnostic use.

## Step 0 — Cluster config check

Check if `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml` exists. If neither exists, run the `/init-cluster` flow first. If a config exists, read it and pre-fill container runtime and SLURM partition/account from saved values.

## Step 1 — Questionnaire

**Q0. Checkpoint path**
Do you have a pretrained SemlaFlow `.ckpt` file? (Download from Google Drive links in the [upstream repo](https://github.com/rssrwn/semla-flow))

**Q1. Dataset**
- **drugs** (GEOM-Drugs, ~450k drug-like conformers)
- **qm9** (QM9, ~134k small organic molecules)

**Q2. torch.compile**
Enable torch.compile for ~44% speedup? (Recommended: yes)

**Q3. Output path**
Where to write the generated SDF file?

---

## Step 2 — Launch

### Docker
```bash
cd healthcare/models/SemlaFlow/examples
CHECKPOINT=<path> DATASET=<qm9|drugs> ./docker_run.sh inference
```

### Manual
```bash
export CHECKPOINT=<path>
export DATASET=<qm9|drugs>
export USE_COMPILE=1
export NO_EMA=1
bash healthcare/models/SemlaFlow/examples/run_inference.sh
```

## Expected results

| Configuration | Notes |
|---|---|
| drugs + compile + no_ema | Fastest; ~44% speedup from compile, ~8% from no_ema |
| qm9 + compile + no_ema | Smaller molecules, faster per batch |

Output: SDF file with generated 3D molecular structures.

## Arguments

$ARGUMENTS
