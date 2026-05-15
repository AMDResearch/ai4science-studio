# Run MATEY spatiotemporal modeling on an AMD cluster

Guide the user through training or inference with MATEY on AMD GPUs.

## Step 0 — Cluster config check

Check if `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml` exists. If neither exists, run the `/init-cluster` flow first. If a config exists, read it and pre-fill container runtime and SLURM partition/account from saved values.

## Step 1 — Questionnaire

**Q0. Task**
- **Training** — Train a MATEY model from scratch
- **Inference** — Autoregressive rollout from a trained checkpoint

**Q1. Container runtime**
- **Apptainer** (recommended for HPC — use `build_sif.sh` to create SIF + overlay)
- **Docker** (simpler setup)

**Q2. (Training) Dataset**
Do you have training data in HDF5 format? The JHTDB turbulence demo data is the default starting point.

**Q3. (Training) Multi-GPU**
Single GPU or multi-GPU DDP? If multi-GPU, how many?

**Q4. (Inference) Checkpoint path**
Full path to your trained `.pt` checkpoint?

**Q5. (Inference) Input file**
Path to HDF5 initial condition file?

**Q6. (SLURM) Partition and account**
How should I determine your SLURM partition and account/project?
- **Provide manually** — type your partition and account names
- **Auto-discover** — I will query SLURM to find available partitions and accounts on this cluster

---

## Step 2 — Setup

### Auto-discovery procedures

Run these when the user chose **Auto-discover** for any question. Present the results and let the user confirm or override.

**SLURM partition and account (Q6):**
```bash
sinfo -h -o "%P %G" | grep -i gpu
sacctmgr show associations where user=$USER format=account%30,partition%30 -n
```
Present the available GPU partitions and the user's associated accounts. If multiple exist, ask the user to pick.

After auto-discovery, always confirm the found values with the user before proceeding.

---

### Docker
```bash
cd physics_simulation/models/MATEY/examples
./docker_run.sh train    # or: ./docker_run.sh inference
```

### Apptainer (build SIF + overlay first)
```bash
bash physics_simulation/models/MATEY/examples/build_sif.sh
```

### SLURM
```bash
# Edit #SBATCH directives first — replace YOUR_PARTITION_HERE and YOUR_ACCOUNT_HERE
sbatch physics_simulation/models/MATEY/examples/sbatch_train_amd.sh
```

## Step 3 — Submit

### Training
```bash
export MATEY_YAML=/path/to/config.yaml
export MATEY_DATA_DIR=/path/to/data
export MATEY_EPOCHS=100
bash physics_simulation/models/MATEY/examples/run_train.sh
```

### Inference
```bash
export MATEY_CHECKPOINT=/path/to/checkpoint.pt
export MATEY_CONFIG=/path/to/config.yaml
export MATEY_INPUT=/path/to/initial_condition.h5
export MATEY_STEPS=100
python physics_simulation/models/MATEY/examples/run_inference.py
```

## Arguments

$ARGUMENTS
