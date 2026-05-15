---
name: ai4science-run-models
description: Step-by-step instructions for how Cursor should run any model in AI4Science Studio — read model.yaml, check preflight, select container runtime, launch.
---

# Running models in AI4Science Studio

Use this skill when a user asks to run, execute, or launch any model in the repository.

## Step 0: Check for cluster config

Before anything else, check if a cluster config exists:

```bash
# Check both locations
test -f .cluster-config.yaml && echo "repo-local" || \
test -f ~/.config/ai4science-studio/cluster.yaml && echo "user-level" || \
echo "missing"
```

If **missing**, tell the user: "No cluster configuration found. Running `/init-cluster` to detect your cluster environment." Then run the init-cluster flow (auto-discover GPU, SLURM, containers, paths) before proceeding.

If a config **exists**, read it and use its values as defaults for SLURM partition, account, container runtime, GPU arch, and scratch paths throughout the run. Still confirm with the user if any value is empty or looks stale.

## Step 1: Identify the model

1. Read `models.yaml` at the repo root to find the model by name, slug, or HF id.
2. Read `<domain>/models/<slug>/model.yaml` for full metadata: recipes, env vars, container image, hardware requirements.

## Step 2: Determine the task

Match the user's request to a recipe in `model.yaml`:
- "run inference" / "predict" / "generate" → look for `task: inference`
- "train" / "fine-tune" / "pair-tune" → look for `task: train` or `task: finetune`
- "ensemble" → look for `task: ensemble`

## Step 3: Ask the user for required inputs

Read the `env_vars` section of `model.yaml`. For every variable marked `required: true` that has `default: null`, ask the user for the value. Present the question with the variable's description.

For common patterns, always offer **three options** — provide manually, generate/build, or auto-discover:
- **SIF path** → "Do you have an Apptainer SIF file? (Yes / No — I'll generate the pull command / Auto-discover — I'll search the filesystem)"
- **Overlay** → "Do you have a pre-built overlay? (Yes / No, build one / No, skip overlay / Auto-discover — I'll search for an existing one)"
- **Upstream repo** → "Do you have the repo cloned? (Yes / No — I'll generate the clone command / Auto-discover — I'll search the filesystem)"
- **SLURM partition/account** → "How should I determine your partition/account? (Provide manually / Auto-discover — I'll query SLURM)"

### Auto-discovery procedures

When the user chooses auto-discover, run the appropriate commands and present results for confirmation:

```bash
# SIF files — use $HOME so we find files even when home is not under /home
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*.sif" 2>/dev/null | head -20

# Overlay images
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*<model>*overlay*" 2>/dev/null | head -20

# Upstream repo clones
find "$HOME" /scratch /projects /opt -maxdepth 4 -type d -name "<RepoName>" 2>/dev/null | head -20

# SLURM partitions and accounts
sinfo -h -o "%P %G" | grep -i gpu
sacctmgr show associations where user=$USER format=account%30,partition%30 -n
```

Always confirm discovered values with the user before proceeding.

## Step 4: Check the environment (if possible)

If the user has access to a terminal:
1. Run the model's preflight script: `python examples/preflight_<slug>.py`
2. Check for the SIF file if using Apptainer
3. Verify the overlay exists if one was specified

## Step 5: Configure and launch

### Apptainer/SLURM path
1. Edit `#SBATCH` directives in the sbatch script (partition, account)
2. Set environment variables
3. Submit with `sbatch`

### Docker path
1. Run `./docker_run.sh <task>`
2. Or use `docker exec` for an already-running container

### Bare-metal path
1. Activate the Python environment
2. Run the `run_<task>.sh` or `run_<task>.py` script

## Step 6: Monitor and validate

- For SLURM: `squeue -j <job_id>` and `tail -f <output>.out`
- Check the output file/directory specified by the output env var
- For StormCast: validate zarr with `xarray.open_zarr()`
- For MatterGen: check generated CIF files
- For GP-MoLFormer: check CSV with RDKit validity

## Model-specific notes

### Models with auto-download weights
StormCast, Walrus, GP-MoLFormer — weights download automatically from HF on first run. No manual download step.

### Models requiring manual weight download
ORBIT-2, HydraGNN, MatterGen — user must download from HF Hub first. SemlaFlow — Google Drive. REINVENT4 — bundled with upstream.

### Models trained from scratch
MATEY, SwinUNETR — no pretrained checkpoints distributed; training recipe is the primary entry point.
