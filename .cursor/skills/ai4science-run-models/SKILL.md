---
name: ai4science-run-models
description: Step-by-step instructions for how Cursor should run any model in AI4Science Studio — read model.yaml, check preflight, select container runtime, launch.
---

# Running models in AI4Science Studio

Use this skill when a user asks to run, execute, or launch any model in the repository.

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

For common patterns:
- **SIF path** → Ask: "Do you have an Apptainer SIF file? If not, I can generate the pull command."
- **Overlay** → Ask: "Do you have a pre-built overlay? If not, would you like to build one (saves time on future runs) or skip it?"
- **SLURM partition/account** → Ask: "What is your SLURM partition and account name?"

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
