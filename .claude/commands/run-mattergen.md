# Run MatterGen crystal generation on an AMD cluster

Guide the user through running MatterGen end-to-end on an AMD cluster via SLURM or Docker.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers before proceeding.

**Q0. Container runtime**
- **Apptainer** (recommended for HPC)
- **Docker** (simpler setup)

**Q1. Task**
- **Inference** (generate novel crystal structures)
- **Training** (train/fine-tune MatterGen from scratch)

**Q2. (Apptainer only) SIF path**
Do you have an Apptainer SIF built from `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1`? If yes, full path? If no, I will generate the pull command.

**Q3. (Inference) Generation mode**
- **Unconditional** — generate without constraints
- **Property-conditioned** — specify properties and guidance factor

**Q4. (Inference, conditioned) Properties**
What property conditioning dict? E.g. `{"chemical_system": "Li-Fe-O", "energy_above_hull": 0.0}`

**Q5. Partition and account**
What is your SLURM partition name and account/project name?

---

## Step 2 — Act on answers

**If SIF is missing:**
```bash
apptainer pull docker://rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
```

**Edit the SBATCH header** in the relevant script — replace `YOUR_PARTITION_HERE` and `YOUR_ACCOUNT_HERE` with the user's values.

## Step 3 — Submit

### Inference (Apptainer)
```bash
export MG_SIF=<path>
export MG_PRETRAINED_NAME=<name>       # default: mattergen_base
export MG_BATCH_SIZE=<n>               # default: 16
export MG_NUM_BATCHES=<n>              # default: 1
export MG_PROPERTIES=<dict>            # empty for unconditional
export MG_GUIDANCE_FACTOR=<n>          # default: 2.0
sbatch material_science/models/MatterGen/examples/sbatch_inference_amd.sh
```

### Training (Docker)
```bash
cd material_science/models/MatterGen/examples
./docker_run.sh train
```

## Step 4 — Monitor

```bash
squeue -j <job_id>
tail -f mattergen-*-<job_id>.out
```

## Expected results

| Task | Time (single MI300X) | Output |
|---|---|---|
| Inference (16 structures) | ~1 min | CIF files in output dir |
| Training (900 epochs) | ~15 hours | Checkpoint files |

## Arguments

$ARGUMENTS
