# Run GP-MoLFormer molecule generation on an AMD cluster

> **Research / engineering use only.** Outputs are novel SMILES strings for
> drug-discovery research. Not validated for clinical or therapeutic use.

Guide the user through running GP-MoLFormer on an AMD cluster via SLURM + Apptainer.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q1. SIF path**
Do you have an Apptainer SIF to use? If yes, what is the full path? The validated image is `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1` — if you don't have a SIF, I will generate the pull command. The rocm7.2.2 image also works if you already have it.

**Q2. Work directory**
Where do you want the GP-MoLFormer repo clone, model weights, and output CSV to live on the host? (full path — this directory is reused across runs so the clone is not re-downloaded each time)

**Q3. Generation mode**
Which generation mode do you want?
- **Unconditional** — generates molecules freely with no structural constraint.
- **Scaffold-constrained** — completes molecules around a SMILES fragment you provide (e.g. `c1ccccc1` for a benzene ring).

**Q4. (Scaffold mode only) Scaffold SMILES**
If scaffold mode: what is the SMILES fragment to use as the scaffold?

**Q5. Number of batches**
How many batches of 1000 molecules do you want to generate? (e.g. `1` = 1000 molecules, `5` = 5000 molecules)

**Q6. Output file**
What do you want the output CSV named? (path inside the container, e.g. `/workspace/generated.csv` — maps to `<work_dir>/generated.csv` on the host)

**Q7. Partition and account**
What is your SLURM partition name and account/project name?

---

## Step 2 — Act on answers

**If SIF is missing:**
```bash
apptainer pull docker://rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
```
Tell the user to set `GPMOL_SIF` to the resulting `.sif` path.

**Edit the SBATCH header** in `healthcare/models/GP-MoLFormer/examples/sbatch_inference_amd.sh` to set the user's partition and account (replacing `YOUR_PARTITION_HERE` / `YOUR_ACCOUNT_HERE`).

Note: the script clones `IBM/gp-molformer` and installs `requirements.txt` on first run — internet access from compute nodes is required. Subsequent runs reuse the existing clone in `GPMOL_WORK_DIR`.

## Step 3 — Submit

```bash
export GPMOL_SIF=<path>
export GPMOL_WORK_DIR=<path>
export SCAFFOLD=<smiles>        # omit entirely if unconditional mode
export NUM_BATCHES=<n>
export OUTPUT_FILE=<container-path>
sbatch healthcare/models/GP-MoLFormer/examples/sbatch_inference_amd.sh
```

## Step 4 — Monitor

```bash
squeue -j <job_id>
tail -f gpmolformer-infer-<job_id>.out
```

On success: output CSV is at `<GPMOL_WORK_DIR>/<output_filename>`. The log prints validity and uniqueness stats.

## Expected results (MI300X)

| Mode | Batches | Molecules | Valid | Wall time |
|---|---|---|---|---|
| Unconditional | 1 | 1000 | ~996 (99.6%) | ~41 s |
| Scaffold `c1ccccc1` | 1 | 1000 | ~653 | ~41 s |

## Arguments

$ARGUMENTS
