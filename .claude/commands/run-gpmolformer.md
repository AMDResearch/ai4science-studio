# Run GP-MoLFormer molecule generation on an AMD cluster

> **Research / engineering use only.** Outputs are novel SMILES strings for
> drug-discovery research. Not validated for clinical or therapeutic use.

Guide the user through running GP-MoLFormer on an AMD cluster via SLURM.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. Container runtime**
Which container runtime do you want to use?
- **Apptainer** (recommended for HPC — supports overlays, `--rocm` flag for GPU)
- **Docker** (simpler setup, no overlay needed, but no MPI support and env vars must be appended not replaced)

**Q1. (Apptainer only) SIF path**
Do you have an Apptainer SIF to use? The validated image is `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1` (the rocm7.2.2 image also works).
- **Yes** — provide the full path
- **No** — I will generate the pull command
- **Auto-discover** — I will search the filesystem for existing ROCm PyTorch `.sif` files

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
How should I determine your SLURM partition and account/project?
- **Provide manually** — type your partition and account names
- **Auto-discover** — I will query SLURM to find available partitions and accounts on this cluster

---

## Step 2 — Act on answers

### Auto-discovery procedures

Run these when the user chose **Auto-discover** for any question. Present the results and let the user confirm or override.

**SIF files (Q1):**
```bash
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*.sif" 2>/dev/null | head -20
```
Use `$HOME` (not `/home`) so the search works when the home directory is under a non-standard prefix (e.g. `/shared/prerelease/home/…`).
Filter results for SIF names containing `rocm` or `pytorch`. Verify with `apptainer inspect <sif>` if multiple candidates.

**SLURM partition and account (Q7):**
```bash
sinfo -h -o "%P %G" | grep -i gpu
sacctmgr show associations where user=$USER format=account%30,partition%30 -n
```
Present the available GPU partitions and the user's associated accounts. If multiple exist, ask the user to pick.

After auto-discovery, always confirm the found values with the user before proceeding.

---

### Apptainer path

**If SIF is missing:**
```bash
apptainer pull docker://rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
```
Tell the user to set `GPMOL_SIF` to the resulting `.sif` path.

**Edit the SBATCH header** in `healthcare/models/GP-MoLFormer/examples/sbatch_inference_amd.sh` to set the user's partition and account (replacing `YOUR_PARTITION_HERE` / `YOUR_ACCOUNT_HERE`).

Note: the script clones `IBM/gp-molformer` and installs deps on first run — internet access from compute nodes is required. Subsequent runs reuse the existing clone in `GPMOL_WORK_DIR`.

**Important:** IBM/gp-molformer has no `requirements.txt`. The Apptainer script uses the py3.10 image where `transformers==4.32.1` and its tokenizers dep have prebuilt wheels. If using the py3.12 image instead, use the Docker script or pin `transformers>=4.36,<4.41`.

### Docker path

**Edit the SBATCH header** in `healthcare/models/GP-MoLFormer/examples/sbatch_inference_docker.sh` to set the user's partition and account.

Docker-specific notes:
- Uses `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` (py3.12)
- Installs deps explicitly (no `requirements.txt` exists in upstream repo): `accelerate`, `datasets`, `networkx`, `pandas`, `peft`, `scikit-learn`, `transformers>=4.36,<4.41`, `rdkit`
- The `transformers` pin avoids two issues: `tokenizers 0.13.x` has no cp312 wheel (fixed by >=4.36), and `transformers.onnx` was removed in >=4.41 (MoLFormer's HF config imports it)
- No SIF or overlay needed

## Step 3 — Submit

### Apptainer
```bash
export GPMOL_SIF=<path>
export GPMOL_WORK_DIR=<path>
export SCAFFOLD=<smiles>        # omit entirely if unconditional mode
export NUM_BATCHES=<n>
export OUTPUT_FILE=<container-path>
sbatch healthcare/models/GP-MoLFormer/examples/sbatch_inference_amd.sh
```

### Docker
```bash
export GPMOL_WORK_DIR=<path>
export SCAFFOLD=<smiles>        # omit entirely if unconditional mode
export NUM_BATCHES=<n>
export OUTPUT_FILE=<container-path>
sbatch healthcare/models/GP-MoLFormer/examples/sbatch_inference_docker.sh
```

## Step 4 — Monitor

```bash
squeue -j <job_id>
tail -f gpmolformer-*-<job_id>.out
```

On success: output CSV is at `<GPMOL_WORK_DIR>/<output_filename>`. The log prints validity and uniqueness stats.

## Expected results (MI300X)

| Mode | Batches | Molecules | Valid | Wall time |
|---|---|---|---|---|
| Unconditional | 1 | 1000 | ~995 (99.5%) | ~90 s |
| Scaffold `c1ccccc1` | 1 | 1000 | ~653 | ~90 s |

## Arguments

$ARGUMENTS
