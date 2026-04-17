# Run GP-MoLFormer molecule generation on an AMD cluster

> **Research / engineering use only.** Outputs are novel SMILES strings for
> drug-discovery research. Not validated for clinical or therapeutic use.

Guide the user through running GP-MoLFormer unconditional or scaffold-constrained generation via SLURM + Apptainer. The script clones `IBM/gp-molformer` and installs deps on first run — internet access from compute nodes required.

## Prerequisites check

Ask the user for any missing values:

1. **`GPMOL_SIF`** — path to an Apptainer SIF. GP-MoLFormer works with an older ROCm image:
   ```bash
   apptainer pull docker://rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.7.1
   export GPMOL_SIF=/path/to/rocm_pytorch_rocm7.0_u22.sif
   ```
   The rocm7.2.2 image also works; rocm7.0 is used here to match the validated baseline.

2. **`GPMOL_WORK_DIR`** (optional) — host directory for the repo clone, weights, and output CSV. Default: the `examples/` directory. Persisted between runs — the clone and HF weights are reused automatically.

3. **Generation mode** — ask the user which mode:

   **A) Unconditional** (default — no scaffold needed):
   ```bash
   # SCAFFOLD is unset
   ```

   **B) Scaffold-constrained** — provide a SMILES fragment:
   ```bash
   export SCAFFOLD="c1ccccc1"   # benzene ring example
   # Any valid SMILES fragment works
   ```

4. **`NUM_BATCHES`** — number of batches of 1000 molecules (default: 1). For more molecules:
   ```bash
   export NUM_BATCHES=5   # generates 5000 molecules
   ```

5. **`OUTPUT_FILE`** — output CSV path inside the container (default: `/workspace/generated.csv`, maps to `GPMOL_WORK_DIR/generated.csv` on the host).

## Partition and account

Remind the user to set `--partition` and `--account` in the SBATCH header (placeholders: `YOUR_PARTITION_HERE` / `YOUR_ACCOUNT_HERE`).

## Submission

```bash
export GPMOL_SIF=<path>
export SCAFFOLD=<smiles-or-unset>
export NUM_BATCHES=<n>
sbatch healthcare/models/GP-MoLFormer/examples/sbatch_inference_amd.sh
```

Note: first run clones `IBM/gp-molformer` and installs `requirements.txt` inside the container (~1–2 min). Subsequent runs reuse the clone.

## Monitoring and output

- `squeue -j <job_id>` to check status
- `tail -f gpmolformer-infer-<job_id>.out` to follow log
- On success: output CSV at `GPMOL_WORK_DIR/generated.csv` (one SMILES per row) with validity and uniqueness stats printed to the log

## Expected results (MI300X)

| Mode | Molecules | Valid | Wall time |
|---|---|---|---|
| Unconditional (1 batch) | 1000 | ~996 (99.6%) | ~41 s |
| Scaffold `c1ccccc1` (1 batch) | 1000 | ~653 | ~41 s |

## Arguments

$ARGUMENTS
