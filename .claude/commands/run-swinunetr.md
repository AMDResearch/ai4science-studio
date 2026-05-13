# Run SwinUNETR medical segmentation on an AMD cluster

Guide the user through training or inference with SwinUNETR on AMD GPUs.

> **Research / engineering use only.** Not for clinical or diagnostic use.

## Step 1 — Questionnaire

**Q0. Task**
- **Training** — Train on NSCLC-Radiomics dataset
- **Inference** — Run optimized inference with a trained checkpoint

**Q1. Container runtime**
- **Docker** (recommended — uses Docker Compose from upstream)
- **Apptainer** (HPC clusters)

**Q2. (Inference only) Checkpoint path**
Full path to your trained `.pth` checkpoint?

**Q3. ROI size**
Default is 96x96x96. MI300X can handle up to 480x480x96. What size?

**Q4. (Training) Max epochs**
Default: 700. How many?

**Q5. (SLURM) Partition and account**
How should I determine your SLURM partition and account/project?
- **Provide manually** — type your partition and account names
- **Auto-discover** — I will query SLURM to find available partitions and accounts on this cluster

---

## Step 2 — Setup

### Auto-discovery procedures

Run these when the user chose **Auto-discover** for any question. Present the results and let the user confirm or override.

**SLURM partition and account (Q5):**
```bash
sinfo -h -o "%P %G" | grep -i gpu
sacctmgr show associations where user=$USER format=account%30,partition%30 -n
```
Present the available GPU partitions and the user's associated accounts. If multiple exist, ask the user to pick.

After auto-discovery, always confirm the found values with the user before proceeding.

---

### Docker
```bash
cd healthcare/models/SwinUNETR/examples
./docker_run.sh train    # or: ./docker_run.sh inference
```

### Apptainer
Edit the `#SBATCH` header in the relevant sbatch script — replace `YOUR_PARTITION_HERE` and `YOUR_ACCOUNT_HERE` with the user's values, then:
```bash
export SU_SIF=<path>
sbatch healthcare/models/SwinUNETR/examples/sbatch_train_amd.sh
# or
export SU_CHECKPOINT=<path>
sbatch healthcare/models/SwinUNETR/examples/sbatch_inference_amd.sh
```

## Step 3 — Monitor

```bash
squeue -j <job_id>
```

## Expected results

| Task | ROI | Time | Notes |
|---|---|---|---|
| Training (700 epochs) | 96x96x96 | ~12 hours | Auto-downloads NSCLC-Radiomics |
| Inference | 96x96x96 | ~1.74 s/case | With AMP + torch.compile |
| Inference | 256x256x128 | ~1.17 s/case | With AMP + torch.compile |

## Arguments

$ARGUMENTS
