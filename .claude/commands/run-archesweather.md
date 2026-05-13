# Run ArchesWeather inference or training on an AMD cluster

Guide the user through running ArchesWeather end-to-end on an AMD cluster.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. Task**
Which task do you want to run?
- **Inference** — evaluate pretrained checkpoints on ERA5 test data
- **Training** — pretrain or fine-tune ArchesWeather / ArchesWeatherGen

**Q1. Container runtime**
Which container runtime?
- **Docker** (recommended for interactive workstations)
- **Apptainer** (recommended for HPC — set `AW_SIF` to SIF path)

**Q2. (Inference) Model variant**
Which model checkpoint?
- `archesweather-m-seed0` through `seed3` (deterministic)
- `archesweathergen` (generative, flow matching)

**Q3. (Training) Phase and model**
- Model: `archesweather` or `archesweathergen`
- Phase: `pretrain` or `finetune`
- If fine-tuning, what is the pretrained checkpoint path?

**Q4. ERA5 data path**
Where is the ERA5 dataset? (~735 GB for training, ~35 GB for one test year)

**Q5. (Apptainer only) SIF path**
Do you have an Apptainer SIF built from the silogen/ai-samples geoarches-training Dockerfile?
- **Yes** — provide the full path
- **No** — I will generate the build/pull command
- **Auto-discover** — I will search the filesystem for existing `.sif` files

**Q6. Partition and account**
How should I determine your SLURM partition and account/project? (if using SLURM)
- **Provide manually** — type your partition and account names
- **Auto-discover** — I will query SLURM to find available partitions and accounts on this cluster

---

## Step 2 — Act on answers

### Auto-discovery procedures

Run these when the user chose **Auto-discover** for any question. Present the results and let the user confirm or override.

**SIF files (Q5):**
```bash
find "$HOME" /scratch /projects /opt -maxdepth 4 -name "*.sif" 2>/dev/null | head -20
```
Use `$HOME` (not `/home`) so the search works when the home directory is under a non-standard prefix (e.g. `/shared/prerelease/home/…`).
Filter results for SIF names containing `geoarches` or `rocm`. Verify with `apptainer inspect <sif>` if multiple candidates.

**SLURM partition and account (Q6):**
```bash
sinfo -h -o "%P %G" | grep -i gpu
sacctmgr show associations where user=$USER format=account%30,partition%30 -n
```
Present the available GPU partitions and the user's associated accounts. If multiple exist, ask the user to pick.

After auto-discovery, always confirm the found values with the user before proceeding.

---

Read `earth_science/models/ArchesWeather/model.yaml` for full env var details.

### Docker path

```bash
cd earth_science/models/ArchesWeather/examples
bash docker_run.sh

# Inside container:
# Inference
bash /examples/run_inference.sh

# Training
bash /examples/run_train.sh
```

### Apptainer / SLURM path

Edit the `#SBATCH` header in the relevant sbatch script — replace `YOUR_PARTITION_HERE` and `YOUR_ACCOUNT_HERE` with the user's values, then:

```bash
# Inference
AW_SIF=<path> sbatch earth_science/models/ArchesWeather/examples/sbatch_inference_amd.sh

# Training
AW_SIF=<path> sbatch earth_science/models/ArchesWeather/examples/sbatch_train_amd.sh
```

## Step 3 — Monitor

```bash
squeue -j <job_id>
tail -f archesweather-*-<job_id>.out
```

## Expected results

| Task | MI300X time | VRAM | Output |
|------|-------------|------|--------|
| Inference (1 year) | ~10–30 min | up to 192 GB | Predictions in `results/predictions/` |
| Training (250K steps) | ~24–48 hours | up to 192 GB | Checkpoints in `checkpoints/` |

## Arguments

$ARGUMENTS
