# Run NeuralGCM inference on an AMD cluster

Guide the user through running NeuralGCM end-to-end on an AMD cluster via Docker or SLURM.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. Container runtime**
Which container runtime?
- **Docker** (recommended — `docker_run.sh` handles image + deps automatically)
- **Apptainer** (for HPC — set `NGC_SIF` to SIF path built from `rocm/dev-ubuntu-22.04:7.0.2-complete`)

**Q1. Checkpoint**
Which NeuralGCM checkpoint?
- `v1/deterministic_0_7_deg.pkl` — 0.7° (~78 km, slowest, sharpest)
- `v1/deterministic_1_4_deg.pkl` — 1.4° (~156 km, default)
- `v1/deterministic_2_8_deg.pkl` — 2.8° (~312 km, fastest)
- `v1/stochastic_1_4_deg.pkl` — stochastic 1.4°
- `v1_precip/stochastic_precip_2_8_deg.pkl` — precipitation
- `v1_precip/stochastic_evap_2_8_deg.pkl` — evaporation

**Q2. Initial condition date**
What date for ERA5 initial conditions? (YYYY-MM-DD, default: 2020-01-01)

**Q3. Forecast steps**
How many 6h forecast steps? (16 = 4 days, 40 = 10 days)

**Q4. (Stochastic only) Seed**
JAX PRNG seed for stochastic checkpoints? (default: 0)

**Q5. Partition and account** (if SLURM)
What is your SLURM partition and account name?

---

## Step 2 — Act on answers

Read `earth_science/models/NeuralGCM/model.yaml` for full env var details.

### Docker path

```bash
cd earth_science/models/NeuralGCM/examples

# 4-day deterministic forecast
./docker_run.sh inference --date 2020-01-01 --steps 16

# 10-day stochastic forecast
./docker_run.sh inference \
    --checkpoint v1/stochastic_1_4_deg.pkl --steps 40 --seed 42

# Interactive shell
./docker_run.sh shell
```

### SLURM path

Edit `#SBATCH` directives in `sbatch_inference_amd.sh` — replace `YOUR_PARTITION_HERE` and `YOUR_ACCOUNT_HERE` with the user's values, then:

```bash
# Bare metal
NGC_DATE=2020-01-01 NGC_STEPS=16 sbatch sbatch_inference_amd.sh

# With Apptainer container
NGC_SIF=/path/to/neuralgcm.sif NGC_STEPS=16 sbatch sbatch_inference_amd.sh
```

## Step 3 — Monitor

```bash
squeue -j <job_id>
tail -f logs/neuralgcm-inference-<job_id>.log
```

## Expected results

| Resolution | MI300X inference time (4 days) | Output |
|-----------|-------------------------------|--------|
| 0.7° | ~110 seconds | NetCDF |
| 1.4° | ~48 seconds | NetCDF |
| 2.8° | ~33 seconds | NetCDF |

No manual downloads required — weights from GCS and ERA5 from public Zarr store are fetched automatically.

## Arguments

$ARGUMENTS
