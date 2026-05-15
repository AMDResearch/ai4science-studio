# Run GenCast ensemble forecast on an AMD cluster

Guide the user through running GenCast end-to-end on an AMD cluster.

## Step 0 — Cluster config check

Check if `.cluster-config.yaml` (repo root) or `~/.config/ai4science-studio/cluster.yaml` exists. If neither exists, run the `/init-cluster` flow first. If a config exists, read it and pre-fill container runtime and SLURM partition/account from saved values.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. CDS credentials**
Do you have a Copernicus CDS API key? If not, create a free account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/).

**Q1. Model variant**
Which checkpoint?
- `gencast-0.25` — full 0.25° model (default)
- `gencast-0.25-Oper` — operational variant
- `gencast-1.0` — faster, lower resolution
- `gencast-1.0-Mini` — lightweight

**Q2. Forecast start date**
What date to forecast from? (YYYYMMDD, default: yesterday)

**Q3. Lead time**
How many hours ahead? (default: 240 = 10 days)

**Q4. Ensemble members**
How many ensemble members? (must be a multiple of GPU count, default: 1)

**Q5. Container runtime**
Which container runtime?
- **Docker** (recommended — `docker_run.sh` builds and launches automatically)

---

## Step 2 — Act on answers

Read `earth_science/models/GenCast/model.yaml` for full env var details.

### Configure credentials

```bash
cd earth_science/models/GenCast/examples
cp env_file.template env_file
# Set CDSAPI_KEY in env_file
```

### Launch container

```bash
bash docker_run.sh
```

Builds `jaxweather:latest` (shared with PanguWeather — skip if already built).

### Run forecast (inside container)

```bash
MODEL_VARIANT=<variant> \
DATE=<YYYYMMDD> \
LEAD_TIME=<hours> \
NUM_ENSEMBLE_MEMBERS=<n> \
bash /examples/run_inference.sh
```

## Step 3 — Monitor

Output is written to `/predictions/<variant>.grib`.

```bash
# Visualize
python3 /recipe/grib_visualizer.py --input /predictions/gencast-0.25.grib
```

## Expected results

| Metric | Value |
|--------|-------|
| Resolution | 0.25° (gencast-0.25) or 1.0° (gencast-1.0) |
| Forecast horizon | Up to 10 days (240 hours) |
| Output | GRIB2 with t, z, u, v, w, q (13 levels) + surface variables |

## Notes

- For large ensemble runs, XLA memory parameters are sourced automatically
- `NUM_ENSEMBLE_MEMBERS` must be a multiple of the number of visible GPUs

## Arguments

$ARGUMENTS
