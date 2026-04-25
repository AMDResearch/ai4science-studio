# Run Aurora inference on an AMD cluster

Guide the user through running Aurora (0.1° resolution) end-to-end on an AMD cluster.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. CDS credentials**
Do you have a Copernicus CDS API key? If not, create a free account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/).

**Q1. Forecast start date and time**
What date and time do you want to forecast from? (YYYYMMDD format, time defaults to 0000)

**Q2. Lead time**
How many hours ahead do you want to forecast? (default: 24, in 6h rollout steps, max ~120 for 5 days)

**Q3. Container runtime**
Which container runtime?
- **Docker** (recommended — `docker_run.sh` builds and launches automatically)

---

## Step 2 — Act on answers

Read `earth_science/models/Aurora/model.yaml` for full env var details.

### Configure credentials

```bash
cd earth_science/models/Aurora/examples
cp env_file.template env_file
# Set CDSAPI_KEY in env_file
```

### Launch container

```bash
bash docker_run.sh
```

This builds `pytorchweather:latest` (a PyTorch image, separate from the JAX image used by PanguWeather/GenCast) and launches an interactive container.

### Run forecast (inside container)

```bash
DATE=<YYYYMMDD> LEAD_TIME=<hours> bash /examples/run_inference.sh
```

## Step 3 — Monitor

Output is written to `/predictions/aurora.grib`.

```bash
# Visualize
python3 /recipe/grib_visualizer.py --input /predictions/aurora.grib
```

## Expected results

| Metric | Value |
|--------|-------|
| Resolution | 0.1° (~11 km) — highest of AMD-validated weather models |
| VRAM | Up to 192 GB HBM3 (MI300X well-suited for 0.1° memory footprint) |
| Output | GRIB2 with z, u, v, t, q (13 levels) + 2t, 10u, 10v, msl |

## Arguments

$ARGUMENTS
