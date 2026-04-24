# Run PanguWeather inference on an AMD cluster

Guide the user through running PanguWeather end-to-end on an AMD cluster.

## Step 1 — Questionnaire (ask ALL questions before doing anything)

Ask the user the following questions. Do not assume any defaults. Wait for answers to all questions before proceeding.

**Q0. CDS credentials**
Do you have a Copernicus CDS API key? If not, create a free account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/).

**Q1. Forecast start date and time**
What date and time to forecast from? (YYYYMMDD format, time defaults to 0000)

**Q2. Lead time**
How many hours ahead? (default: 24, max ~168 = 7 days)

**Q3. Container runtime**
Which container runtime?
- **Docker** (recommended — `docker_run.sh` builds and launches automatically)

---

## Step 2 — Act on answers

Read `earth_science/models/PanguWeather/model.yaml` for full env var details.

### Configure credentials

```bash
cd earth_science/models/PanguWeather/examples
cp env_file.template env_file
# Set CDSAPI_KEY in env_file
```

### Launch container

```bash
bash docker_run.sh
```

Builds `jaxweather:latest` (shared with GenCast — skip if already built). Clones `silogen/ai-samples` for the Dockerfiles.

### Run forecast (inside container)

```bash
DATE=<YYYYMMDD> LEAD_TIME=<hours> bash /examples/run_inference.sh
```

## Step 3 — Monitor

Output is written to `/predictions/panguweather.grib`.

```bash
# Visualize
python3 /recipe/grib_visualizer.py --input /predictions/panguweather.grib
```

Produces per-level GIFs under `/predictions/outputs/level_*/`.

## Expected results

| Metric | Value |
|--------|-------|
| Resolution | 0.25° (~28 km) |
| Forecast horizon | Up to 7 days (168 hours) |
| Weights | ~1.1 GB (auto-downloaded on first run) |
| Output | GRIB2 with z, q, t, u, v (13 levels) + msl, 10u, 10v, 2t |

## Notes

- One ONNX backend patch adds ROCm execution provider — handled automatically by the Docker image
- PanguWeather and GenCast share the same `jaxweather:latest` image

## Arguments

$ARGUMENTS
