# GenCast — Inference Recipe

Run probabilistic ensemble global weather forecasts up to 10 days ahead.

## Prerequisites

- AMD Instinct GPU with ROCm kernel-mode driver (`amdgpu-dkms`)
- Docker
- Free [Copernicus CDS account](https://cds.climate.copernicus.eu/) (for ERA5 initial conditions)

## Step 1 — Configure credentials

```bash
cd earth_science/models/GenCast/examples/
cp env_file.template env_file
# Edit env_file: replace YOUR_CDS_API_KEY_HERE with your real key
```

## Step 2 — Launch the container

```bash
bash examples/docker_run.sh
```

This builds `jaxweather:latest` (shared with PanguWeather — skip the build if already done) and launches an interactive container.

## Step 3 — Run a forecast (inside the container)

```bash
bash /examples/run_inference.sh
```

Default: 10-day (240h) single-member forecast from yesterday's 00Z analysis.

Customize with environment variables:

```bash
MODEL_VARIANT=gencast-0.25 \
DATE=20240101 \
LEAD_TIME=240 \
NUM_ENSEMBLE_MEMBERS=4 \
bash /examples/run_inference.sh
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_VARIANT` | `gencast-0.25` | Checkpoint variant (see below) |
| `DATE` | Yesterday | Forecast start date (`YYYYMMDD`) |
| `TIME` | `0000` | Forecast start time (`HHMM`) |
| `LEAD_TIME` | `240` | Forecast horizon in hours |
| `NUM_ENSEMBLE_MEMBERS` | `1` | Ensemble size (must be multiple of GPU count) |

## Model variants

| Variant | Resolution | Description |
|---|---|---|
| `gencast-0.25` | 0.25° | Full model |
| `gencast-0.25-Oper` | 0.25° | Operational variant |
| `gencast-1.0` | 1.0° | Faster, lower resolution |
| `gencast-1.0-Mini` | 1.0° | Lightweight variant |

## Output

- **Format:** GRIB2 at model resolution
- **File:** `/predictions/<variant>.grib`
- **Variables:** t, z, u, v, w, q (13 levels) + lsm, 2t, sst, msl, 10u, 10v, tp (surface)

Read in Python:

```python
import cfgrib
datasets = cfgrib.open_datasets("gencast-0.25.grib")
```

## Visualize

```bash
python3 /recipe/grib_visualizer.py --input /predictions/gencast-0.25.grib
```

## Memory tuning

For large ensemble runs, source XLA parameters before running to reduce peak memory:

```bash
source /recipe/set_XLA_params.sh
```

This is handled automatically by `run_inference.sh` if the file is present on the recipe mount.

## AMD / ROCm notes

- No code changes required — runs out-of-the-box on ROCm
- XLA memory parameters optional but recommended for multi-member ensembles
- `NUM_ENSEMBLE_MEMBERS` must be a multiple of the number of GPUs visible to the container

## References

- [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html)
- [GenCast paper (Nature, 2025)](https://www.nature.com/articles/s41586-024-08252-9)
- [ai-models framework](https://github.com/ecmwf-lab/ai-models)
