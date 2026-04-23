# StormCast: ensemble inference

Run a multi-member probabilistic forecast using StormCast's diffusion residual sampler. Each ensemble member draws independent random residuals at every step, producing spread that reflects convective uncertainty.

Source: [AMD ROCm blog — StormCast ensembles](https://rocm.blogs.amd.com/artificial-intelligence/stormcast-ensembles/README.html)

> **Ready-to-run scripts** live in [`../../examples/`](../../examples/).
> Use [`run_ensemble.py`](../../examples/run_ensemble.py) directly instead of
> copying snippets from this doc.  The SLURM driver is
> [`sbatch_ensemble_amd.sh`](../../examples/sbatch_ensemble_amd.sh).

## How this differs from deterministic inference

| Aspect | Deterministic | Ensemble |
|--------|--------------|----------|
| Runner | `run.deterministic()` | `run.ensemble()` |
| Perturbation | — | `Zero()` (no IC perturbation; spread comes from diffusion sampling) |
| Divergence | — | Independent random residuals per member at each step |
| Output shape | `[time, lead_time, y, x]` | `[ensemble, time, lead_time, y, x]` |
| Analysis products | Point forecast | Member forecasts, ensemble mean, PMM, RMSE |

Same model checkpoint as deterministic inference — no separate download needed.

## Environment

Follow the container setup from [`../inference/README.md`](../inference/README.md), then install the same packages:

```bash
pip install "earth2studio[stormcast]" cartopy
```

## Running ensemble inference

### Script (from the AMD blog)

Download `run-stormcast.py` from the [blog post](https://rocm.blogs.amd.com/artificial-intelligence/stormcast-ensembles/README.html), then:

```bash
python run-stormcast.py --ensemble --ensemble-size 4 2025-08-09T12:00 12
```

Runs 4 members for 12 one-hour steps. Output: `outputs/pred-2025-08-09.zarr`.

### Python API

```python
import earth2studio.run as run
from earth2studio.models.px import StormCast
from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.perturbation import Zero
from datetime import datetime

package = StormCast.load_default_package()
model = StormCast.load_model(package)
data = HRRR()
io = ZarrBackend("model-output.zarr")

starting_datetime = datetime(year=2025, month=8, day=9, hour=12)
nsteps = 12
nensemble = 4

io = run.ensemble(
    time=[starting_datetime],
    nsteps=nsteps,
    nensemble=nensemble,
    prognostic=model,
    data=data,
    io=io,
    perturbation=Zero(),
)
```

> **Note:** `Zero()` applies no perturbation to initial conditions. Ensemble spread comes entirely from the diffusion model sampling different residuals at each step.

## Output structure

```
zarr store
├── ensemble   — member index (0 … nensemble-1)
├── time       — initial timestamp
├── lead_time  — forecast steps as timedeltas
├── hrrr_y     — spatial y coordinate
├── hrrr_x     — spatial x coordinate
└── variables  — refc, t2m, u10m, v10m, …
```

Accessing data:

```python
import xarray as xr

ds = xr.open_zarr("model-output.zarr", consolidated=False)

# Ensemble mean of composite reflectivity at lead step 4
ensemble_mean = ds["refc"][:, 0, 4].mean(axis=0)
```

## Probability Matched Mean (PMM)

Ensemble averaging smooths out extreme values. PMM replaces sorted ensemble-mean values with corresponding percentiles from the pooled ensemble distribution, preserving peak intensities.

```python
import numpy as np
import xarray as xr

def pmm(arr: xr.DataArray) -> xr.DataArray:
    """Probability-matched mean preserving extreme values."""
    n_ens = arr.sizes["ensemble"]
    ens_mean = arr.mean(axis=0)
    np_mean = ens_mean.to_numpy()

    ys, xs = np_mean.shape
    ymat, xmat = np.mgrid[0:ys, 0:xs]
    sorted_ix = np.argsort(np_mean.flatten())[::-1]

    ens_sorted_desc = np.sort(arr.to_numpy().flatten())[::-1]
    ens_sorted = ens_sorted_desc[::n_ens]

    np_updated = np.zeros_like(np_mean)
    np_updated[ymat.flatten()[sorted_ix], xmat.flatten()[sorted_ix]] = ens_sorted

    updated = ens_mean.copy()
    updated.values = np_updated
    return updated
```

## Visualization

### Script (from the AMD blog)

```bash
python plot-stormcast.py outputs/pred-2025-08-09.zarr
```

Produces six-panel plots per timestep: HRRR forecast, HRRR analysis, one ensemble member, PMM, ensemble mean, and RMSE.

### Animation (requires ImageMagick)

```bash
convert -loop 0 -delay 100 scast-2025-08-09-refc-*.jpg animation.gif
```

## Known caveats

- Ensemble spread is generated purely by the diffusion sampler, not initial-condition perturbations. This differs from traditional NWP ensembles.
- The Zarr backend does not support overwriting — delete or rename output before re-running.
- HRRR data availability may lag real-time by ~1 hour; account for this when selecting start times.
