#!/usr/bin/env python3
"""NeuralGCM deterministic or stochastic inference from ERA5 initial conditions.

Fetches ERA5 data from public GCS, loads the selected checkpoint, runs a
multi-step forecast, and writes output as a NetCDF file.

Usage
-----
    python run_inference.py [options]

Options
-------
--checkpoint PATH   GCS path suffix, e.g. v1/deterministic_1_4_deg.pkl
                    (default: v1/deterministic_1_4_deg.pkl)
--date DATE         Initial condition date, YYYY-MM-DD (default: 2020-01-01)
--hour HOUR         Initial condition hour 0–23 UTC (default: 0)
--steps N           Number of forecast steps at 6h cadence (default: 16 = 4 days)
--output PATH       Output NetCDF path (default: outputs/neuralgcm-<ckpt>-<date>.nc)
--seed N            JAX PRNG seed for stochastic checkpoints (default: 0)

Examples
--------
    # 4-day deterministic forecast at 1.4°
    python run_inference.py --checkpoint v1/deterministic_1_4_deg.pkl --steps 16

    # 10-day stochastic forecast at 1.4°
    python run_inference.py --checkpoint v1/stochastic_1_4_deg.pkl --steps 40 --seed 42
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import gcsfs
import jax
import jax.numpy as jnp
import neuralgcm
import xarray as xr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuralGCM inference")
    p.add_argument("--checkpoint", default="v1/deterministic_1_4_deg.pkl",
                   help="GCS path suffix under gs://neuralgcm/models/")
    p.add_argument("--date", default="2020-01-01", help="Initial condition date YYYY-MM-DD")
    p.add_argument("--hour", type=int, default=0, help="Initial condition hour UTC")
    p.add_argument("--steps", type=int, default=16, help="Number of 6h forecast steps")
    p.add_argument("--output", default=None, help="Output NetCDF path")
    p.add_argument("--seed", type=int, default=0, help="JAX PRNG seed (stochastic models)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ERA5_ZARR = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)

NEURALGCM_GCS_PREFIX = "gs://neuralgcm/models/"


def load_checkpoint(ckpt_suffix: str):
    fs = gcsfs.GCSFileSystem(token="anon")
    gcs_path = f"{NEURALGCM_GCS_PREFIX}{ckpt_suffix}"
    print(f"Loading checkpoint: {gcs_path}")
    with fs.open(gcs_path, "rb") as f:
        return pickle.load(f)


def load_era5_slice(date: str, hour: int, n_steps: int) -> xr.Dataset:
    """Open a time slice of the public ARCO-ERA5 Zarr store."""
    import pandas as pd

    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(ERA5_ZARR, gcs=fs)
    ds = xr.open_zarr(store, consolidated=True, chunks=None)

    start = pd.Timestamp(f"{date}T{hour:02d}:00")
    # NeuralGCM needs the initial state plus forcing fields at each step
    times = pd.date_range(start, periods=n_steps + 1, freq="6h")
    return ds.sel(time=times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"JAX devices: {jax.devices()}")

    # Load model
    ckpt = load_checkpoint(args.checkpoint)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    print(f"Model loaded  resolution={model.data_coords.horizontal.latitude.shape}")

    # Load ERA5 initial conditions
    print("Fetching ERA5 initial conditions …")
    era5 = load_era5_slice(args.date, args.hour, args.steps)

    # Prepare inputs and forcings expected by NeuralGCM
    inputs, forcings = model.inputs_from_xarray(era5.isel(time=0)), \
                       model.forcings_from_xarray(era5)

    # Run forecast
    rng = jax.random.PRNGKey(args.seed)
    print(f"Running {args.steps} forecast steps ({args.steps * 6}h) …")
    predictions = model.unroll(inputs, forcings, steps=args.steps, rng_key=rng)

    # Convert to xarray and save
    pred_ds = model.data_to_xarray(predictions, times=era5.time.values[1:])

    # Output path
    if args.output is None:
        ckpt_tag = args.checkpoint.replace("/", "_").replace(".pkl", "")
        out_path = Path("outputs") / f"neuralgcm-{ckpt_tag}-{args.date}.nc"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_ds.to_netcdf(out_path)
    print(f"Forecast written to {out_path}")


if __name__ == "__main__":
    main()
