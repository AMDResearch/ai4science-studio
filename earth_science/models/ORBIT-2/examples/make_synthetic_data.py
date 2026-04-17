#!/usr/bin/env python3
"""Generate a minimal synthetic ORBIT-2 dataset for smoke-testing.

Produces the exact directory layout and NumPy file formats that ORBIT-2's
climate_learn data loader expects, without requiring real ERA5 / PRISM / DAYMET
data.  The spatial grids, variable names, and tensor shapes are all consistent
with the ``interm_8m.yaml`` PRISM downscaling config.

Output layout (written to --out-dir):
  <out-dir>/
    low_res/          # inp_root_dir pointed at by the YAML
      normalize_mean.npz
      normalize_std.npz
      lat.npy
      lon.npy
      train/
        2020_0.npz
      val/
        2021_0.npz
      test/
        2022_0.npz
        climatology.npz
    high_res/         # out_root_dir pointed at by the YAML
      (same structure)

File sizes (defaults): ~2 MB total.

Usage:
    python3 make_synthetic_data.py --out-dir /path/to/orbit2-synthetic
    python3 make_synthetic_data.py --out-dir /path/to/orbit2-synthetic --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Variable lists — must match exactly what the YAML config specifies under
# dict_in_variables['PRISM'] and dict_out_variables['PRISM'].
# ---------------------------------------------------------------------------
IN_VARS = [
    "land_sea_mask",
    "orography",
    "lattitude",
    "landcover",
    "total_precipitation_24hr",
    "2m_temperature_min",
    "2m_temperature_max",
]

OUT_VARS = [
    "total_precipitation_24hr",
    "2m_temperature_min",
    "2m_temperature_max",
]

# Precipitation variable — uses LogTransform, so normalize stats are still
# written but the loader skips reading them for this var.
PRECIP_VARS = {"total_precipitation_24hr"}

# Realistic-ish value ranges (used to seed mean/std and plausible data).
# Temperature in Kelvin, precipitation in mm/day, others dimensionless.
VAR_STATS: dict[str, tuple[float, float]] = {
    "land_sea_mask":            (0.5,  0.5),
    "orography":                (500.0, 300.0),
    "lattitude":                (40.0,  10.0),
    "landcover":                (5.0,   3.0),
    "total_precipitation_24hr": (2.0,   4.0),
    "2m_temperature_min":       (270.0, 15.0),
    "2m_temperature_max":       (285.0, 15.0),
}

# ---------------------------------------------------------------------------
# Grid dimensions.  low_res is 10-arcmin PRISM CONUS (≈180×260 real).
# We use a small proxy that keeps the 4× upscale ratio exact and stays small.
#   low-res:  20 lat × 40 lon  →  ~0.3 MB per variable per shard
#   high-res: 80 lat × 160 lon →  ~5× larger per variable per shard
# Total with default T=16 across all splits: ≈ 2 MB.
# ---------------------------------------------------------------------------
LO_H, LO_W = 20, 40       # low-res spatial dims
HI_H, HI_W = 80, 160      # high-res spatial dims (4× each)
T_PER_SHARD = 16           # timesteps per shard file


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_lat_lon(h: int, w: int, lat_range=(25.0, 50.0), lon_range=(-125.0, -65.0)) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(lat_range[0], lat_range[1], h, dtype=np.float32)
    lon = np.linspace(lon_range[0], lon_range[1], w, dtype=np.float32)
    return lat, lon


def make_normalize_stats(variables: list[str]) -> tuple[dict, dict]:
    """Build normalize_mean / normalize_std dicts with shape (1,) per variable."""
    means, stds = {}, {}
    for var in variables:
        mu, sigma = VAR_STATS.get(var, (0.0, 1.0))
        means[var] = np.array([mu], dtype=np.float32)
        stds[var]  = np.array([sigma if sigma > 0 else 1.0], dtype=np.float32)
    return means, stds


def make_shard(
    variables: list[str],
    h: int,
    w: int,
    t: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Create one npz shard.  Shape per variable: (T, 1, H, W) float32."""
    data = {}
    for var in variables:
        mu, sigma = VAR_STATS.get(var, (0.0, 1.0))
        arr = rng.normal(mu, sigma, size=(t, 1, h, w)).astype(np.float32)
        # Precipitation must be non-negative (log transform will be applied)
        if var in PRECIP_VARS:
            arr = np.abs(arr)
        data[var] = arr
    return data


def make_climatology(variables: list[str], h: int, w: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Climatology: shape (1, H, W) per variable.  Loader squeezes axis=0 → (H, W)."""
    clim = {}
    for var in variables:
        mu, sigma = VAR_STATS.get(var, (0.0, 1.0))
        clim[var] = rng.normal(mu, sigma * 0.1, size=(1, h, w)).astype(np.float32)
    return clim


def write_dataset(root: str, variables: list[str], out_variables: list[str],
                  h: int, w: int, rng: np.random.Generator) -> None:
    """Write all files for one resolution directory (low_res or high_res)."""
    os.makedirs(root, exist_ok=True)

    # lat / lon grids
    lat, lon = make_lat_lon(h, w)
    np.save(os.path.join(root, "lat.npy"), lat)
    np.save(os.path.join(root, "lon.npy"), lon)

    # normalization stats
    means, stds = make_normalize_stats(variables)
    np.savez(os.path.join(root, "normalize_mean.npz"), **means)
    np.savez(os.path.join(root, "normalize_std.npz"),  **stds)

    # shards + climatology: train / val / test — one shard file per split.
    # get_climatology() is called for all three splits (loaders.py:116,141,178)
    # so climatology.npz must exist in every split directory.
    for split, year in (("train", 2020), ("val", 2021), ("test", 2022)):
        split_dir = os.path.join(root, split)
        os.makedirs(split_dir, exist_ok=True)
        shard = make_shard(variables, h, w, T_PER_SHARD, rng)
        np.savez(os.path.join(split_dir, f"{year}_0.npz"), **shard)
        clim = make_climatology(out_variables, h, w, rng)
        np.savez(os.path.join(split_dir, "climatology.npz"), **clim)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(root)
        for f in fnames
    )
    print(f"  {root}: {total / 1024:.0f} KB ({len(list(os.walk(root)))} dirs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="./orbit2-synthetic",
        help="Root output directory (default: ./orbit2-synthetic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility (default: 0)",
    )
    args = parser.parse_args()

    rng = make_rng(args.seed)
    out = os.path.abspath(args.out_dir)
    print(f"Writing synthetic ORBIT-2 dataset to: {out}")
    print(f"  low_res  grid: {LO_H}×{LO_W}  |  {T_PER_SHARD} timesteps/shard")
    print(f"  high_res grid: {HI_H}×{HI_W}  |  {T_PER_SHARD} timesteps/shard")
    print(f"  in_vars:  {IN_VARS}")
    print(f"  out_vars: {OUT_VARS}")
    print()

    write_dataset(
        root=os.path.join(out, "low_res"),
        variables=IN_VARS,
        out_variables=OUT_VARS,
        h=LO_H, w=LO_W,
        rng=rng,
    )
    write_dataset(
        root=os.path.join(out, "high_res"),
        variables=OUT_VARS,
        out_variables=OUT_VARS,
        h=HI_H, w=HI_W,
        rng=rng,
    )

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(out)
        for f in fnames
    )
    print(f"\nTotal dataset size: {total / 1024:.0f} KB")
    print("\nTo point ORBIT-2 at this data, use a config with:")
    print(f"  low_res_dir:  {{'SYNTH': '{os.path.join(out, 'low_res')}'}}")
    print(f"  high_res_dir: {{'SYNTH': '{os.path.join(out, 'high_res')}'}}")


if __name__ == "__main__":
    main()
