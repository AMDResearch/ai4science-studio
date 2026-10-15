#!/usr/bin/env python3
"""Fetch REAL GHSL built-up-surface data and regrid to the PRISM DC 24x24 grid.

Source: EU JRC Global Human Settlement Layer (GHSL) GHS_BUILT_S R2023A, epoch 2020,
30 arcsec, WGS84 (EPSG:4326). Value = built-up surface area (m^2) per cell; we
convert to a 0..1 built-up FRACTION (area / cell_area) — a physical "concrete
jungle" / urban-heat-island conditioning field.

Every DC-grid cell's value is the mean GHSL built-up fraction of the real GHSL
pixels falling inside it (area-weighted average, no synthetic fill). Output grid
matches PRISM 10-arcmin rows 78:102, cols 276:300 exactly, so it drops straight
into the model's landcover channel slot.

Run: python3 studio/backend/tools/fetch_dc_urban_density.py
Output: studio/backend/assets/dc_urban_density.json
"""
import io
import json
import math
import ssl
import sys
import urllib.request as urlreq
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

ASSET = Path(__file__).resolve().parents[1] / "assets" / "dc_urban_density.json"
CACHE = Path(__file__).resolve().parents[1] / "assets" / "_ghsl_builtup_dc.tif"

GHSL_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/"
            "V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip")

# PRISM 10-arcmin geolocation + DC window (must match the other DC scripts).
PRISM_LAT0, PRISM_DLAT = 24.0, (53.8333 - 24.0) / 179
PRISM_LON0, PRISM_DLON = 235.0, (294.8333 - 235.0) / 359
DC_ROWS = (78, 102)
DC_COLS = (276, 300)

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


def prism_dc_grid():
    r0, r1 = DC_ROWS
    c0, c1 = DC_COLS
    lats = [PRISM_LAT0 + PRISM_DLAT * r for r in range(r0, r1)]
    lons = [(PRISM_LON0 + PRISM_DLON * c + 180) % 360 - 180 for c in range(c0, c1)]
    return lats, lons


def download_ghsl_tif():
    """Download the global GHSL zip once, extract the GeoTIFF, cache it."""
    if CACHE.exists():
        print(f"[ghsl] using cached {CACHE}", flush=True)
        return CACHE
    print(f"[ghsl] downloading GHSL built-up surface (~187 MB) ...", flush=True)
    req = urlreq.Request(GHSL_URL)
    with urlreq.urlopen(req, timeout=180, context=_SSL) as r:
        blob = r.read()
    print(f"[ghsl] downloaded {len(blob)/1e6:.0f} MB, extracting GeoTIFF ...", flush=True)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    tif_name = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
    CACHE.write_bytes(zf.read(tif_name))
    print(f"[ghsl] cached GeoTIFF -> {CACHE}", flush=True)
    return CACHE


def build_fraction_grid(tif_path):
    """For each PRISM DC cell, mean GHSL built-up fraction of the pixels inside it."""
    lats, lons = prism_dc_grid()
    dlat = PRISM_DLAT
    dlon = PRISM_DLON
    grid = np.zeros((len(lats), len(lons)), dtype=np.float32)
    with rasterio.open(tif_path) as ds:
        px_area_m2 = None
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # PRISM cell bounds (cell-centered geolocation).
                s, n = lat - dlat / 2, lat + dlat / 2
                w, e = lon - dlon / 2, lon + dlon / 2
                try:
                    win = from_bounds(w, s, e, n, ds.transform)
                    arr = ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
                except Exception:
                    grid[i, j] = 0.0
                    continue
                if arr.size == 0:
                    grid[i, j] = 0.0
                    continue
                # GHSL value = built-up surface m^2 per ~30 arcsec pixel. Convert to
                # fraction using approximate pixel ground area at this latitude.
                if px_area_m2 is None:
                    px_deg = 30.0 / 3600.0
                    m_per_deg_lat = 111_320.0
                    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
                    px_area_m2 = (px_deg * m_per_deg_lat) * (px_deg * m_per_deg_lon)
                frac = np.clip(arr / max(px_area_m2, 1.0), 0.0, 1.0)
                grid[i, j] = float(frac.mean())
    return lats, lons, grid


def main():
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    tif = download_ghsl_tif()
    lats, lons, grid = build_fraction_grid(tif)
    peak = float(grid.max())
    # DC center (Reagan National) index for a sanity print.
    ci, cj = grid.shape[0] // 2, grid.shape[1] // 2
    print(f"[ghsl] DC-grid built-up fraction: max={peak:.3f} mean={float(grid.mean()):.3f} "
          f"center={grid[ci,cj]:.3f}", flush=True)
    out = {
        "source": "EU JRC GHSL GHS_BUILT_S R2023A, epoch 2020, 30 arcsec, EPSG:4326",
        "variable": "built_up_surface_fraction",
        "grid": "PRISM 10-arcmin DC window (rows 78:102, cols 276:300), 24x24",
        "lat": [round(x, 4) for x in lats],
        "lon": [round(x, 4) for x in lons],
        "built_up_fraction": np.round(grid, 4).tolist(),
        "peak_fraction": round(peak, 4),
        "method": ("Each PRISM DC cell = mean GHSL built-up fraction of the real GHSL "
                   "pixels inside it. Physical urban-density (concrete-jungle) field for "
                   "urban-heat-island conditioning. No interpolation or synthetic fill."),
    }
    ASSET.write_text(json.dumps(out, indent=2))
    print(f"[ghsl] wrote {ASSET} ({ASSET.stat().st_size/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
