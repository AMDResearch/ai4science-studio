"""Build GHSL built-up fraction on the full PRISM CONUS grid (180x360), fast.

Reads the CONUS window of the cached global GHSL GeoTIFF once, converts m^2 ->
fraction, then vectorized area-averages each GHSL pixel into its PRISM cell via
index binning. Emits urban_conus.npz aligned to PRISM 10-arcmin.
"""
import math, numpy as np, rasterio
from rasterio.windows import from_bounds

TIF = "/home/spannala/Projects/ai4science-studio/studio/backend/assets/_ghsl_builtup_dc.tif"
OUT = "/shared/spannala/orbit2_sr/urban_conus.npz"

PRISM_LAT0, PRISM_DLAT = 24.0, (53.8333 - 24.0) / 179
PRISM_LON0, PRISM_DLON = 235.0, (294.8333 - 235.0) / 359   # 0-360
NROW, NCOL = 180, 360


def main():
    # PRISM CONUS extent in -180..180 lon.
    lat_s, lat_n = PRISM_LAT0 - PRISM_DLAT / 2, PRISM_LAT0 + PRISM_DLAT * (NROW - 1) + PRISM_DLAT / 2
    lon_w = (PRISM_LON0 + 180) % 360 - 180 - PRISM_DLON / 2                 # ~ -125.4
    lon_e = ((PRISM_LON0 + PRISM_DLON * (NCOL - 1) + 180) % 360 - 180) + PRISM_DLON / 2  # ~ -65.0
    with rasterio.open(TIF) as ds:
        win = from_bounds(lon_w, lat_s, lon_e, lat_n, ds.transform)
        arr = ds.read(1, window=win).astype(np.float64)                    # m^2 per pixel
        wt = ds.window_transform(win)
    hh, ww = arr.shape
    # Pixel centers (lon/lat) of the CONUS window.
    px_lon = wt.c + (np.arange(ww) + 0.5) * wt.a
    px_lat = wt.f + (np.arange(hh) + 0.5) * wt.e   # wt.e < 0 (north-up)
    # m^2 -> fraction using per-row pixel ground area.
    px_deg = 30.0 / 3600.0
    m_lat = 111_320.0
    frac = np.empty_like(arr)
    for i in range(hh):
        m_lon = 111_320.0 * math.cos(math.radians(px_lat[i]))
        px_area = (px_deg * m_lat) * (px_deg * m_lon)
        frac[i] = np.clip(arr[i] / max(px_area, 1.0), 0.0, 1.0)
    # Map each GHSL pixel to a PRISM (row,col); accumulate mean per PRISM cell.
    prism_lat = PRISM_LAT0 + PRISM_DLAT * np.arange(NROW)
    prism_lon = np.array([(PRISM_LON0 + PRISM_DLON * c + 180) % 360 - 180 for c in range(NCOL)])
    ri = np.round((px_lat[:, None] - PRISM_LAT0) / PRISM_DLAT).astype(int)         # (hh,1)
    cj = np.round((px_lon[None, :] - prism_lon[0]) / PRISM_DLON).astype(int)       # (1,ww)
    ri = np.clip(ri, 0, NROW - 1).repeat(ww, axis=1)
    cj = np.clip(cj, 0, NCOL - 1).repeat(hh, axis=0)
    flat = ri.ravel() * NCOL + cj.ravel()
    sums = np.bincount(flat, weights=frac.ravel(), minlength=NROW * NCOL)
    cnts = np.bincount(flat, minlength=NROW * NCOL)
    grid = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0).reshape(NROW, NCOL).astype(np.float32)
    print(f"[urban] CONUS built-up fraction: max={grid.max():.3f} mean={grid.mean():.3f}", flush=True)
    # sanity: DC window
    print(f"[urban] DC-window max={grid[78:102,276:300].max():.3f}", flush=True)
    np.savez(OUT, built_up_fraction=grid,
             lat=prism_lat.astype(np.float32), lon=prism_lon.astype(np.float32))
    print(f"[urban] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
