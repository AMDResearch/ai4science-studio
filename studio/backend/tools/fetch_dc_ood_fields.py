#!/usr/bin/env python3
"""Fetch REAL Open-Meteo ERA5 daily fields on the exact PRISM DC 24x24 grid.

These are the DYNAMIC inputs (tmax, tmin, precip) for a true out-of-distribution
test: PRISM-trained ORBIT-2 applied to independent Open-Meteo DC heatwave data it
never saw. The 4 STATIC channels (land-sea mask, orography, latitude, landcover)
come from PRISM's DC crop at run time (time-invariant geography). Nothing is
interpolated or synthesized — every grid point is fetched individually.

Grid matches PRISM 10-arcmin rows 78:102, cols 276:300 (24x24), so it coarsens
4x cleanly to 6x6 for the ORBIT-2 super-resolution model.

Run: python3 studio/backend/tools/fetch_dc_ood_fields.py
Output: studio/backend/assets/dc_ood_fields.json
"""
import json
import sys
import time
from pathlib import Path

# Reuse the audited fetch/retry/SSL machinery from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_dc_era5 import fetch, EVENTS  # noqa: E402

ASSET = Path(__file__).resolve().parents[1] / "assets" / "dc_ood_fields.json"

# PRISM 10-arcmin geolocation + DC window (must match orbit2 harness exactly).
PRISM_LAT0, PRISM_DLAT = 24.0, (53.8333 - 24.0) / 179
PRISM_LON0, PRISM_DLON = 235.0, (294.8333 - 235.0) / 359
DC_ROWS = (78, 102)   # 24 rows
DC_COLS = (276, 300)  # 24 cols

DAILY_VARS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]


def prism_grid():
    """Real lat/lon of every PRISM DC-window cell (lon converted to -180..180)."""
    r0, r1 = DC_ROWS
    c0, c1 = DC_COLS
    lats = [round(PRISM_LAT0 + PRISM_DLAT * r, 4) for r in range(r0, r1)]
    lons = [round((PRISM_LON0 + PRISM_DLON * c + 180) % 360 - 180, 4) for c in range(c0, c1)]
    return lats, lons


def fetch_point_daily(lat, lon, date):
    """Return (tmax_c, tmin_c, precip_mm) for one grid point, or (None,)*3."""
    data = fetch({
        "latitude":   round(lat, 4),
        "longitude":  round(lon, 4),
        "start_date": date,
        "end_date":   date,
        "daily":      ",".join(DAILY_VARS),
        "models":     "era5",
        "timezone":   "UTC",
    })
    d = data.get("daily", {})
    def first(key):
        vals = d.get(key) or []
        return vals[0] if vals and vals[0] is not None else None
    return first("temperature_2m_max"), first("temperature_2m_min"), first("precipitation_sum")


def fetch_event_fields(key, ev):
    date = ev["date"]
    lats, lons = prism_grid()
    n = len(lats) * len(lons)
    print(f"\n[{key}] Fetching {len(lats)}x{len(lons)}={n} real ERA5 daily points "
          f"({date}) on the PRISM DC grid ...", flush=True)

    tmax, tmin, precip = [], [], []
    done = 0
    for lat in lats:
        rmax, rmin, rpre = [], [], []
        for lon in lons:
            try:
                tx, tn, pr = fetch_point_daily(lat, lon, date)
            except Exception as e:
                print(f"    warn ({lat:.2f},{lon:.2f}): {e}", file=sys.stderr)
                tx = tn = pr = None
            rmax.append(round(tx, 2) if tx is not None else None)
            rmin.append(round(tn, 2) if tn is not None else None)
            rpre.append(round(pr, 2) if pr is not None else None)
            done += 1
            time.sleep(0.5)   # 2 req/s, under the free-tier limit
        valid = [v for v in rmax if v is not None]
        if valid:
            print(f"    lat={lat:.2f}: tmax_max={max(valid):.1f}C  ({done*100//n}%)", flush=True)
        tmax.append(rmax); tmin.append(rmin); precip.append(rpre)
        time.sleep(1.0)

    flat = [v for row in tmax for v in row if v is not None]
    return {
        "label":       ev["label"],
        "date":        date,
        "lat":         lats,
        "lon":         lons,
        "tmax_c":      tmax,
        "tmin_c":      tmin,
        "precip_mm":   precip,
        "peak_tmax_c": round(max(flat), 2) if flat else None,
        "grid":        {"rows": DC_ROWS, "cols": DC_COLS, "source_grid": "PRISM 10-arcmin"},
        "note": (
            f"All {n} grid points fetched individually from Open-Meteo ERA5 "
            f"on the exact PRISM DC 24x24 grid. No interpolation or synthetic fill. "
            f"Dynamic inputs (tmax/tmin/precip) for a true OOD test of the "
            f"PRISM-trained ORBIT-2 model."
        ),
    }


def main():
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    lats, lons = prism_grid()
    total = len(lats) * len(lons) * len(EVENTS)
    print(f"=== DC OOD fields fetch — ALL REAL Open-Meteo ERA5, no interpolation "
          f"(~{total} requests) ===")
    print(f"Output: {ASSET}")
    print(f"Grid: {len(lats)}x{len(lons)} (PRISM rows {DC_ROWS}, cols {DC_COLS})")
    print(f"Events: {list(EVENTS.keys())}\n")

    events_out = {}
    for key, ev in EVENTS.items():
        events_out[key] = fetch_event_fields(key, ev)

    out = {
        "events":        events_out,
        "default_event": "july16_2024",
        "source":        "Open-Meteo ERA5 reanalysis (https://open-meteo.com)",
        "grid":          "PRISM 10-arcmin DC window (rows 78:102, cols 276:300), 24x24",
        "units":         {"tmax_c": "degC", "tmin_c": "degC", "precip_mm": "mm/day"},
        "purpose":       ("Dynamic OOD inputs for PRISM-trained ORBIT-2. Static channels "
                          "(land-sea mask, orography, latitude, landcover) come from the "
                          "PRISM DC crop at run time."),
        "method":        "Every grid point fetched individually from Open-Meteo. No interpolation.",
    }
    ASSET.write_text(json.dumps(out, indent=2))
    kb = ASSET.stat().st_size / 1024
    print(f"\n=== Written: {ASSET} ({kb:.0f} KB) ===")
    for k, ev in events_out.items():
        print(f"  {k}: peak tmax {ev['peak_tmax_c']}C")


if __name__ == "__main__":
    main()
