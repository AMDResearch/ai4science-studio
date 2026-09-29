#!/usr/bin/env python3
"""Fetch REAL ERA5 temperature data for DC heatwave events from Open-Meteo.

Every grid point is fetched individually — no interpolation, no synthetic fill.
Events:
  - july16_2024 : DC record heatwave (104F / 40C), hottest since 1930
  - july4_2026  : Independence Day 2026 heatwave (103.3F / 39.6C)

Run: python3 studio/backend/tools/fetch_dc_era5.py
Output: studio/backend/assets/dc_temperature.json
"""
import json
import ssl
import sys
import time
import urllib.request as urlreq
import urllib.parse as urlparse
from pathlib import Path

ASSET = Path(__file__).resolve().parents[1] / "assets" / "dc_temperature.json"

DC_LAT, DC_LON = 38.8951, -77.0364

EVENTS = {
    "july16_2024": {
        "label":        "July 16 2024 — DC Record Heatwave",
        "date":         "2024-07-16",
        "window_start": "2024-07-14",
        "window_end":   "2024-07-17",
        "note": (
            "July 14-17 2024 was the hottest stretch in DC since 1930. "
            "July 16 reached 104 F (40 C) at Reagan National Airport — "
            "the first four consecutive triple-digit days since 2012."
        ),
    },
    "july4_2026": {
        "label":        "July 4 2026 — Independence Day Heatwave",
        "date":         "2026-07-04",
        "window_start": "2026-07-03",
        "window_end":   "2026-07-05",
        "note": (
            "Independence Day 2026 was a major heatwave in DC: peak 39.6 C (103.3 F) "
            "at Reagan National Airport — the second triple-digit Independence Day "
            "in recorded history."
        ),
    },
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch(params: dict, retries: int = 4) -> dict:
    """Fetch one Open-Meteo request with retry + SSL workaround."""
    qs = urlparse.urlencode(params)
    url = f"{BASE_URL}?{qs}"
    for attempt in range(retries):
        try:
            req = urlreq.Request(url)
            with urlreq.urlopen(req, timeout=20, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt < retries - 1:
                wait = 3 * (2 ** attempt)
                print(f"  retry {attempt+1}/{retries-1} after {wait}s ({e})", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


def fetch_daily_max(lat: float, lon: float, date: str, model: str) -> float | None:
    """Return the daily maximum 2m temperature (°C) at one grid point."""
    data = fetch({
        "latitude":   round(lat, 4),
        "longitude":  round(lon, 4),
        "start_date": date,
        "end_date":   date,
        "hourly":     "temperature_2m",
        "models":     model,
        "timezone":   "UTC",
    })
    vals = [t for t in data["hourly"]["temperature_2m"] if t is not None]
    return max(vals) if vals else None


def build_real_grid(center_lat, center_lon, date, model, n, step, label):
    """Fetch every grid point from Open-Meteo — all values are real ERA5 data.
    n×n grid centered at (center_lat, center_lon) with spacing `step` degrees.
    No interpolation or synthetic fill.
    """
    half = (n - 1) * step / 2
    lats = [round(center_lat - half + i * step, 4) for i in range(n)]
    lons = [round(center_lon - half + j * step, 4) for j in range(n)]
    total = n * n
    print(f"  Fetching {n}×{n}={total} real ERA5 points for {label} ...", flush=True)

    all_rows = []
    fetched = 0
    for lat in lats:
        row = []
        for lon in lons:
            try:
                t = fetch_daily_max(lat, lon, date, model)
                row.append(round(t, 2) if t is not None else None)
            except Exception as e:
                print(f"    warn ({lat:.2f},{lon:.2f}): {e}", file=sys.stderr)
                row.append(None)
            fetched += 1
            time.sleep(0.5)   # 2 req/s — safely under the 10k/day free-tier limit
        valid = [r for r in row if r is not None]
        pct = fetched * 100 // total
        if valid:
            print(f"    lat={lat:.2f}: max={max(valid):.1f}°C  ({pct}%)", flush=True)
        all_rows.append(row)
        time.sleep(1.0)   # extra 1s pause between rows to avoid burst detection

    all_vals = [t for row in all_rows for t in row if t is not None]
    peak = max(all_vals) if all_vals else 0.0
    return {
        "lat": lats,
        "lon": lons,
        "temp_c": all_rows,
        "peak_temp_c": round(peak, 2),
        "resolution_deg": step,
        "model": model,
        "label": label,
        "note": (
            f"All {total} grid points fetched individually from Open-Meteo "
            f"ERA5 ({label}). No interpolation or synthetic fill."
        ),
    }


def fetch_event(key, ev):
    date = ev["date"]

    print(f"\n[{key}] Coarse 20×20 grid at 0.25° (ERA5 real values) ...")
    coarse = build_real_grid(DC_LAT, DC_LON, date, model="era5",
                              n=20, step=0.25, label="ERA5 0.25° (~28 km)")

    print(f"\n[{key}] Fine 20×20 grid at 0.1° (ERA5-Land real values) ...")
    fine = build_real_grid(DC_LAT, DC_LON, date, model="era5_land",
                            n=20, step=0.1, label="ERA5-Land 0.1° (~11 km)")

    print(f"[{key}] Hourly series ({ev['window_start']} → {ev['window_end']}) ...")
    ts = fetch({
        "latitude":   DC_LAT, "longitude": DC_LON,
        "start_date": ev["window_start"], "end_date": ev["window_end"],
        "hourly":     "temperature_2m", "models": "era5",
        "timezone":   "America/New_York",
    })
    time.sleep(1.5)

    peak = max(coarse["peak_temp_c"], fine["peak_temp_c"])
    return {
        "label":       ev["label"],
        "date":        date,
        "peak_temp_c": round(peak, 2),
        "peak_temp_f": round(peak * 9 / 5 + 32, 1),
        "dc":          {"lat": DC_LAT, "lon": DC_LON},
        "coarse":      coarse,
        "fine":        fine,
        "time_series": {
            "times":  ts["hourly"]["time"],
            "temp_c": ts["hourly"]["temperature_2m"],
        },
        "note": ev["note"],
    }


def main():
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    n_per_event = 20 * 20 + 20 * 20 + 1   # coarse + fine + time-series
    total = n_per_event * len(EVENTS)
    print(f"=== DC ERA5 fetch — ALL REAL DATA, no interpolation (~{total} requests) ===")
    print(f"Output: {ASSET}")
    print(f"Events: {list(EVENTS.keys())}\n")

    events_out = {}
    for key, ev in EVENTS.items():
        print(f"--- Event: {ev['label']} ---")
        events_out[key] = fetch_event(key, ev)

    out = {
        "events":        events_out,
        "default_event": "july16_2024",
        "source":        "Open-Meteo ERA5 reanalysis (https://open-meteo.com)",
        "location":      "Washington DC (Reagan National Airport, 38.89°N 77.04°W)",
        "method":        "Every grid point fetched individually from Open-Meteo. No interpolation.",
    }

    ASSET.write_text(json.dumps(out, indent=2))
    kb = ASSET.stat().st_size / 1024
    print(f"\n=== Written: {ASSET} ({kb:.0f} KB) ===")
    for k, ev in events_out.items():
        print(f"  {k}: peak {ev['peak_temp_c']}°C / {ev['peak_temp_f']}°F")


if __name__ == "__main__":
    main()
