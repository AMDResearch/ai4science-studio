#!/usr/bin/env python3
"""Bake the demo telemetry asset from a REAL completed run.

Given a per-run perf dir (train_work/perf-runs/<jobid>/manifest.json), query its
Omnistat VictoriaMetrics DB and write the full `training_telemetry` result object to
studio/backend/assets/hydragnn_telemetry_8gpu.json, which the demo replays verbatim.

Usage:
    export PERF_TOOLS_DIR=/shared/spannala/perf-tools
    python bake_telemetry_asset.py /shared/spannala/models/HydraGNN/train_work/perf-runs/<jobid>

Scientific integrity: this ONLY reads real measured telemetry. It never fabricates.
"""
import json
import os
import sys
from pathlib import Path

# Reuse the backend harvester (single source of truth for the PromQL).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import telemetry  # noqa: E402

ASSET = Path(__file__).resolve().parents[1] / "backend" / "assets" / "hydragnn_telemetry_8gpu.json"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    perf_dir = Path(sys.argv[1])
    manifest = perf_dir / "manifest.json"
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    os.environ.setdefault("PERF_TOOLS_DIR", "/shared/spannala/perf-tools")

    tel = telemetry.harvest(manifest)
    if not tel:
        print("ERROR: no telemetry harvested (Omnistat DB missing/unreadable)", file=sys.stderr)
        return 1

    result = {
        "type": "training_telemetry",
        "slug": "HydraGNN",
        "model": "HydraGNN (Predictive GFM 2024)",
        "n_gpus": tel.get("n_gpus", 8),
        "epochs": tel.get("epochs"),
        "runtime_s": tel.get("runtime_s"),
        "telemetry": {k: tel[k] for k in ("peaks", "means", "units", "series") if k in tel},
    }
    val_json = perf_dir / "validation.json"
    if val_json.exists():
        v = json.loads(val_json.read_text())
        m = v.get("metrics", v)  # metrics are nested under "metrics"
        result["metrics"] = {k: m.get(k) for k in ("corr", "mae", "r2") if k in m}

    # Parse the per-epoch loss curve from the training log (for the Results tab).
    import re
    jobid = json.loads((perf_dir / "manifest.json").read_text()).get("jobid", perf_dir.name)
    log = Path(f"/shared/spannala/models/HydraGNN/train_work/logs/hg_tele8_{jobid}.log")
    if log.exists():
        pat = re.compile(r"Epoch:\s*(\d+),\s*Train Loss:\s*([\d.eE+-]+),\s*"
                         r"Val Loss:\s*([\d.eE+-]+),\s*Test Loss:\s*([\d.eE+-]+)")
        curve = [{"ep": int(x.group(1)), "train": float(x.group(2)),
                  "val": float(x.group(3)), "test": float(x.group(4))}
                 for x in (pat.search(l) for l in log.read_text().splitlines()) if x]
        if curve:
            result["loss_curve"] = curve

    ASSET.write_text(json.dumps(result, indent=2))
    _pk = result["telemetry"].get("peaks", {})
    print(f"Wrote {ASSET}")
    print(f"  peaks: power={_pk.get('power_w')}W util={_pk.get('gpu_util_pct')}% "
          f"fp64={_pk.get('fp64_tflops')}TFLOP/s energy={_pk.get('energy_kj')}kJ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
