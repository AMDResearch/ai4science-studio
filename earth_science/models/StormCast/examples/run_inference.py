#!/usr/bin/env python3
"""Run a deterministic StormCast forecast and save output to a Zarr store.

Model weights are fetched automatically from Hugging Face
(nvidia/stormcast-v1-era5-hrrr) on the first run.  Initial conditions are
pulled live from the NOAA HRRR analysis archive.

Usage
-----
    python run_inference.py --start 2025-01-01T06:00 --steps 6
    python run_inference.py --start 2025-01-01T06:00 --steps 6 --output my-run.zarr

Reference
---------
https://rocm.blogs.amd.com/artificial-intelligence/stormcast-inference/README.html
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic StormCast inference → Zarr output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DDTHH:MM",
        help="Forecast initialisation time (UTC).  Use HRRR analysis hours: 00, 06, 12, 18.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        metavar="N",
        help="Number of 1-hour forecast steps to produce.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Zarr output path.  Defaults to "
            "outputs/pred-<YYYY-MM-DD>.zarr.  "
            "The Zarr backend does not support overwriting — delete the path first if it exists."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Parse start time
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            start = datetime.strptime(args.start, fmt)
            break
        except ValueError:
            continue
    else:
        print(f"error: cannot parse --start '{args.start}'", file=sys.stderr)
        print("       Expected formats: 2025-01-01T06:00  or  2025-01-01T06  or  2025-01-01", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else Path("outputs") / f"pred-{start.strftime('%Y-%m-%d')}.zarr"

    if output.exists():
        print(
            f"error: output path already exists: {output}\n"
            "       The Zarr backend does not support overwriting.  Delete it first:\n"
            f"       rm -rf {output}",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading StormCast model …")
    try:
        from earth2studio.models.px import StormCast
        from earth2studio.data import HRRR
        from earth2studio.io import ZarrBackend
        import earth2studio.run as run
    except ImportError as exc:
        print(f'error: {exc}\nRun: pip install "earth2studio[stormcast]"', file=sys.stderr)
        return 1

    package = StormCast.load_default_package()
    model = StormCast.load_model(package)
    data = HRRR()
    io = ZarrBackend(str(output))

    print(f"Start : {start.isoformat()}Z")
    print(f"Steps : {args.steps} × 1 h")
    print(f"Output: {output}")

    run.deterministic([start], args.steps, model, data, io)

    print(f"\nDone.  Output written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
