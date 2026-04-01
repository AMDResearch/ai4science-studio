#!/usr/bin/env python3
"""Run a multi-member StormCast ensemble forecast and save output to a Zarr store.

Each ensemble member draws independent random residuals from the diffusion
model at every step.  No initial-condition perturbation is applied (Zero()
perturbation); spread is generated entirely by the stochastic sampler.

Model weights are fetched automatically from Hugging Face
(nvidia/stormcast-v1-era5-hrrr) on the first run.

Usage
-----
    python run_ensemble.py --start 2025-08-09T12:00 --steps 12 --members 4
    python run_ensemble.py --start 2025-08-09T12:00 --steps 12 --members 8 --output my-ens.zarr

Reference
---------
https://rocm.blogs.amd.com/artificial-intelligence/stormcast-ensembles/README.html
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StormCast ensemble inference → Zarr output.",
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
        default=12,
        metavar="N",
        help="Number of 1-hour forecast steps per member.",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=4,
        metavar="N",
        help="Number of ensemble members to generate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Zarr output path.  Defaults to "
            "outputs/ens-<YYYY-MM-DD>.zarr.  "
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
        print("       Expected formats: 2025-08-09T12:00  or  2025-08-09T12  or  2025-08-09", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else Path("outputs") / f"ens-{start.strftime('%Y-%m-%d')}.zarr"

    if output.exists():
        print(
            f"error: output path already exists: {output}\n"
            "       The Zarr backend does not support overwriting.  Delete it first:\n"
            f"       rm -rf {output}",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading StormCast model …")
    try:
        from earth2studio.models.px import StormCast
        from earth2studio.data import HRRR
        from earth2studio.io import ZarrBackend
        from earth2studio.perturbation import Zero
        import earth2studio.run as run
    except ImportError as exc:
        print(f'error: {exc}\nRun: pip install "earth2studio[stormcast]"', file=sys.stderr)
        return 1

    package = StormCast.load_default_package()
    model = StormCast.load_model(package)
    data = HRRR()
    io = ZarrBackend(str(output))

    print(f"Start  : {start.isoformat()}Z")
    print(f"Steps  : {args.steps} × 1 h")
    print(f"Members: {args.members}")
    print(f"Output : {output}")

    run.ensemble(
        time=[start],
        nsteps=args.steps,
        nensemble=args.members,
        prognostic=model,
        data=data,
        io=io,
        perturbation=Zero(),
    )

    print(f"\nDone.  Output written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
