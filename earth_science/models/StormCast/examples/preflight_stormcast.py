#!/usr/bin/env python3
"""Single-process environment checks before running StormCast inference or ensemble jobs.

Run on a login or build node after installing dependencies.  No GPU required.

Usage
-----
    python preflight_stormcast.py

Exit codes
----------
0 — all checks passed
1 — one or more checks failed (details printed to stderr)
"""

from __future__ import annotations

import importlib
import sys


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "ok" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if not ok and detail:
        msg += f"\n         {detail}"
    print(msg, file=sys.stderr if not ok else sys.stdout)
    return ok


def main() -> int:
    failures = 0

    # --- Required packages ---
    for pkg, install_hint in [
        ("earth2studio", 'pip install "earth2studio[stormcast]"'),
        ("cartopy", "pip install cartopy"),
        ("xarray", "pip install xarray"),
        ("zarr", "pip install zarr"),
        ("torch", "installed via ROCm PyTorch container or pip"),
    ]:
        try:
            importlib.import_module(pkg)
            ok = True
            detail = ""
        except ImportError as exc:
            ok = False
            detail = f"ImportError: {exc}  →  {install_hint}"
        failures += 0 if check(f"import {pkg}", ok, detail) else 1

    # --- StormCast model class ---
    try:
        from earth2studio.models.px import StormCast  # noqa: F401
        failures += 0 if check("earth2studio.models.px.StormCast importable", True) else 1
    except Exception as exc:
        failures += 1
        check("earth2studio.models.px.StormCast importable", False, str(exc))

    # --- torch.cuda ---
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        check(
            "torch.cuda.is_available()",
            cuda_ok,
            "Expected False on CPU-only login nodes; must be True on GPU nodes.",
        )
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                print(f"  [info] GPU {i}: {name}  ({mem} GiB)")
    except ImportError:
        pass  # already caught above

    # --- HRRR data connector reachable ---
    try:
        from earth2studio.data import HRRR  # noqa: F401
        check("earth2studio.data.HRRR importable", True)
    except Exception as exc:
        failures += 1
        check("earth2studio.data.HRRR importable", False, str(exc))

    if failures == 0:
        print("preflight ok — environment is ready for StormCast.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
