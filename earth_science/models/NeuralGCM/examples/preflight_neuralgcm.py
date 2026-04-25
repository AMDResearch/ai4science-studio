#!/usr/bin/env python3
"""Single-process environment checks before running NeuralGCM inference jobs.

Run on a login or build node after installing dependencies.  No GPU required
for the import checks; JAX device check requires a GPU node.

Usage
-----
    python preflight_neuralgcm.py

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
        ("neuralgcm", "pip install neuralgcm"),
        ("gcsfs", "pip install gcsfs"),
        ("xarray", "pip install xarray"),
        ("jax", "see recipes/inference/README.md for ROCm JAX install"),
        ("jaxlib", "see recipes/inference/README.md for ROCm JAX install"),
        ("matplotlib", "pip install matplotlib"),
    ]:
        try:
            importlib.import_module(pkg)
            ok = True
            detail = ""
        except ImportError as exc:
            ok = False
            detail = f"ImportError: {exc}  →  {install_hint}"
        failures += 0 if check(f"import {pkg}", ok, detail) else 1

    # --- JAX version ---
    try:
        import jax
        ver = jax.__version__
        check(f"jax version ({ver})", True)
    except ImportError:
        pass  # already caught above

    # --- JAX devices (GPU check) ---
    try:
        import jax
        devices = jax.devices()
        gpu_found = any("rocm" in str(d).lower() or "gpu" in str(d).lower() for d in devices)
        check(
            f"JAX GPU device visible ({devices})",
            gpu_found,
            "Expected RocmDevice — run this check on a GPU node, not a login node.",
        )
        for d in devices:
            print(f"  [info] device: {d}")
    except Exception as exc:
        failures += 1
        check("JAX device probe", False, str(exc))

    # --- Anonymous GCS access ---
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        exists = fs.exists("gs://neuralgcm/models/v1/deterministic_1_4_deg.pkl")
        failures += 0 if check("GCS checkpoint accessible (anon)", exists,
                               "Check network / firewall; no auth needed.") else 1
    except Exception as exc:
        failures += 1
        check("GCS checkpoint accessible (anon)", False, str(exc))

    # --- NeuralGCM model class ---
    try:
        import neuralgcm
        _ = neuralgcm.PressureLevelModel
        failures += 0 if check("neuralgcm.PressureLevelModel importable", True) else 1
    except Exception as exc:
        failures += 1
        check("neuralgcm.PressureLevelModel importable", False, str(exc))

    if failures == 0:
        print("preflight ok — environment is ready for NeuralGCM.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
