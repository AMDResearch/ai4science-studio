#!/usr/bin/env python3
"""Single-process environment checks before running PanguWeather forecast jobs.

Run inside the container launched by docker_run.sh.  The GPU check requires
a node with an AMD GPU.

Usage
-----
    python preflight_panguweather.py

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

    for pkg, install_hint in [
        ("jax", "see docker_run.sh — built from silogen/ai-samples jax.dockerfile"),
        ("jaxlib", "see docker_run.sh — built from silogen/ai-samples jax.dockerfile"),
        ("ai_models", "pip install ai-models"),
        ("onnxruntime", "pip install onnxruntime (ROCm-patched version in Docker image)"),
    ]:
        try:
            importlib.import_module(pkg)
            ok = True
            detail = ""
        except ImportError as exc:
            ok = False
            detail = f"ImportError: {exc}  →  {install_hint}"
        failures += 0 if check(f"import {pkg}", ok, detail) else 1

    try:
        import jax
        ver = jax.__version__
        check(f"jax version ({ver})", True)
    except ImportError:
        pass

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

    import os
    cds_key = os.environ.get("CDSAPI_KEY", "")
    failures += 0 if check(
        "CDSAPI_KEY set",
        bool(cds_key) and cds_key != "YOUR_CDS_API_KEY_HERE",
        "Set CDSAPI_KEY in env_file — get a free key at https://cds.climate.copernicus.eu/",
    ) else 1

    if failures == 0:
        print("preflight ok — environment is ready for PanguWeather.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
