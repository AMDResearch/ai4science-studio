#!/usr/bin/env python3
"""Single-process environment checks before running Aurora forecast jobs.

Run inside the container launched by docker_run.sh.  The GPU check requires
a node with an AMD GPU.

Usage
-----
    python preflight_aurora.py

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
        ("torch", "see docker_run.sh — built from silogen/ai-samples pytorch.dockerfile"),
        ("ai_models", "pip install ai-models"),
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
        import torch
        ver = torch.__version__
        check(f"torch version ({ver})", True)
    except ImportError:
        pass

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if gpu_available else 0
        check(
            f"GPU visible (cuda.is_available={gpu_available}, devices={device_count})",
            gpu_available,
            "Expected ROCm GPU — run this check on a GPU node, not a login node.",
        )
        if gpu_available:
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                print(f"  [info] device {i}: {name}")
    except Exception as exc:
        failures += 1
        check("GPU probe", False, str(exc))

    import os
    cds_key = os.environ.get("CDSAPI_KEY", "")
    failures += 0 if check(
        "CDSAPI_KEY set",
        bool(cds_key) and cds_key != "YOUR_CDS_API_KEY_HERE",
        "Set CDSAPI_KEY in env_file — get a free key at https://cds.climate.copernicus.eu/",
    ) else 1

    if failures == 0:
        print("preflight ok — environment is ready for Aurora.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
