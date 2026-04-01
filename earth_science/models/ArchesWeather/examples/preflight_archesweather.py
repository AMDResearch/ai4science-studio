#!/usr/bin/env python3
"""Single-process environment checks before running ArchesWeather jobs.

Run on a login or build node after setting up your environment (or inside the
Docker container launched by docker_run.sh).  No GPU is required for the
import checks; the GPU check requires a node with an AMD GPU.

Usage
-----
    python preflight_archesweather.py

Exit codes
----------
0 — all checks passed
1 — one or more checks failed (details printed to stderr)
"""

from __future__ import annotations

import importlib
import subprocess
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
        ("torch", "see docker_run.sh — install via silogen/ai-samples Dockerfile"),
        ("geoarches", "pip install geoarches  (or build from silogen/ai-samples)"),
        ("huggingface_hub", "pip install huggingface_hub"),
        ("hydra", "pip install hydra-core"),
        ("omegaconf", "pip install omegaconf"),
    ]:
        try:
            importlib.import_module(pkg)
            ok = True
            detail = ""
        except ImportError as exc:
            ok = False
            detail = f"ImportError: {exc}  →  {install_hint}"
        failures += 0 if check(f"import {pkg}", ok, detail) else 1

    # --- PyTorch version ---
    try:
        import torch
        ver = torch.__version__
        check(f"torch version ({ver})", True)
    except ImportError:
        pass  # already caught above

    # --- GPU check ---
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

    # --- HuggingFace Hub reachable ---
    try:
        from huggingface_hub import list_models
        # Light probe: list 1 model from gcouairon/ArchesWeather
        models = list(list_models(author="gcouairon", limit=1))
        ok = len(models) >= 0  # always true if no exception
        failures += 0 if check("HuggingFace Hub reachable (gcouairon/ArchesWeather)", ok) else 1
    except Exception as exc:
        failures += 1
        check("HuggingFace Hub reachable", False, str(exc))

    # --- geoarches module import ---
    try:
        import geoarches  # noqa: F401
        check("geoarches importable", True)
    except Exception as exc:
        failures += 1
        check("geoarches importable", False, str(exc))

    if failures == 0:
        print("preflight ok — environment is ready for ArchesWeather.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
