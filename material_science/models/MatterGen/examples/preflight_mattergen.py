#!/usr/bin/env python3
"""Single-process environment checks before running MatterGen jobs.

Run on a login or build node after installing MatterGen (via docker_run.sh or
manually).  No GPU is required for the import checks; the GPU check requires
a node with an AMD GPU.

Usage
-----
    python preflight_mattergen.py

Exit codes
----------
0 — all checks passed
1 — one or more checks failed (details printed to stderr)

Note: MatterGen requires ROCm-compatible forks of pytorch_scatter and
pytorch_sparse (silogen/pytorch_scatter, silogen/pytorch_sparse).
The upstream CUDA-only versions will fail on ROCm.
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
        ("torch", "see docker_run.sh — uses rocm/pytorch base image"),
        ("torch_scatter", "install ROCm fork: pip install git+https://github.com/silogen/pytorch_scatter.git"),
        ("torch_sparse", "install ROCm fork: pip install git+https://github.com/silogen/pytorch_sparse.git"),
        ("mattergen", "run docker_run.sh or: cd /workspace/mattergen && bash src/setup.bash"),
        ("ase", "pip install ase"),
        ("pymatgen", "pip install pymatgen"),
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

    # --- ROCm fork check: torch_scatter ops ---
    try:
        import torch
        import torch_scatter
        # Quick scatter_add smoke test
        src = torch.ones(5)
        idx = torch.tensor([0, 0, 1, 1, 2])
        out = torch_scatter.scatter_add(src, idx)
        ok = len(out) == 3
        failures += 0 if check("torch_scatter.scatter_add smoke test", ok,
                               "ROCm fork required — upstream CUDA-only version will fail.") else 1
    except Exception as exc:
        failures += 1
        check("torch_scatter smoke test", False,
              f"{exc}  →  install silogen/pytorch_scatter ROCm fork")

    # --- MatterGen model class ---
    try:
        from mattergen.diffusion.model import MatterGenModel  # noqa: F401
        check("mattergen.diffusion.model.MatterGenModel importable", True)
    except Exception as exc:
        failures += 1
        check("MatterGenModel importable", False, str(exc))

    if failures == 0:
        print("preflight ok — environment is ready for MatterGen.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
