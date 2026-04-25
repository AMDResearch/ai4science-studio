#!/usr/bin/env python3
"""
Preflight check for HydraGNN on AMD GPUs.

Verifies:
  1. GPU is accessible via ROCm/CUDA
  2. PyTorch can run a forward pass on GPU
  3. HydraGNN package is importable
  4. Key dependencies are present (torch_geometric, mpi4py)

Run inside the Docker container before a full inference job:
  python preflight_hydragnn.py
"""

import sys

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = 0


def check(label, fn):
    global errors
    try:
        result = fn()
        print(f"[{PASS}] {label}" + (f": {result}" if result else ""))
    except Exception as exc:
        print(f"[{FAIL}] {label}: {exc}")
        errors += 1


check("import torch", lambda: __import__("torch").__version__)


def _gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    return torch.cuda.get_device_name(0)


check("GPU accessible", _gpu)


def _tensor():
    import torch
    x = torch.randn(8, 8).cuda()
    return f"result={( x @ x).sum().item():.4f}"


check("GPU tensor ops", _tensor)


def _hydragnn():
    import hydragnn  # noqa: F401
    return "ok"


check("import hydragnn", _hydragnn)


def _torch_geometric():
    import torch_geometric  # noqa: F401
    return torch_geometric.__version__


check("import torch_geometric", _torch_geometric)


def _mpi4py():
    try:
        from mpi4py import MPI  # noqa: F401
        return "ok"
    except ImportError:
        return "not installed (optional — single-GPU inference works without it)"


check("mpi4py availability", _mpi4py)

check("import adios2", lambda: __import__("adios2").__version__ if hasattr(__import__("adios2"), "__version__") else "ok")

print()
if errors:
    print(f"{errors} check(s) failed — fix the issues above before running HydraGNN.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run HydraGNN.")
