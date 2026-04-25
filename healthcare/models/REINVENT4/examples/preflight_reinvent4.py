#!/usr/bin/env python3
"""
Preflight check for REINVENT4 on AMD GPUs.

Verifies:
  1. GPU is accessible via ROCm/CUDA
  2. PyTorch can run a forward pass on GPU
  3. REINVENT4 package is importable
  4. reinvent CLI is available

Run inside the Docker container before a full inference job:
  python preflight_reinvent4.py
"""

import shutil
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


def _reinvent():
    import reinvent  # noqa: F401
    return "ok"


check("import reinvent", _reinvent)


def _cli():
    path = shutil.which("reinvent")
    if path is None:
        raise FileNotFoundError("'reinvent' CLI not found on PATH")
    return path


check("reinvent CLI on PATH", _cli)

check("import rdkit", lambda: __import__("rdkit").__version__ if hasattr(__import__("rdkit"), "__version__") else "ok")

print()
if errors:
    print(f"{errors} check(s) failed — fix the issues above before running REINVENT4.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run REINVENT4.")
