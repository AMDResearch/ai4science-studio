#!/usr/bin/env python3
"""
Preflight check for MATEY on AMD GPUs.

Verifies:
  1. GPU is accessible via ROCm/CUDA
  2. PyTorch can run a forward pass on GPU
  3. MATEY package is importable
  4. HDF5 / h5py available

Run inside the Docker container before a full training or inference job:
  python preflight_matey.py
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


# 1. PyTorch import
check("import torch", lambda: __import__("torch").__version__)

# 2. GPU available
def _gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    return torch.cuda.get_device_name(0)

check("GPU accessible", _gpu)

# 3. Tensor to GPU
def _tensor():
    import torch
    x = torch.randn(4, 4).cuda()
    y = (x @ x).sum()
    return f"result={y.item():.4f}"

check("GPU tensor ops", _tensor)

# 4. h5py
check("import h5py", lambda: __import__("h5py").version.version)

# 5. yaml
check("import yaml", lambda: __import__("yaml").__version__)

# 6. numpy
check("import numpy", lambda: __import__("numpy").__version__)

# 7. MATEY package
def _matey():
    import matey  # type: ignore  # noqa: F401
    return "ok"

check("import matey", _matey)

# ---------------------------------------------------------------------------
print()
if errors:
    print(f"{errors} check(s) failed — fix the issues above before running MATEY.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run MATEY.")
