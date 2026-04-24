#!/usr/bin/env python3
"""
Preflight check for GP-MoLFormer on AMD GPUs.

Verifies:
  1. GPU is accessible via ROCm/CUDA
  2. PyTorch can run a forward pass on GPU
  3. Key dependencies are importable (transformers, tokenizers)
  4. GP-MoLFormer scripts are reachable

Run inside the Docker container before a full inference job:
  python preflight_gpmolformer.py
"""

import os
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

check("import transformers", lambda: __import__("transformers").__version__)
check("import tokenizers", lambda: __import__("tokenizers").__version__)
check("import datasets", lambda: __import__("datasets").__version__)
check("import accelerate", lambda: __import__("accelerate").__version__)


def _rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit import __version__
        return __version__
    except ImportError:
        return "not installed (optional — validity checks will be skipped)"


check("rdkit availability", _rdkit)


def _scripts():
    candidates = [
        "/workspace/gp-molformer/scripts/unconditional_generation.py",
        os.path.join(os.path.dirname(__file__), "gp-molformer/scripts/unconditional_generation.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("gp-molformer scripts not found — clone IBM/gp-molformer first")


check("gp-molformer scripts", _scripts)

print()
if errors:
    print(f"{errors} check(s) failed — fix the issues above before running GP-MoLFormer.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run GP-MoLFormer.")
