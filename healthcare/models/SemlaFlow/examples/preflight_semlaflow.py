#!/usr/bin/env python3
"""
Preflight check for SemlaFlow on AMD GPUs.

Verifies:
  1. GPU is accessible via ROCm/CUDA
  2. PyTorch can run a forward pass on GPU
  3. SemlaFlow package is importable
  4. torch.compile is functional

Run inside the Docker container before a full inference job:
  python preflight_semlaflow.py
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


def _semlaflow():
    import semlaflow  # noqa: F401
    return "ok"


check("import semlaflow", _semlaflow)


def _torch_compile():
    import torch

    @torch.compile
    def add(a, b):
        return a + b

    a = torch.tensor(1.0).cuda()
    b = torch.tensor(2.0).cuda()
    result = add(a, b)
    return f"result={result.item():.1f}"


check("torch.compile functional", _torch_compile)

print()
if errors:
    print(f"{errors} check(s) failed — fix the issues above before running SemlaFlow.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run SemlaFlow.")
