#!/usr/bin/env python3
"""
Preflight check for Walrus on AMD GPUs.

Verifies:
  1. GPU is accessible via ROCm/CUDA
  2. PyTorch can run a forward pass on GPU
  3. huggingface-hub is importable
  4. walrus package is importable

Run inside the Docker container before a full inference job:
  python preflight_walrus.py
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


# 1. PyTorch
check("import torch", lambda: __import__("torch").__version__)

# 2. GPU
def _gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    return torch.cuda.get_device_name(0)

check("GPU accessible", _gpu)

# 3. Tensor ops on GPU
def _tensor():
    import torch
    x = torch.randn(8, 8).cuda()
    return f"result={( x @ x).sum().item():.4f}"

check("GPU tensor ops", _tensor)

# 4. huggingface-hub
check("import huggingface_hub", lambda: __import__("huggingface_hub").__version__)

# 5. walrus package
def _walrus():
    import walrus  # type: ignore  # noqa: F401
    return "ok"

check("import walrus", _walrus)

# ---------------------------------------------------------------------------
print()
if errors:
    print(f"{errors} check(s) failed — fix the issues above before running Walrus.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run Walrus.")
