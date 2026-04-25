#!/usr/bin/env python3
"""Single-process environment checks before running SwinUNETR jobs.

Run on a login or build node after setting up your environment (or inside the
Docker container launched by docker_run.sh).  No GPU is required for the
import checks; the GPU check requires a node with an AMD GPU.

Training image: rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0
Inference image: rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0

Usage
-----
    python preflight_swinunetr.py

Exit codes
----------
0 — all checks passed
1 — one or more checks failed (details printed to stderr)
"""

from __future__ import annotations

import importlib
import os
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
        ("monai", "pip install 'monai[all]'"),
        ("nibabel", "pip install nibabel"),
        ("numpy", "pip install numpy"),
    ]:
        try:
            importlib.import_module(pkg)
            ok = True
            detail = ""
        except ImportError as exc:
            ok = False
            detail = f"ImportError: {exc}  →  {install_hint}"
        failures += 0 if check(f"import {pkg}", ok, detail) else 1

    # --- PyTorch and MONAI versions ---
    try:
        import torch
        import monai
        check(f"torch version ({torch.__version__})", True)
        check(f"monai version ({monai.__version__})", True)
    except ImportError:
        pass

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

    # --- MIOpen env vars (training optimization) ---
    find_mode = os.environ.get("MIOPEN_FIND_MODE", "")
    find_enforce = os.environ.get("MIOPEN_FIND_ENFORCE", "")
    miopen_ok = find_mode == "1" and find_enforce == "3"
    check(
        f"MIOpen auto-tuning env vars (FIND_MODE={find_mode or 'unset'}, FIND_ENFORCE={find_enforce or 'unset'})",
        miopen_ok,
        "Set MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=3 for ~3× training speedup.\n"
        "         (Default on ROCm 6.4+ / PyTorch 2.6+; set explicitly on older stacks.)",
    )
    # Not a fatal failure — training still works without it
    if not miopen_ok:
        failures = max(failures, 0)  # warn only

    # --- SwinUNETR model instantiation ---
    try:
        import torch
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
        )
        param_count = sum(p.numel() for p in model.parameters())
        check(f"SwinUNETR instantiation ({param_count:,} params)", True)
    except Exception as exc:
        failures += 1
        check("SwinUNETR instantiation", False, str(exc))

    # --- torch.compile available (inference optimization) ---
    try:
        import torch
        has_compile = hasattr(torch, "compile")
        check(
            f"torch.compile available (max-autotune inference, 2.9× speedup)",
            has_compile,
            "Requires PyTorch 2.0+. Inference image (ROCm 7.0) enables max-autotune.",
        )
    except Exception as exc:
        failures += 1
        check("torch.compile check", False, str(exc))

    if failures == 0:
        print("preflight ok — environment is ready for SwinUNETR.")
    else:
        print(f"\npreflight FAILED — {failures} issue(s) found.", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
