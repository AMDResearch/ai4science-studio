#!/usr/bin/env python3
"""Run ORBIT2_RANK_PRE_TRAIN_HOOK only (e.g. profiler) before launch_diffusion.sh.

Mirrors rank-gating in run_orbit2_train._run_hook. Expects ORBIT2_ROOT and cwd
compatible with importing intermediate_downscaling (chdir to .../examples).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    hook = os.environ.get("ORBIT2_RANK_PRE_TRAIN_HOOK", "").strip()
    if not hook:
        return 0
    hook_path = Path(hook)
    if not hook_path.is_file():
        raise FileNotFoundError(f"ORBIT2_RANK_PRE_TRAIN_HOOK not found: {hook}")

    rank0_only = os.environ.get("PROFILE_RANK0_ONLY", "1") == "1"
    world_rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank0_only and world_rank != 0:
        return 0

    root = os.environ.get("ORBIT2_ROOT", "/orbit2")
    examples = Path(root) / "examples"
    launch = Path(root) / "launch"
    if examples.is_dir():
        os.chdir(examples)
    elif launch.is_dir():
        os.chdir(launch)
    for pref in (str(Path(root) / "src"), str(examples), str(launch), root):
        if pref not in sys.path:
            sys.path.insert(0, pref)

    runpy.run_path(str(hook_path), run_name="__orbit2_hook__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
