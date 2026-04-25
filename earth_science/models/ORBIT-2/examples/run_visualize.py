#!/usr/bin/env python3
"""Launch upstream ORBIT-2 ``examples/visualize.py`` with correct cwd and PYTHONPATH.

Upstream visualization is distributed and expects SLURM environment variables
(``SLURM_NTASKS``, ``SLURM_PROCID``, ``SLURM_LOCALID``). Run this script under
``srun`` (or equivalent) with a task count that matches the YAML ``parallelism``
settings. See ``../recipes/inference-and-visualization.md``.

Reference: https://github.com/XiaoWang-Github/ORBIT-2/blob/main/examples/visualize.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve_orbit2_root(explicit: str | None) -> Path:
    raw = explicit or os.environ.get("ORBIT2_ROOT")
    if not raw:
        print(
            "error: set ORBIT2_ROOT or pass --orbit2-root (clone of XiaoWang-Github/ORBIT-2)",
            file=sys.stderr,
        )
        sys.exit(2)
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        print(f"error: ORBIT2_ROOT is not a directory: {root}", file=sys.stderr)
        sys.exit(2)
    return root


def _resolve_config(orbit2_root: Path, config: str) -> Path:
    p = Path(config).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(orbit2_root / "configs" / p)
        candidates.append(orbit2_root / p)
    tried: list[Path] = []
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        tried.append(r)
        if r.is_file():
            return r
    print("error: config file not found. Tried:", file=sys.stderr)
    for t in tried:
        print(f"  {t}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run upstream ORBIT-2 visualize.py (distributed; use inside srun).",
    )
    parser.add_argument(
        "--orbit2-root",
        type=str,
        default=None,
        help="Path to ORBIT-2 clone (default: env ORBIT2_ROOT)",
    )
    parser.add_argument(
        "config",
        type=str,
        help="YAML config path (absolute, or relative to cwd / ORBIT-2/configs / ORBIT-2 root)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Test sample index (default: 0)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="total_precipitation_24hr",
        help="Variable to visualize",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="29500",
        help="Distributed master port",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["float32", "bfloat16"],
        default=None,
        help="Override trainer data type (optional)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (.ckpt)",
    )
    args = parser.parse_args()

    orbit2_root = _resolve_orbit2_root(args.orbit2_root)
    examples_dir = orbit2_root / "examples"
    vis = examples_dir / "visualize.py"
    util = examples_dir / "utils.py"
    for need, label in ((vis, "visualize.py"), (util, "utils.py")):
        if not need.is_file():
            print(
                f"error: missing {label} under {examples_dir} "
                "(clone https://github.com/XiaoWang-Github/ORBIT-2 and pip install -e .)",
                file=sys.stderr,
            )
            sys.exit(2)

    config_path = _resolve_config(orbit2_root, args.config)

    env = os.environ.copy()
    src = str(orbit2_root / "src")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + prev if prev else "")

    child_argv = [
        sys.executable,
        str(vis),
        str(config_path),
        "--index",
        str(args.index),
        "--variable",
        args.variable,
        "--master-port",
        args.master_port,
    ]
    if args.data_type is not None:
        child_argv.extend(["--data-type", args.data_type])
    if args.checkpoint is not None:
        child_argv.extend(["--checkpoint", args.checkpoint])

    os.chdir(examples_dir)
    os.execvpe(sys.executable, child_argv, env)


if __name__ == "__main__":
    main()
