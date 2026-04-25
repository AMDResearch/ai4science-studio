#!/usr/bin/env python3
"""Single-process checks before submitting ORBIT-2 visualization jobs (no SLURM required).

Run on a login or build node after activating the same conda env you use on GPU nodes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for ORBIT-2 + Studio launchers.")
    parser.add_argument(
        "--orbit2-root",
        type=str,
        default=None,
        help="Path to ORBIT-2 clone (default: env ORBIT2_ROOT)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config to verify exists and parses",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional .ckpt path to verify exists",
    )
    args = parser.parse_args()

    raw = args.orbit2_root or os.environ.get("ORBIT2_ROOT")
    if not raw:
        print("error: set ORBIT2_ROOT or pass --orbit2-root", file=sys.stderr)
        return 2
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    examples = root / "examples"
    for name in ("visualize.py", "utils.py"):
        p = examples / name
        if not p.is_file():
            print(f"error: missing {p}", file=sys.stderr)
            return 2

    src = root / "src"
    if not src.is_dir():
        print(f"error: missing src tree: {src}", file=sys.stderr)
        return 2

    if args.checkpoint:
        ck = Path(args.checkpoint).expanduser().resolve()
        if not ck.is_file():
            print(f"error: checkpoint not found: {ck}", file=sys.stderr)
            return 2

    if args.config:
        cfg = Path(args.config).expanduser()
        if cfg.is_file():
            cfg = cfg.resolve()
        else:
            alt = (Path.cwd() / cfg).resolve()
            if alt.is_file():
                cfg = alt
            else:
                alt2 = root / "configs" / Path(args.config).name
                if alt2.is_file():
                    cfg = alt2.resolve()
                else:
                    alt3 = (root / args.config).resolve()
                    if alt3.is_file():
                        cfg = alt3
                    else:
                        print(f"error: config not found: {args.config}", file=sys.stderr)
                        return 2
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            print("error: PyYAML not installed (pip install pyyaml)", file=sys.stderr)
            return 2
        with open(cfg, "r", encoding="utf-8") as f:
            yaml.safe_load(f)

    try:
        import torch
    except ImportError:
        print("error: torch not installed in this environment", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "warning: torch.cuda.is_available() is False (expected on GPU nodes; "
            "login nodes may lack GPUs)",
            file=sys.stderr,
        )

    sys.path.insert(0, str(src))
    try:
        import climate_learn  # noqa: F401
    except ImportError as e:
        print(
            f"error: cannot import climate_learn from {src} ({e}). "
            "From the ORBIT-2 clone root run: pip install -e .",
            file=sys.stderr,
        )
        return 2

    print("preflight ok:", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
