#!/usr/bin/env python3
"""Heuristic max batch estimate from ORBIT-2 SLURM logs or two manual calibration points.

ORBIT rank-0 logs often include::
    torch.cuda.memory_reserved: 12.07GB

This tool either:
  * parses **peak** reserved memory from a log (optional filter on ``batch_idx``), or
  * takes two (batch_size, reserved_GB) pairs and linearly extrapolates to ``--target-gb``.

**Caveats:** reserved memory is **not** strictly linear in ``batch_size`` (attention/FFN
terms, fragmentation, checkpoints, FSDP sharding). Use the output as a **starting
guess**, then confirm with **binary-search** short jobs or a direct OOM probe.

Examples::

    # Two calibration jobs (float32), extrapolate to 240 GiB budget
    python3 orbit2_estimate_batch_from_memory.py \\
        --batch-a 4 --gb-a 12.07 --batch-b 8 --gb-b 28.14 --target-gb 240

    # Parse one log; pair with known batch_size from rendered YAML
    python3 orbit2_estimate_batch_from_memory.py --log orbit2-train-9145.out --target-gb 240
    # (prints peak GB only unless you pass --known-batch)

    python3 orbit2_estimate_batch_from_memory.py \\
        --log orbit2-train-9145.out --known-batch 4 --log2 orbit2-train-9146.out --known-batch2 8 \\
        --target-gb 240
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_MEM_RE = re.compile(r"memory_reserved:\s*([0-9.]+)\s*GB", re.IGNORECASE)
_BATCH_IDX_RE = re.compile(r"batch_idx\s+(\d+)", re.IGNORECASE)


def _peak_gb_from_log(
    path: Path,
    *,
    batch_idx_only: int | None,
) -> float:
    mx = 0.0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if batch_idx_only is not None:
                mbi = _BATCH_IDX_RE.search(line)
                if not mbi or int(mbi.group(1)) != batch_idx_only:
                    continue
            m = _MEM_RE.search(line)
            if m:
                mx = max(mx, float(m.group(1)))
    return mx


def _linear_extrapolate(b1: float, g1: float, b2: float, g2: float, target: float) -> float | None:
    if b1 == b2:
        return None
    slope = (g2 - g1) / (b2 - b1)
    if slope <= 0:
        return None
    # g = g1 + slope * (b - b1)  =>  b = b1 + (target - g1) / slope
    return b1 + (target - g1) / slope


def main() -> int:
    p = argparse.ArgumentParser(description="Rough max batch from memory calibration.")
    p.add_argument("--batch-a", type=float, help="First batch_size")
    p.add_argument("--gb-a", type=float, help="First peak reserved GiB")
    p.add_argument("--batch-b", type=float, help="Second batch_size")
    p.add_argument("--gb-b", type=float, help="Second peak reserved GiB")
    p.add_argument("--log", type=Path, help="SLURM stdout log (orbit2-train-*.out)")
    p.add_argument("--log2", type=Path, help="Second log for two-point fit with --known-batch/--known-batch2")
    p.add_argument("--known-batch", type=float, help="batch_size for --log")
    p.add_argument("--known-batch2", type=float, help="batch_size for --log2")
    p.add_argument(
        "--batch-idx-filter",
        type=int,
        default=None,
        help="Only consider lines containing this batch_idx (e.g. 19 for last batch in cap-20 epochs)",
    )
    p.add_argument(
        "--target-gb",
        type=float,
        default=240.0,
        help="Desired peak reserved GiB (leave headroom below hardware HBM)",
    )
    args = p.parse_args()

    if args.batch_a is not None and args.gb_a is not None and args.batch_b is not None and args.gb_b is not None:
        b_hat = _linear_extrapolate(args.batch_a, args.gb_a, args.batch_b, args.gb_b, args.target_gb)
        if b_hat is None:
            print("error: invalid calibration (need distinct batch sizes and positive slope)", file=sys.stderr)
            return 2
        print(
            f"linear_fit: slope={(args.gb_b-args.gb_a)/(args.batch_b-args.batch_a):.3f} GiB per batch, "
            f"estimated_batch≈{b_hat:.1f} at {args.target_gb} GiB (floor for integer batch: {math.floor(b_hat)})"
        )
        print(
            "note: linear extrapolation often **over**-estimates usable batch; binary-search short jobs to confirm.",
            file=sys.stderr,
        )
        return 0

    if args.log and args.log2 and args.known_batch is not None and args.known_batch2 is not None:
        g1 = _peak_gb_from_log(args.log, batch_idx_only=args.batch_idx_filter)
        g2 = _peak_gb_from_log(args.log2, batch_idx_only=args.batch_idx_filter)
        b_hat = _linear_extrapolate(args.known_batch, g1, args.known_batch2, g2, args.target_gb)
        print(f"parsed peak GiB: log1={g1:.2f} (batch={args.known_batch}), log2={g2:.2f} (batch={args.known_batch2})")
        if b_hat is None:
            print("error: could not extrapolate (check logs / batch sizes)", file=sys.stderr)
            return 2
        print(
            f"linear_fit: slope={(g2-g1)/(args.known_batch2-args.known_batch):.3f} GiB per batch, "
            f"estimated_batch≈{b_hat:.1f} at {args.target_gb} GiB (floor: {math.floor(b_hat)})"
        )
        print("note: confirm with binary search or OOM probe — do not trust extrapolation alone.", file=sys.stderr)
        return 0

    if args.log:
        g = _peak_gb_from_log(args.log, batch_idx_only=args.batch_idx_filter)
        print(f"peak_reserved_gib={g:.2f} from {args.log}")
        if args.known_batch is None:
            print(
                "hint: pass --known-batch and --log2/--known-batch2 for two-point estimate, "
                "or use --batch-a/--gb-a/--batch-b/--gb-b.",
                file=sys.stderr,
            )
        return 0

    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
