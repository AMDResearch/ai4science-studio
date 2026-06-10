#!/usr/bin/env python3
"""Parse ORBIT-2 training metrics from SLURM stdout (orbit2-train-*.out).

Upstream intermediate_downscaling.py emits on rank 0:
  Starting epoch {epoch}
  Batch {idx}: {seconds} seconds   (every 10 batches — wall time for that step)
  Epoch {epoch} completed. Loss: {value}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BATCH_RE = re.compile(r"^Batch (\d+): ([\d.]+) seconds")
EPOCH_DONE_RE = re.compile(r"^Epoch (\d+) completed\. Loss: ([\d.eE+-]+)")
EPOCH_START_RE = re.compile(r"^Starting epoch (\d+)")


def parse_slurm_log(log_path: Path) -> list[dict]:
    records: list[dict] = []
    current_epoch: int | None = None

    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            m = EPOCH_START_RE.match(line)
            if m:
                current_epoch = int(m.group(1))
                continue

            m = BATCH_RE.match(line)
            if m:
                records.append(
                    {
                        "kind": "batch",
                        "batch": int(m.group(1)),
                        "batch_time_s": float(m.group(2)),
                        "epoch": current_epoch,
                        "loss": None,
                    }
                )
                continue

            m = EPOCH_DONE_RE.match(line)
            if m:
                epoch = int(m.group(1))
                records.append(
                    {
                        "kind": "epoch",
                        "batch": None,
                        "batch_time_s": None,
                        "epoch": epoch,
                        "loss": float(m.group(2)),
                    }
                )
                current_epoch = epoch + 1
    return records


def check_loss_sanity(
    epoch_losses: list[dict],
    *,
    require_strict_decrease: bool = True,
) -> dict:
    """Verify epoch losses decrease epoch-over-epoch (crash detector, not science gate)."""
    losses = [(r["epoch"], r["loss"]) for r in epoch_losses if r.get("loss") is not None]
    losses.sort(key=lambda x: x[0])

    violations: list[dict] = []
    for i in range(len(losses) - 1):
        e0, l0 = losses[i]
        e1, l1 = losses[i + 1]
        if require_strict_decrease and l1 >= l0:
            violations.append(
                {
                    "from_epoch": e0,
                    "to_epoch": e1,
                    "loss_from": l0,
                    "loss_to": l1,
                }
            )

    if len(losses) < 2:
        return {
            "loss_sanity_pass": None,
            "loss_sanity_skipped": True,
            "loss_curve": [{"epoch": e, "loss": l} for e, l in losses],
            "loss_violations": [],
            "loss_epoch_count": len(losses),
        }

    return {
        "loss_sanity_pass": len(violations) == 0,
        "loss_sanity_skipped": False,
        "loss_curve": [{"epoch": e, "loss": l} for e, l in losses],
        "loss_violations": violations,
        "loss_epoch_count": len(losses),
    }


def aggregate_foms(
    records: list[dict],
    *,
    warmup_batches_per_epoch: int = 1,
    steady_epoch_start: int = 2,
) -> dict:
    """Primary FOM: mean steady-state batch wall time (non-warmup epochs and batches)."""
    batch_rows = [r for r in records if r["kind"] == "batch" and r["batch_time_s"] is not None]
    epoch_rows = [r for r in records if r["kind"] == "epoch" and r["epoch"] is not None]

    warmup_batches = [
        r
        for r in batch_rows
        if r.get("epoch") is not None
        and r["epoch"] < steady_epoch_start
    ]
    steady_batches = [
        r
        for r in batch_rows
        if r.get("epoch") is not None
        and r["epoch"] >= steady_epoch_start
        and r["batch"] >= warmup_batches_per_epoch
    ]

    def _mean(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return sum(r["batch_time_s"] for r in rows) / len(rows)

    warmup_mean = _mean(
        [r for r in batch_rows if r.get("epoch") is not None and r["epoch"] < steady_epoch_start]
        + [r for r in batch_rows if r.get("epoch") is None]
    )
    steady_mean = _mean(steady_batches)

    sanity = check_loss_sanity(epoch_rows)

    steady_epoch_losses = [r for r in epoch_rows if r["epoch"] >= steady_epoch_start]
    final_loss = (
        steady_epoch_losses[-1]["loss"]
        if steady_epoch_losses
        else (epoch_rows[-1]["loss"] if epoch_rows else None)
    )

    return {
        "primary_fom": "batch_time_s",
        "batch_time_s": steady_mean,
        "steady_batch_time_s": steady_mean,
        "warmup_batch_time_s": warmup_mean,
        "steady_batch_count": len(steady_batches),
        "warmup_batch_count": len(warmup_batches),
        "steady_epoch_start": steady_epoch_start,
        "warmup_batches_per_epoch_excluded": warmup_batches_per_epoch,
        "final_loss": final_loss,
        "epoch_records": len(epoch_rows),
        **sanity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse ORBIT-2 training SLURM log.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Write CSV of parsed records")
    parser.add_argument("--json", type=Path, default=None, help="Write aggregated FOM JSON")
    parser.add_argument(
        "--steady-epoch-start",
        type=int,
        default=2,
        help="First epoch included in steady-state batch FOM (default: 2)",
    )
    parser.add_argument(
        "--warmup-batches-per-epoch",
        type=int,
        default=1,
        help="Exclude batch indices < N within each steady epoch (default: 1 = skip batch 0)",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"error: log not found: {args.log}", file=sys.stderr)
        return 2

    records = parse_slurm_log(args.log)
    if not records:
        print("warning: no batch/epoch records found", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["kind", "batch", "batch_time_s", "epoch", "loss"],
            )
            w.writeheader()
            w.writerows(records)

    foms = aggregate_foms(
        records,
        warmup_batches_per_epoch=args.warmup_batches_per_epoch,
        steady_epoch_start=args.steady_epoch_start,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(foms, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(foms, indent=2))
    return 0 if foms.get("loss_sanity_pass") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
