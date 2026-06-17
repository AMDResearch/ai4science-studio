#!/usr/bin/env python3
"""Parse ORBIT-2 training metrics from SLURM stdout (orbit2-train-*.out).

Two upstream log formats are supported:

1. **intermediate_downscaling.py** (studio path), rank 0:
     Starting epoch {epoch}
     Batch {idx}: {seconds} seconds   (every 10 batches — wall time for that step)
     Epoch {epoch} completed. Loss: {value}

2. **Bayes-CAST EDM `launch/train_edm.py`** (perf-optimizer-loop default), rank 0:
     epoch  {epoch}
     epoch:  {epoch} batch_idx {idx} world_rank 0  loss  tensor({value}, device='cuda:0')
     my rank 0. tic4-tic1 in {seconds} seconds
   i.e. there is **no** "Batch N: seconds" or "Epoch N completed" line — per-step
   wall time comes from the ``tic4-tic1`` line (emitted right after each step's loss
   line) and per-epoch loss is synthesized from the per-step losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Format 1 (intermediate_downscaling)
BATCH_RE = re.compile(r"^Batch (\d+): ([\d.]+) seconds")
EPOCH_DONE_RE = re.compile(r"^Epoch (\d+) completed\. Loss: ([\d.eE+-]+)")
EPOCH_START_RE = re.compile(r"^Starting epoch (\d+)")

# Format 2 (Bayes-CAST EDM train_edm.py)
EDM_STEP_RE = re.compile(
    r"^epoch:\s+(\d+)\s+batch_idx\s+(\d+)\s+world_rank\s+\d+\s+loss\s+tensor\(\s*([\d.eE+-]+)"
)
EDM_TIME_RE = re.compile(r"tic4-tic1 in\s+([\d.]+)\s+seconds")
# Real per-step per-rank batch dim. The EDM logs print the input tensor shape right
# before each step's loss line: "y.shape torch.Size([<batch>, ...". A dataloader lever
# (num_workers, sharding) can make the *realized* batch differ from trainer.batch_size
# (e.g. per-worker partial trailing batches), so throughput MUST use this real dim, not
# the nominal global_batch_size. See loop overnight-3x-001 iter-1 + clean nw test 10504/10505.
EDM_YSHAPE_RE = re.compile(r"^y\.shape torch\.Size\(\[(\d+)")


def _synthesize_edm_epoch_rows(batch_rows: list[dict]) -> list[dict]:
    """Build per-epoch loss records (mean of per-step losses) for loss-sanity on EDM logs."""
    by_epoch: dict[int, list[float]] = {}
    for r in batch_rows:
        if r.get("epoch") is None or r.get("loss") is None:
            continue
        by_epoch.setdefault(r["epoch"], []).append(r["loss"])
    rows: list[dict] = []
    for epoch in sorted(by_epoch):
        losses = by_epoch[epoch]
        rows.append(
            {
                "kind": "epoch",
                "batch": None,
                "batch_time_s": None,
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
            }
        )
    return rows


def parse_slurm_log(log_path: Path) -> list[dict]:
    records: list[dict] = []
    current_epoch: int | None = None
    edm_batch_rows: list[dict] = []
    saw_upstream_epoch = False
    pending_batch_samples: int | None = None  # last y.shape[0] seen (real per-rank batch)

    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            m = EPOCH_START_RE.match(line)
            if m:
                current_epoch = int(m.group(1))
                continue

            m = EDM_YSHAPE_RE.match(line)
            if m:
                pending_batch_samples = int(m.group(1))
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
                saw_upstream_epoch = True
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
                continue

            # Format 2: Bayes-CAST EDM per-step loss line
            m = EDM_STEP_RE.match(line)
            if m:
                row = {
                    "kind": "batch",
                    "batch": int(m.group(2)),
                    "batch_time_s": None,
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(3)),
                    "batch_samples": pending_batch_samples,
                }
                records.append(row)
                edm_batch_rows.append(row)
                continue

            # Format 2: Bayes-CAST EDM per-step wall time (follows the step's loss line).
            m = EDM_TIME_RE.search(line)
            if m and edm_batch_rows:
                for row in reversed(edm_batch_rows):
                    if row["batch_time_s"] is None:
                        row["batch_time_s"] = float(m.group(1))
                        break
                continue

    # EDM logs have no "Epoch N completed" line — synthesize per-epoch loss rows for sanity.
    if edm_batch_rows and not saw_upstream_epoch:
        records.extend(_synthesize_edm_epoch_rows(edm_batch_rows))
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
    global_batch_size: int | None = None,
    nominal_per_rank_batch: int | None = None,
) -> dict:
    """Primary FOM for sysopt: throughput_samples_per_s when global_batch_size is set.

    global_batch_size = trainer batch × fsdp × simple_ddp (samples per optimizer step).

    When the log carries the real per-step per-rank batch dim (``batch_samples``, from the
    EDM ``y.shape`` line) and ``nominal_per_rank_batch`` is known, throughput is computed
    from REAL samples processed (``Σ batch_samples × ranks / Σ step_time``) rather than the
    nominal ``global_batch_size × n_steps``. This is mandatory because dataloader levers can
    emit partial trailing batches (e.g. num_workers>1 with per-worker sharding): those steps
    finish faster but process fewer samples, so the nominal formula reports a phantom speedup.
    See clean num_workers test (jobs 10504 nw=1 vs 10505 nw=4): nominal said +80%, real ≈ 0%.
    """
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

    total_steady_wall_s = (
        sum(r["batch_time_s"] for r in steady_batches) if steady_batches else None
    )

    # ranks (= fsdp × simple_ddp) recovered from nominal global / per-rank batch.
    ranks: int | None = None
    if (
        global_batch_size is not None
        and nominal_per_rank_batch
        and nominal_per_rank_batch > 0
    ):
        ranks = max(1, round(global_batch_size / nominal_per_rank_batch))

    # Are real per-step batch dims available for ALL steady steps? Only then is the honest
    # (real-samples) throughput computable; otherwise fall back to the nominal estimate.
    steady_have_real = [
        r for r in steady_batches if r.get("batch_samples") is not None
    ]
    real_throughput_ok = (
        ranks is not None
        and steady_batches
        and len(steady_have_real) == len(steady_batches)
        and total_steady_wall_s
        and total_steady_wall_s > 0
    )

    throughput: float | None = None
    throughput_method: str | None = None
    partial_step_fraction: float | None = None
    if real_throughput_ok:
        real_samples_steady = sum(r["batch_samples"] * ranks for r in steady_batches)
        throughput = real_samples_steady / total_steady_wall_s
        throughput_method = "real_per_step_batch"
        n_partial = sum(
            1 for r in steady_batches if r["batch_samples"] < nominal_per_rank_batch
        )
        partial_step_fraction = round(n_partial / len(steady_batches), 3)
    elif (
        global_batch_size is not None
        and global_batch_size > 0
        and total_steady_wall_s is not None
        and total_steady_wall_s > 0
    ):
        samples_steady = global_batch_size * len(steady_batches)
        throughput = samples_steady / total_steady_wall_s
        throughput_method = "nominal_global_batch"

    primary_fom = "throughput_samples_per_s" if throughput is not None else "batch_time_s"
    primary_value = throughput if throughput is not None else steady_mean

    sanity = check_loss_sanity(epoch_rows)

    steady_epoch_losses = [r for r in epoch_rows if r["epoch"] >= steady_epoch_start]
    final_loss = (
        steady_epoch_losses[-1]["loss"]
        if steady_epoch_losses
        else (epoch_rows[-1]["loss"] if epoch_rows else None)
    )

    # Batches/epoch is an integrity signal: throughput uses the *nominal* global_batch_size,
    # so if a lever (e.g. num_workers) silently changes the dataloader's effective per-step
    # batch, batches/epoch shifts and HBM drops — flagging that a cross-run throughput
    # comparison is INVALID (the apparent speedup is a measurement artifact, not real work).
    # See loop overnight-3x-001 iter-1 anomaly (num_workers=4 -> phantom +129%).
    epoch_batch_counts: dict[int, int] = {}
    for r in batch_rows:
        ep = r.get("epoch")
        if ep is not None:
            epoch_batch_counts[ep] = epoch_batch_counts.get(ep, 0) + 1
    max_batches_per_epoch = max(epoch_batch_counts.values()) if epoch_batch_counts else None

    # Distinct realized per-rank batch sizes across steady steps (integrity surface).
    steady_real_dims = sorted(
        {r["batch_samples"] for r in steady_batches if r.get("batch_samples") is not None}
    )

    return {
        "primary_fom": primary_fom,
        "primary_value": primary_value,
        "throughput_samples_per_s": throughput,
        "throughput_method": throughput_method,
        "partial_step_fraction": partial_step_fraction,
        "ranks_inferred": ranks,
        "nominal_per_rank_batch": nominal_per_rank_batch,
        "steady_realized_batch_dims": steady_real_dims or None,
        "global_batch_size": global_batch_size,
        "batch_time_s": steady_mean,
        "steady_batch_time_s": steady_mean,
        "warmup_batch_time_s": warmup_mean,
        "steady_batch_count": len(steady_batches),
        "warmup_batch_count": len(warmup_batches),
        "max_batches_per_epoch": max_batches_per_epoch,
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
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="trainer.batch_size × fsdp × simple_ddp for throughput_samples_per_s",
    )
    parser.add_argument(
        "--nominal-per-rank-batch",
        type=int,
        default=None,
        help="trainer.batch_size (per-rank); enables honest real-per-step throughput",
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
                fieldnames=["kind", "batch", "batch_time_s", "epoch", "loss", "batch_samples"],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(records)

    foms = aggregate_foms(
        records,
        warmup_batches_per_epoch=args.warmup_batches_per_epoch,
        steady_epoch_start=args.steady_epoch_start,
        global_batch_size=args.global_batch_size,
        nominal_per_rank_batch=args.nominal_per_rank_batch,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(foms, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(foms, indent=2))
    return 0 if foms.get("loss_sanity_pass") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
