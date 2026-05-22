#!/usr/bin/env python3
"""parse_convergence.py — Extract training metrics from HydraGNN SLURM logs or TensorBoard event files.

Usage:
    # From SLURM stdout log:
    python parse_convergence.py --log hydragnn-train-6294.out --output convergence.csv

    # From TensorBoard log directory:
    python parse_convergence.py --tbdir logs/hydragnn-train-6294-N1/ --output convergence.csv

    # Generate a plot:
    python parse_convergence.py --log hydragnn-train-6294.out --plot convergence.png

Does NOT modify HydraGNN source code. Parses whatever HydraGNN already emits.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


def parse_slurm_log(log_path: str) -> list[dict]:
    """Parse HydraGNN training metrics from SLURM stdout or run.log.

    HydraGNN (pinned SHA 6c45f168) emits these patterns:
      - Epoch summary (requires HYDRAGNN_VALTEST=1):
            0: Epoch: 01, Train Loss: 0.12345678, Val Loss: 0.23456789, Test Loss: 0.34567890
      - Per-task losses:
            0: Tasks Train Loss: [0.123, 0.456]
      - Timing:
            0: Process 0 - Local timer:  train_validate_test  :  183.3
      - tqdm progress (always emitted):
            Train: 100%|...|50/50 [03:03<00:00,  3.66s/it]
    """
    records = []

    # Epoch summary: "Epoch: XX, Train Loss: X.XX, Val Loss: X.XX, Test Loss: X.XX"
    epoch_loss_pattern = re.compile(
        r"^0:\s*Epoch:\s*(\d+),\s*Train Loss:\s*([\d.eE+-]+),\s*"
        r"Val Loss:\s*([\d.eE+-]+),\s*Test Loss:\s*([\d.eE+-]+)",
    )

    # Per-task losses: "Tasks Train Loss: [0.123, 0.456]"
    tasks_loss_pattern = re.compile(
        r"^0:\s*Tasks\s+(Train|Val|Test)\s+Loss:\s*\[(.*?)\]",
    )

    # Timer lines: "0: Process 0 - Local timer:  train_validate_test  :  183.3"
    timer_pattern = re.compile(
        r"^0:\s*(?:Process 0 - )?(?:Local )?timer:\s*([\w]+)\s*:\s*([\d.]+)",
    )

    # tqdm progress bar: "Train: 100%|...|50/50 [03:03<00:00,  3.66s/it]"
    tqdm_pattern = re.compile(
        r"Train:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]",
    )

    # Memory line (per-batch indicator): "0: Max memory allocated after optimizer step: X.XX GB Y.YY GB"
    mem_pattern = re.compile(
        r"^0:\s*Max memory allocated after optimizer step:\s*([\d.]+)\s*GB\s*([\d.]+)\s*GB",
    )

    current_epoch = 0
    batch_count = 0
    last_tasks_epoch = None

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()

            m = epoch_loss_pattern.match(line)
            if m:
                rec = {
                    "epoch": int(m.group(1)),
                    "batch": None,
                    "train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "test_loss": float(m.group(4)),
                    "tasks_train": None,
                    "tasks_val": None,
                    "wall_time_s": None,
                    "source": "epoch_summary",
                }
                records.append(rec)
                last_tasks_epoch = rec
                continue

            m = tasks_loss_pattern.match(line)
            if m and last_tasks_epoch:
                split_name = m.group(1).lower()
                values = [float(x.strip()) for x in m.group(2).split(",")]
                key = f"tasks_{split_name}"
                last_tasks_epoch[key] = values
                continue

            m = tqdm_pattern.search(line)
            if m:
                pct = int(m.group(1))
                done = int(m.group(2))
                total = int(m.group(3))
                timing_str = m.group(4)
                rate_match = re.search(r"([\d.]+)(s|ms)/it", timing_str)
                rate_s = None
                if rate_match:
                    rate_s = float(rate_match.group(1))
                    if rate_match.group(2) == "ms":
                        rate_s /= 1000.0
                if pct == 100:
                    rec = {
                        "epoch": current_epoch,
                        "batch": done,
                        "train_loss": None,
                        "val_loss": None,
                        "test_loss": None,
                        "tasks_train": None,
                        "tasks_val": None,
                        "wall_time_s": rate_s,
                        "source": "tqdm_final",
                    }
                    records.append(rec)
                    current_epoch += 1
                continue

            m = mem_pattern.match(line)
            if m:
                batch_count += 1
                continue

            m = timer_pattern.match(line)
            if m:
                timer_name = m.group(1)
                timer_val = float(m.group(2))
                if timer_name == "train_validate_test" and records:
                    for rec in reversed(records):
                        if rec.get("source") in ("epoch_summary", "tqdm_final"):
                            rec["wall_time_s"] = timer_val
                            break
                continue

    # Add batch count metadata if available
    if batch_count > 0 and records:
        records[0].setdefault("total_batches_observed", batch_count)

    return records


def parse_tensorboard(tb_dir: str) -> list[dict]:
    """Parse TensorBoard event files for scalar metrics."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard package required for --tbdir mode.", file=sys.stderr)
        print("  pip install tensorboard", file=sys.stderr)
        sys.exit(1)

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"WARNING: No scalar tags found in {tb_dir}", file=sys.stderr)
        return []

    records = []
    for tag in tags:
        events = ea.Scalars(tag)
        for ev in events:
            records.append({
                "tag": tag,
                "step": ev.step,
                "value": ev.value,
                "wall_time_s": ev.wall_time,
            })

    records.sort(key=lambda r: (r["step"], r["tag"]))
    return records


def write_csv(records: list[dict], output_path: str, mode: str = "slurm"):
    """Write records to CSV."""
    if not records:
        print("WARNING: No records to write.", file=sys.stderr)
        return

    if mode == "slurm":
        fieldnames = [
            "epoch", "batch", "train_loss", "val_loss", "test_loss",
            "tasks_train", "tasks_val", "wall_time_s", "source",
        ]
    else:
        fieldnames = ["step", "tag", "value", "wall_time_s"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k) for k in fieldnames}
            if row.get("tasks_train") and isinstance(row["tasks_train"], list):
                row["tasks_train"] = ";".join(f"{v:.8f}" for v in row["tasks_train"])
            if row.get("tasks_val") and isinstance(row["tasks_val"], list):
                row["tasks_val"] = ";".join(f"{v:.8f}" for v in row["tasks_val"])
            writer.writerow(row)

    print(f"Wrote {len(records)} records to {output_path}")


def plot_convergence(records: list[dict], plot_path: str, mode: str = "slurm"):
    """Generate a convergence plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required for --plot mode.", file=sys.stderr)
        print("  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    if mode == "slurm":
        epoch_records = [r for r in records if r.get("source") == "epoch_summary"]
        if not epoch_records:
            epoch_records = [r for r in records if r.get("train_loss") is not None]

        if not epoch_records:
            print("WARNING: No training loss values found for plotting.", file=sys.stderr)
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = [r["epoch"] for r in epoch_records]
        train_loss = [r["train_loss"] for r in epoch_records]
        ax.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)

        val_data = [(r["epoch"], r["val_loss"]) for r in epoch_records
                    if r.get("val_loss") is not None]
        if val_data:
            val_x, val_y = zip(*val_data)
            ax.plot(val_x, val_y, "r-s", label="Val Loss", markersize=4)

        test_data = [(r["epoch"], r["test_loss"]) for r in epoch_records
                     if r.get("test_loss") is not None]
        if test_data:
            test_x, test_y = zip(*test_data)
            ax.plot(test_x, test_y, "g-^", label="Test Loss", markersize=4)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MAE)")
        ax.set_title("HydraGNN Training Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if all(v > 0 for v in train_loss):
            ax.set_yscale("log")

    else:
        tags = sorted(set(r["tag"] for r in records))
        fig, ax = plt.subplots(figsize=(10, 6))
        for tag in tags:
            tag_data = [(r["step"], r["value"]) for r in records if r["tag"] == tag]
            if tag_data:
                steps, values = zip(*tag_data)
                ax.plot(steps, values, label=tag, markersize=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("HydraGNN Training Convergence (TensorBoard)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def aggregate_foms_from_records(records: list[dict]) -> dict:
    """Aggregate per-epoch records into the FOM JSON consumed by the
    perf-optimizer-loop's fom_extractor subagent.

    Returns a dict with:
      num_epochs_completed, mean_epoch_time_excluding_epoch_0,
      epoch_times_s, final_train_loss, final_val_loss.
    Any field may be None if the underlying data is absent (e.g. HYDRAGNN_VALTEST=0
    omits val/test losses; older HydraGNN may omit per-epoch wall times).
    """
    epoch_summary_records = [r for r in records if r.get("source") == "epoch_summary"]
    tqdm_records = [r for r in records if r.get("source") == "tqdm_final"]

    # Prefer epoch_summary wall_time_s only when it provides per-epoch values
    # (>= 2 of them). HydraGNN's existing log format usually attaches the
    # "train_validate_test" timer to only the FINAL epoch (cumulative whole-job
    # wall), which is useless for per-epoch FOMs. In that case, fall through to
    # the per-epoch tqdm_final records which always provide s/it × batches.
    epoch_times: list[float] = []
    summary_walls = [float(r["wall_time_s"]) for r in epoch_summary_records
                     if r.get("wall_time_s") is not None]
    if len(summary_walls) >= 2:
        epoch_times = summary_walls
    elif tqdm_records:
        for r in tqdm_records:
            rate = r.get("wall_time_s")
            batch = r.get("batch")
            if rate is not None and batch:
                epoch_times.append(float(rate) * float(batch))
    elif summary_walls:
        epoch_times = summary_walls  # single epoch; better than nothing

    if epoch_times:
        if len(epoch_times) >= 2:
            mean_excluding_epoch_0 = sum(epoch_times[1:]) / float(len(epoch_times) - 1)
            warning = None
        else:
            mean_excluding_epoch_0 = epoch_times[0]
            warning = "single epoch, epoch_0 not dropped"
    else:
        mean_excluding_epoch_0 = None
        warning = "no epoch wall_time_s present in log"

    final_train_loss = None
    final_val_loss = None
    if epoch_summary_records:
        final_train_loss = epoch_summary_records[-1].get("train_loss")
        final_val_loss = epoch_summary_records[-1].get("val_loss")

    foms = {
        "num_epochs_completed": len(epoch_summary_records) if epoch_summary_records else len(tqdm_records),
        "mean_epoch_time_excluding_epoch_0": mean_excluding_epoch_0,
        "epoch_times_s": epoch_times if epoch_times else None,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        # Note: total_batches_observed is intentionally NOT derived from the log
        # here. The existing HydraGNN entrypoint emits the "Max memory allocated"
        # line ONCE per job on rank 0, not per batch. The fom_extractor subagent
        # computes samples_processed from manifest values
        # (num_epochs * max_num_batch * batch_size * ranks) which is the
        # correct denominator for throughput / energy_per_sample FOMs.
    }
    if warning:
        foms["warning"] = warning
    return foms


def main():
    parser = argparse.ArgumentParser(
        description="Extract convergence metrics from HydraGNN training runs."
    )
    parser.add_argument(
        "--log", type=str, help="Path to SLURM stdout log file (hydragnn-train-*.out)"
    )
    parser.add_argument(
        "--tbdir", type=str, help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="convergence.csv",
        help="Output CSV file path (default: convergence.csv)"
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="If set, generate a convergence plot at this path (e.g. convergence.png)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also write a JSON summary alongside the CSV"
    )
    parser.add_argument(
        "--json-foms", type=str, default=None,
        help=("If set, write an aggregated FOM JSON to this path "
              "(consumed by perf-optimizer-loop's fom_extractor subagent). "
              "Includes num_epochs_completed, mean_epoch_time_excluding_epoch_0, "
              "epoch_times_s, final_train_loss, final_val_loss."),
    )

    args = parser.parse_args()

    if not args.log and not args.tbdir:
        parser.error("Must specify either --log or --tbdir")

    if args.log and args.tbdir:
        parser.error("Specify only one of --log or --tbdir")

    if args.log:
        if not os.path.isfile(args.log):
            print(f"ERROR: Log file not found: {args.log}", file=sys.stderr)
            sys.exit(1)
        records = parse_slurm_log(args.log)
        mode = "slurm"
    else:
        if not os.path.isdir(args.tbdir):
            print(f"ERROR: TensorBoard dir not found: {args.tbdir}", file=sys.stderr)
            sys.exit(1)
        records = parse_tensorboard(args.tbdir)
        mode = "tensorboard"

    if not records:
        print("WARNING: No metrics found. The log may not contain training output.",
              file=sys.stderr)
        print("  (Jobs that were cancelled before training started won't have metrics.)",
              file=sys.stderr)
        sys.exit(0)

    write_csv(records, args.output, mode=mode)

    if args.json:
        json_path = Path(args.output).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"JSON written to {json_path}")

    if args.plot:
        plot_convergence(records, args.plot, mode=mode)

    if args.json_foms:
        if mode != "slurm":
            print("WARNING: --json-foms only supported with --log mode; skipping.",
                  file=sys.stderr)
        else:
            foms = aggregate_foms_from_records(records)
            with open(args.json_foms, "w") as f:
                json.dump(foms, f, indent=2)
            print(f"FOMs JSON written to {args.json_foms}")

    # Print summary
    if mode == "slurm" and records:
        train_values = [r["train_loss"] for r in records
                        if r.get("train_loss") is not None]
        if train_values:
            print(f"\nSummary:")
            print(f"  Total records: {len(records)}")
            print(f"  Epochs with loss data: {len(train_values)}")
            print(f"  Train Loss range: {min(train_values):.8f} -> {max(train_values):.8f}")
            if len(train_values) > 1:
                reduction = (train_values[0] - train_values[-1]) / train_values[0] * 100
                print(f"  Reduction: {reduction:.1f}%")
        else:
            tqdm_recs = [r for r in records if r.get("source") == "tqdm_final"]
            if tqdm_recs:
                print(f"\nSummary (no epoch-level loss — run with HYDRAGNN_VALTEST=1):")
                print(f"  Epochs completed: {len(tqdm_recs)}")
                rates = [r["wall_time_s"] for r in tqdm_recs if r.get("wall_time_s")]
                if rates:
                    print(f"  Avg time per batch: {sum(rates)/len(rates):.2f} s/it")


if __name__ == "__main__":
    main()
