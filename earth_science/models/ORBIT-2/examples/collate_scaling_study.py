#!/usr/bin/env python3
"""Build ORBIT-2 strong-scaling table from orbit2-train-*.out logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_training_log import aggregate_foms, parse_slurm_log  # noqa: E402


def _parse_banner(log_path: Path) -> dict:
    meta: dict = {}
    patterns = {
        "nodes": r"^\s*Nodes\s+:\s*(\d+)",
        "total_ranks": r"^\s*Total ranks\s+:\s*(\d+)",
        "batch_size": r"^\s*Batch size\s+:\s*(\d+)",
        "max_epochs": r"^\s*Max epochs\s+:\s*(\d+)",
        "max_batches": r"^\s*Max batches\s+:\s*(\S+)",
        "nodelist": r"^\s*Node\(s\)\s+:\s*(.+)",
    }
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            for key, pat in patterns.items():
                m = re.match(pat, line)
                if m:
                    meta[key] = m.group(1).strip()
    if "nodes" in meta:
        meta["nodes"] = int(meta["nodes"])
    if "total_ranks" in meta:
        meta["total_ranks"] = int(meta["total_ranks"])
    m = re.search(r"orbit2-train-(\d+)", log_path.name)
    if m:
        meta["job_id"] = m.group(1)
    meta["log_path"] = str(log_path)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Collate ORBIT-2 scaling study logs.")
    parser.add_argument("--log", action="append", default=[], help="log file or N:path")
    parser.add_argument("--log-dir", type=Path, default=Path("."))
    parser.add_argument("--jobs", type=str, default="", help="Comma-separated job IDs")
    parser.add_argument("-o", "--output", type=str, default="scaling_study")
    parser.add_argument(
        "--steady-epoch-start",
        type=int,
        default=2,
        help="Steady-state FOM uses batch times from this epoch onward (default: 2)",
    )
    parser.add_argument(
        "--warmup-batches-per-epoch",
        type=int,
        default=1,
        help="Within steady epochs, exclude batch indices < N (default: 1)",
    )
    parser.add_argument(
        "--require-loss-sanity",
        action="store_true",
        help="Exit 1 if any job fails epoch-over-epoch loss decrease check",
    )
    args = parser.parse_args()

    logs: list[tuple[str, Path]] = []
    for spec in args.log:
        if ":" in spec:
            label, path = spec.split(":", 1)
            logs.append((label, Path(path)))
        else:
            logs.append(("", Path(spec)))

    if args.jobs:
        for jid in args.jobs.split(","):
            jid = jid.strip()
            p = args.log_dir / f"orbit2-train-{jid}.out"
            if p.is_file():
                logs.append((jid, p))

    if not logs:
        print("error: no logs found", file=sys.stderr)
        return 2

    rows = []
    baseline_throughput = None
    any_sanity_fail = False

    for label, log_path in logs:
        if not log_path.is_file():
            print(f"warning: missing {log_path}", file=sys.stderr)
            continue
        meta = _parse_banner(log_path)
        records = parse_slurm_log(log_path)
        foms = aggregate_foms(
            records,
            warmup_batches_per_epoch=args.warmup_batches_per_epoch,
            steady_epoch_start=args.steady_epoch_start,
        )
        if foms.get("loss_sanity_pass") is False:
            any_sanity_fail = True
            print(
                f"warning: job {label or meta.get('job_id')} failed loss sanity: "
                f"{foms.get('loss_violations')}",
                file=sys.stderr,
            )

        nodes = meta.get("nodes", 1)
        batch_s = foms.get("batch_time_s")
        batch_size = int(meta.get("batch_size", 4))
        ranks = meta.get("total_ranks", nodes * 8)
        throughput = (batch_size * ranks / batch_s) if batch_s else None
        if baseline_throughput is None and throughput:
            baseline_throughput = throughput
        efficiency = (
            (throughput / (baseline_throughput * nodes)) if throughput and baseline_throughput else None
        )
        rows.append(
            {
                "label": label or meta.get("job_id", ""),
                "nodes": nodes,
                "ranks": ranks,
                "batch_time_s": foms.get("batch_time_s"),
                "warmup_batch_time_s": foms.get("warmup_batch_time_s"),
                "steady_batch_count": foms.get("steady_batch_count"),
                "throughput_samples_per_s": throughput,
                "scaling_efficiency": efficiency,
                "final_loss": foms.get("final_loss"),
                "loss_sanity_pass": foms.get("loss_sanity_pass"),
                "loss_curve": json.dumps(foms.get("loss_curve", [])),
                "job_id": meta.get("job_id"),
                "log_path": str(log_path),
            }
        )

    rows.sort(key=lambda r: r["nodes"])
    out_base = Path(args.output)
    csv_path = out_base.with_suffix(".csv")
    md_path = out_base.with_suffix(".md")
    json_path = out_base.with_suffix(".json")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if rows:
            w.writeheader()
            w.writerows(rows)

    lines = [
        "# ORBIT-2 scaling study",
        "",
        f"Steady-state FOM: mean batch wall time from epoch {args.steady_epoch_start}+ "
        f"(exclude warmup epochs 0–{args.steady_epoch_start - 1}), "
        f"excluding batch indices < {args.warmup_batches_per_epoch} within each epoch.",
        "",
        "Loss sanity: each epoch loss must be strictly less than the previous epoch.",
        "",
        "| nodes | ranks | steady batch_time_s | warmup batch_time_s | throughput | efficiency | loss OK |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lp = r.get("loss_sanity_pass")
        ok = "yes" if lp is True else ("n/a" if lp is None else "**no**")
        lines.append(
            f"| {r['nodes']} | {r['ranks']} | {r.get('batch_time_s', 'n/a')} | "
            f"{r.get('warmup_batch_time_s', 'n/a')} | {r.get('throughput_samples_per_s', 'n/a')} | "
            f"{r.get('scaling_efficiency', 'n/a')} | {ok} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "steady_epoch_start": args.steady_epoch_start,
        "warmup_batches_per_epoch": args.warmup_batches_per_epoch,
        "rows": rows,
        "all_loss_sanity_pass": not any_sanity_fail,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}, {md_path}, and {json_path}")
    if args.require_loss_sanity and any_sanity_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
