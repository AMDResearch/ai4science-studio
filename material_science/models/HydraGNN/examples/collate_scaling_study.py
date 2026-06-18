#!/usr/bin/env python3
"""collate_scaling_study.py — Build DCSWEAP-4502 scaling table from HydraGNN SLURM logs.

Parses one or more hydragnn-train-*.out files (or job IDs resolved under a log dir)
and emits a CSV + Markdown summary with throughput, epoch time, scaling efficiency,
and loss sanity checks.

Usage:
    python collate_scaling_study.py \\
        --log hydragnn-train-<jobid>.out \\
        --log 2node:hydragnn-train-<jobid>.out \\
        -o scaling_study

    python collate_scaling_study.py --log-dir . --jobs <id1>,<id2>,... -o scaling_study
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Reuse the SLURM log parser from parse_convergence.py in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_convergence import aggregate_foms_from_records, parse_slurm_log  # noqa: E402


def _parse_job_header(log_path: Path) -> dict:
    """Extract run configuration from the sbatch banner at the top of the log."""
    meta: dict = {"log_path": str(log_path)}
    patterns = {
        "nodes": r"^\s*Nodes\s+:\s*(\d+)",
        "gpus_per_node": r"^\s*GPUs/node\s+:\s*(\d+)",
        "total_ranks": r"^\s*Total ranks\s+:\s*(\d+)",
        "batch_size": r"^\s*Batch size\s+:\s*(\d+)",
        "num_epochs": r"^\s*Num epochs\s+:\s*(\d+)",
        "max_batches": r"^\s*Max batches\s+:\s*(\S+)",
        "precision": r"^\s*Precision\s+:\s*(\S+)",
        "nodelist": r"^\s*Node\(s\)\s+:\s*(.+)",
        "rccl_priority": r"^\s*RCCL priority:\s*TORCH_NCCL_HIGH_PRIORITY=(\d+)",
        "hw_queues": r"^\s*HW queues\s+:\s*GPU_MAX_HW_QUEUES=(\d+)",
    }
    job_id_m = re.search(r"hydragnn-train-(\d+)", log_path.name)
    if job_id_m:
        meta["job_id"] = job_id_m.group(1)

    with log_path.open() as f:
        for line in f:
            for key, pat in patterns.items():
                m = re.match(pat, line)
                if m:
                    meta[key] = m.group(1).strip()
            if line.startswith("Welcome to SLURM") and "nodes" in meta:
                break

    if "nodes" in meta:
        meta["nodes"] = int(meta["nodes"])
    if "gpus_per_node" in meta:
        meta["gpus_per_node"] = int(meta["gpus_per_node"])
    if "total_ranks" in meta:
        meta["total_ranks"] = int(meta["total_ranks"])
    if "batch_size" in meta:
        meta["batch_size"] = int(meta["batch_size"])
    if "num_epochs" in meta:
        meta["num_epochs"] = int(meta["num_epochs"])
    if "max_batches" in meta and meta["max_batches"].isdigit():
        meta["max_batches"] = int(meta["max_batches"])
    return meta


def _all_train_epoch_rates(log_path: Path, num_epochs: int | None = None) -> list[float]:
    """Return per-epoch train s/it (instantaneous rate at Train 100%, not cumulative avg).

    HydraGNN tqdm emits two Train 100% rates on one line; the first is the
    instantaneous rate at completion, the second includes epoch startup overhead.
    """
    pat = re.compile(
        r"Train:\s*100%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*?,\s*([\d.]+)s/it\]"
    )
    rates: list[float] = []
    with log_path.open(errors="replace") as f:
        for line in f:
            if "Train:" not in line or "100%" not in line:
                continue
            ms = list(pat.finditer(line))
            if ms:
                # First match on the line = instantaneous rate at epoch end.
                rates.append(float(ms[0].group(3)))
    # HydraGNN may emit duplicate Train 100% lines per epoch; keep the last N.
    if num_epochs and len(rates) > num_epochs:
        rates = rates[-num_epochs:]
    return rates


def _train_tqdm_timing(log_path: Path, steady_start: int = 2, num_epochs: int | None = None) -> dict:
    """Steady-state train timing from multi-epoch Train tqdm lines."""
    rates = _all_train_epoch_rates(log_path, num_epochs=num_epochs)
    if not rates:
        return {}

    warmup = rates[:steady_start] if len(rates) > steady_start else []
    steady = (
        rates[steady_start:num_epochs]
        if num_epochs and len(rates) >= num_epochs
        else rates[steady_start:]
    )

    sec_per_batch = sum(steady) / len(steady)
    batches = None
    pat_batches = re.compile(
        r"Train:\s*100%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*?,\s*([\d.]+)s/it\]"
    )
    with log_path.open(errors="replace") as f:
        for line in f:
            m = pat_batches.search(line)
            if m:
                batches = int(m.group(1))

    epoch_time_s = sec_per_batch * batches if batches else None
    return {
        "sec_per_batch_train": sec_per_batch,
        "epoch_time_train_s": epoch_time_s,
        "batches_per_epoch": batches,
        "batches_per_sec_train": 1.0 / sec_per_batch if sec_per_batch else None,
        "steady_state_epochs": list(range(steady_start, steady_start + len(steady))),
        "per_epoch_train_s_it": rates,
        "warmup_epochs_s_it": warmup,
        "steady_epochs_s_it": steady,
    }


def _peak_gpu_mem_gb(log_path: Path) -> float | None:
    pat = re.compile(
        r"Max memory allocated after optimizer step:\s*([\d.]+)\s*GB"
    )
    peak = None
    with log_path.open(errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                peak = max(peak or 0.0, float(m.group(1)))
    return peak


def _epoch_timing(
    records: list[dict],
    max_batches: int | None,
    log_path: Path,
    steady_start: int = 2,
    num_epochs: int | None = None,
) -> dict:
    """Derive per-epoch and per-batch timing from parsed log records."""
    foms = aggregate_foms_from_records(records)
    epoch_times = foms.get("epoch_times_s") or []

    # Prefer steady-state (exclude epoch 0 warmup) when we have multiple epochs.
    if len(epoch_times) >= 2:
        steady_epochs = epoch_times[1:]
        epoch_time_s = sum(steady_epochs) / len(steady_epochs)
    elif epoch_times:
        epoch_time_s = epoch_times[0]
    else:
        epoch_time_s = None

    batches_per_epoch = max_batches
    if batches_per_epoch is None:
        tqdm_recs = [r for r in records if r.get("source") == "tqdm_final"]
        if tqdm_recs:
            batches_per_epoch = tqdm_recs[0].get("batch")

    sec_per_batch = None
    if epoch_time_s is not None and batches_per_epoch:
        sec_per_batch = epoch_time_s / float(batches_per_epoch)

    # Direct tqdm s/it average (often more stable for single-epoch runs).
    tqdm_rates = [
        float(r["wall_time_s"])
        for r in records
        if r.get("source") == "tqdm_final" and r.get("wall_time_s") is not None
    ]
    if tqdm_rates:
        tqdm_sec_per_batch = sum(tqdm_rates) / len(tqdm_rates)
        if sec_per_batch is None:
            sec_per_batch = tqdm_sec_per_batch
        elif len(tqdm_rates) >= 1:
            # Prefer tqdm for single-epoch strong-scaling runs.
            sec_per_batch = tqdm_sec_per_batch

    train_only = _train_tqdm_timing(log_path, steady_start=steady_start, num_epochs=num_epochs)
    if train_only.get("sec_per_batch_train"):
        sec_per_batch = train_only["sec_per_batch_train"]
        epoch_time_s = train_only.get("epoch_time_train_s", epoch_time_s)
        batches_per_epoch = train_only.get("batches_per_epoch", batches_per_epoch)

    batches_per_sec = (1.0 / sec_per_batch) if sec_per_batch else None
    peak_mem = _peak_gpu_mem_gb(log_path)

    return {
        "epoch_time_s": epoch_time_s,
        "epoch_time_train_s": train_only.get("epoch_time_train_s"),
        "sec_per_batch": sec_per_batch,
        "batches_per_sec": batches_per_sec,
        "batches_per_epoch": batches_per_epoch,
        "peak_gpu_mem_gb": peak_mem,
        "final_train_loss": foms.get("final_train_loss"),
        "final_val_loss": foms.get("final_val_loss"),
        "num_epochs_completed": foms.get("num_epochs_completed"),
        "steady_state_epoch_start": steady_start,
        "steady_epochs_s_it": train_only.get("steady_epochs_s_it"),
    }


def analyze_log(log_path: Path, label: str | None = None, steady_start: int = 2) -> dict:
    records = parse_slurm_log(str(log_path))
    meta = _parse_job_header(log_path)
    timing = _epoch_timing(
        records,
        meta.get("max_batches") if isinstance(meta.get("max_batches"), int) else None,
        log_path,
        steady_start,
        meta.get("num_epochs") if isinstance(meta.get("num_epochs"), int) else None,
    )

    row = {**meta, **timing}
    if label:
        row["label"] = label
    row["batch_size"] = meta.get("batch_size", 200)
    if row.get("batches_per_sec") and row.get("batch_size"):
        row["samples_per_sec"] = row["batches_per_sec"] * row["batch_size"]
    return row


def _scaling_efficiency(rows: list[dict]) -> None:
    """Add strong-scaling efficiency vs 1-node baseline (in-place)."""
    baseline = None
    for r in sorted(rows, key=lambda x: x.get("nodes", 0)):
        if r.get("nodes") == 1 and r.get("sec_per_batch"):
            baseline = r["sec_per_batch"]
            break
    if baseline is None:
        return
    for r in rows:
        if r.get("sec_per_batch") and r.get("nodes"):
            ideal = baseline  # strong scaling: same wall time per batch regardless of nodes
            actual = r["sec_per_batch"]
            r["strong_scaling_efficiency"] = ideal / actual if actual else None
            r["speedup_vs_1node"] = baseline / actual if actual else None


def _resolve_log(path_or_job: str, log_dir: Path) -> Path:
    p = Path(path_or_job)
    if p.is_file():
        return p
    if ":" in path_or_job:
        _label, rest = path_or_job.split(":", 1)
        return Path(rest)
    # Treat as job id
    candidates = list(log_dir.glob(f"hydragnn-train-{path_or_job}.out"))
    if not candidates:
        candidates = list(log_dir.glob(f"**/hydragnn-train-{path_or_job}.out"))
    if not candidates:
        raise FileNotFoundError(f"No log for job {path_or_job} under {log_dir}")
    return candidates[0]


def write_outputs(rows: list[dict], stem: Path, steady_start: int = 2) -> None:
    _scaling_efficiency(rows)
    rows_sorted = sorted(rows, key=lambda r: r.get("nodes", 0))

    csv_path = stem.with_suffix(".csv")
    fieldnames = [
        "job_id", "label", "nodes", "total_ranks", "batch_size", "max_batches",
        "precision", "batches_per_epoch", "sec_per_batch", "epoch_time_train_s",
        "batches_per_sec", "samples_per_sec", "epoch_time_s", "peak_gpu_mem_gb",
        "final_train_loss", "final_val_loss",
        "strong_scaling_efficiency", "speedup_vs_1node", "nodelist",
        "rccl_priority", "hw_queues", "log_path",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    md_path = stem.with_suffix(".md")
    lines = [
        "# HydraGNN MI355X Scaling Study (DCSWEAP-4502)",
        "",
        "Strong-scaling configuration: fixed 50 batches/epoch, batch_size=200, fp64, "
        "ANI1x+Alexandria, MPI/ADIOS phase-1, `HYDRAGNN_VALTEST=0` (train-only timing).",
        "",
        "**Steady-state metric:** mean train s/batch over epochs "
        f"{steady_start}+ (exclude warmup epochs 0–{steady_start - 1}).",
        "",
        "| Nodes | GPUs | Job | Train s/batch | Train epoch (s) | samples/s | "
        "Peak GPU mem (GB) | Train loss | Val loss | Strong-scaling eff. |",
        "|------:|-----:|----:|--------------:|----------------:|----------:|"
        "------------------:|-----------:|---------:|--------------------:|",
    ]
    for r in rows_sorted:
        train_epoch = r.get("epoch_time_train_s") or r.get("epoch_time_s") or 0
        tl = r.get("final_train_loss")
        vl = r.get("final_val_loss")
        tl_s = f"{tl:.4f}" if tl is not None else "—"
        vl_s = f"{vl:.4f}" if vl is not None else "—"
        mem = r.get("peak_gpu_mem_gb") or 0
        lines.append(
            f"| {r.get('nodes', '?')} "
            f"| {r.get('total_ranks', '?')} "
            f"| {r.get('job_id', '?')} "
            f"| {r.get('sec_per_batch', 0):.2f} "
            f"| {train_epoch:.0f} "
            f"| {r.get('samples_per_sec', 0):.0f} "
            f"| {mem:.1f} "
            f"| {tl_s} "
            f"| {vl_s} "
            f"| {r.get('strong_scaling_efficiency', 0):.2f} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- **All runs share:** `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, "
        "`HG_NUM_EPOCH=6`, `HYDRAGNN_MAX_NUM_BATCH=50`, `HYDRAGNN_VALTEST=0`.",
        f"- **Steady-state:** mean of train s/it for epochs {steady_start}–5 (0-indexed).",
        "- **GPU utilization**: optional; SLURM log timing is the primary metric.",
        "- **Generated outputs** (`scaling_study*.csv/json/md`) are gitignored run artifacts.",
        "",
    ])
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")

    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(rows_sorted, indent=2, default=str))
    print(f"Wrote {json_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collate HydraGNN scaling study results.")
    ap.add_argument("--log", action="append", default=[], metavar="JOB[:PATH]",
                    help="Job id (resolved under --log-dir) or label:path")
    ap.add_argument("--log-dir", default=".", help="Directory to search for hydragnn-train-<id>.out")
    ap.add_argument("--jobs", default="", help="Comma-separated job ids (shorthand for --log)")
    ap.add_argument("--steady-start", type=int, default=2,
                    help="First epoch index (0-based) included in steady-state average (default: 2)")
    ap.add_argument("-o", "--output", default="scaling_study", help="Output stem (no extension)")
    args = ap.parse_args()

    specs = list(args.log)
    if args.jobs:
        specs.extend(j.strip() for j in args.jobs.split(",") if j.strip())

    if not specs:
        ap.error("Provide --log and/or --jobs")

    log_dir = Path(args.log_dir)
    rows = []
    for spec in specs:
        label = None
        if ":" in spec and not Path(spec.split(":", 1)[1]).exists():
            # label:jobid
            label, job_part = spec.split(":", 1)
            log_path = _resolve_log(job_part, log_dir)
        elif ":" in spec:
            label, log_path_str = spec.split(":", 1)
            log_path = Path(log_path_str)
        else:
            log_path = _resolve_log(spec, log_dir)
        print(f"Parsing {log_path} ...")
        rows.append(analyze_log(log_path, label=label, steady_start=args.steady_start))

    write_outputs(rows, Path(args.output), steady_start=args.steady_start)


if __name__ == "__main__":
    main()
