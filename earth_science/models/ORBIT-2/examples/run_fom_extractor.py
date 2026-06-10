#!/usr/bin/env python3
"""Extract ORBIT-2 training FOMs from a perf-run job directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_training_log import aggregate_foms, parse_slurm_log  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ORBIT-2 perf-run FOM extractor.")
    parser.add_argument("--job-dir", type=Path, required=True, help="perf-runs/<jobid>/")
    parser.add_argument("--steady-epoch-start", type=int, default=2)
    parser.add_argument("--warmup-batches-per-epoch", type=int, default=1)
    args = parser.parse_args()

    job_dir = args.job_dir
    job_id = job_dir.name
    log_candidates = [
        job_dir / f"orbit2-train-{job_id}.out",
        Path.cwd() / f"orbit2-train-{job_id}.out",
    ]
    shared_root = os.environ.get("AI4S_SHARED_DIR")
    if shared_root:
        log_candidates.insert(
            1,
            Path(shared_root)
            / "models"
            / "ORBIT-2"
            / "outputs"
            / "train"
            / "logs"
            / f"orbit2-train-{job_id}.out",
        )
    log_path = next((p for p in log_candidates if p.is_file()), None)
    if log_path is None:
        print(f"error: SLURM log not found for job {job_id}", file=sys.stderr)
        return 2

    records = parse_slurm_log(log_path)
    foms = aggregate_foms(
        records,
        warmup_batches_per_epoch=args.warmup_batches_per_epoch,
        steady_epoch_start=args.steady_epoch_start,
    )
    foms["job_id"] = job_id
    foms["log_path"] = str(log_path)

    out = job_dir / "foms.json"
    out.write_text(json.dumps(foms, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(foms, indent=2))
    if foms.get("loss_sanity_pass") is False:
        print("warning: loss sanity check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
