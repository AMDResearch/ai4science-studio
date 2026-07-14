#!/usr/bin/env python3
"""Post-process HydraGNN OmniHub job results into processed-data/."""

import json
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <job_results_dir>", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1])
    processed = job_dir / "processed-data"
    processed.mkdir(parents=True, exist_ok=True)

    summary = {
        "app": "ai4science-hydragnn-train",
        "job_dir": str(job_dir),
    }

    status = job_dir / "job-status.yaml"
    if status.is_file():
        summary["job_status_file"] = str(status)

    logs = list(job_dir.glob("logs/srun-*.out"))
    summary["log_count"] = len(logs)

    out = processed / "app-parser.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
