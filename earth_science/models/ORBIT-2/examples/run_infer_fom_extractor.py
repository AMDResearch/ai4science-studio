#!/usr/bin/env python3
"""Extract ORBIT-2 inference FOMs from an infer-perf-runs job directory.

Does not reuse training loss / steady_batch_time parsers. Records wall latency
and repeats honestly; throughput is left unset until sample/rank semantics are
validated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Reuse PromQL enrichment helpers from the training extractor when available.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from run_fom_extractor import _enrich_omnistat, _read_manifest  # type: ignore
except Exception:  # noqa: BLE001

    def _read_manifest(job_dir: Path) -> dict:
        p = job_dir / "manifest.json"
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _enrich_omnistat(foms: dict, manifest: dict, job_dir: Path) -> None:
        foms["mfma_bf16_tflops_per_card_avg"] = None
        foms["hbm_read_GBps_per_card_avg"] = None
        foms["xgmi_GBps_avg"] = None
        foms["omnistat_enrichment_note"] = "PromQL enrichment unavailable"


def _parse_markers(job_dir: Path) -> tuple[float | None, float | None]:
    path = job_dir / "timing_markers.log"
    if not path.is_file():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = re.findall(r"inference_start\s+\S+\s+([0-9.]+)", text)
    ends = re.findall(r"inference_end\s+\S+\s+([0-9.]+)", text)
    try:
        t0 = float(starts[-1]) if starts else None
        t1 = float(ends[-1]) if ends else None
    except ValueError:
        return None, None
    return t0, t1


def main() -> int:
    parser = argparse.ArgumentParser(description="ORBIT-2 inference FOM extractor")
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    job_dir = args.job_dir
    manifest = _read_manifest(job_dir)
    job_id = str(manifest.get("job_id") or job_dir.name)

    t0, t1 = _parse_markers(job_dir)
    wall = None
    if t0 is not None and t1 is not None and t1 >= t0:
        wall = t1 - t0
    if wall is None:
        wall = manifest.get("inference_wall_seconds")

    warmup = int(manifest.get("warmup_repeats") or 0)
    timed = int(manifest.get("timed_repeats") or 1)
    latency = None
    if wall is not None and timed > 0:
        # Attribute full marked window / timed repeats (warmup included in wall if run sequentially).
        # Documented as job-level, not per-rank steady-state.
        latency = float(wall) / float(max(timed, 1))

    foms = {
        "job_id": job_id,
        "workload": "visualize",
        "inference_wall_seconds": wall,
        "warmup_repeats": warmup,
        "timed_repeats": timed,
        "latency_per_forward_s": latency,
        "throughput_samples_per_s": None,
        "throughput_note": (
            "Unset: distributed visualize sample/rank semantics are not yet validated "
            "for a honest throughput FOM."
        ),
        "manifest_runtime_seconds": manifest.get("runtime_seconds"),
        "use_synthetic": manifest.get("use_synthetic"),
        "slurm_ntasks": manifest.get("slurm_ntasks"),
    }
    _enrich_omnistat(foms, manifest, job_dir)

    out = job_dir / "foms.json"
    out.write_text(json.dumps(foms, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(foms, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
