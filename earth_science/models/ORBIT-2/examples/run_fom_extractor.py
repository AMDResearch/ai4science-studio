#!/usr/bin/env python3
"""Extract ORBIT-2 training FOMs from a perf-run job directory."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_training_log import aggregate_foms, parse_slurm_log  # noqa: E402


def _infer_ai4s_shared_from_job_dir(job_dir: Path) -> Path | None:
    try:
        parts = job_dir.resolve().parts
        if "models" in parts and "ORBIT-2" in parts:
            i = parts.index("models")
            if i >= 2:
                return Path(parts[0]).joinpath(*parts[1:i])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_manifest(job_dir: Path) -> dict:
    p = job_dir / "manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _promql_scalar(tsdb_url: str, promql: str, t_unix: int) -> float | None:
    """Instant query at time=t_unix (VictoriaMetrics / Prometheus API)."""
    q = urllib.parse.urlencode({"query": promql, "time": str(t_unix)})
    url = f"{tsdb_url.rstrip('/')}/api/v1/query?{q}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    res = body.get("data", {}).get("result") or []
    if not res:
        return None
    try:
        return float(res[0]["value"][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _enrich_omnistat(foms: dict, manifest: dict, job_dir: Path) -> None:
    """Optional PromQL enrichment when ORBIT2_TSDB_URL is set (login-node VM)."""
    tsdb = os.environ.get("ORBIT2_TSDB_URL", "").strip()
    job_id = manifest.get("job_id") or job_dir.name
    runtime = manifest.get("runtime_seconds")
    if not tsdb or not job_id or not runtime or runtime <= 0:
        foms["mfma_bf16_tflops_per_card_avg"] = None
        foms["hbm_read_GBps_per_card_avg"] = None
        foms["xgmi_GBps_avg"] = None
        foms["omnistat_enrichment_note"] = (
            "Set ORBIT2_TSDB_URL to a running VictoriaMetrics (e.g. from omnistat_analyst) "
            "and ensure runtime_seconds + job_id are in manifest.json."
        )
        return

    t_end = int(Path(job_dir / "manifest.json").stat().st_mtime) if job_dir.exists() else 0
    if t_end <= 0:
        t_end = int(__import__("time").time())
    mfma_q = (
        f"avg(rate(SQ_INSTS_VALU_MFMA_MOPS_BF16[30s]) * on(instance) group_left() "
        f'max by(instance)(rmsjob_info{{jobid="{job_id}"}})) * 512 / 1e12'
    )
    hbm_q = (
        f"avg(rate(FETCH_SIZE[30s]) * on(instance) group_left() "
        f'max by(instance)(rmsjob_info{{jobid="{job_id}"}})) * 1024 / 1e9'
    )
    # XGMI naming varies by omnistat version; try a common byte counter.
    xgmi_q = (
        f"avg(rate(rocm_xgmi_link0_tx_bytes[30s]) * on(instance) group_left() "
        f'max by(instance)(rmsjob_info{{jobid="{job_id}"}})) / 1e9'
    )

    t_mid = max(1, t_end - runtime // 2)
    foms["mfma_bf16_tflops_per_card_avg"] = _promql_scalar(tsdb, mfma_q, t_mid)
    foms["hbm_read_GBps_per_card_avg"] = _promql_scalar(tsdb, hbm_q, t_mid)
    foms["xgmi_GBps_avg"] = _promql_scalar(tsdb, xgmi_q, t_mid)
    foms["omnistat_enrichment_note"] = (
        f"PromQL at time={t_mid} via ORBIT2_TSDB_URL; metric names may differ by omnistat build — nulls mean rename/query fix."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ORBIT-2 perf-run FOM extractor.")
    parser.add_argument("--job-dir", type=Path, required=True, help="perf-runs/<jobid>/")
    parser.add_argument("--steady-epoch-start", type=int, default=2)
    parser.add_argument("--warmup-batches-per-epoch", type=int, default=1)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Override manifest global_batch_size (trainer batch × fsdp × simple_ddp)",
    )
    args = parser.parse_args()

    job_dir = args.job_dir
    job_id = job_dir.name
    manifest = _read_manifest(job_dir)
    if manifest.get("job_id"):
        job_id = str(manifest["job_id"])

    log_candidates = [
        job_dir / f"orbit2-train-{job_id}.out",
        Path.cwd() / f"orbit2-train-{job_id}.out",
    ]
    shared_root = os.environ.get("AI4S_SHARED_DIR")
    if not shared_root:
        inferred = _infer_ai4s_shared_from_job_dir(job_dir)
        if inferred is not None:
            shared_root = str(inferred)
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
    gbatch = args.global_batch_size
    if gbatch is None and manifest.get("global_batch_size") is not None:
        gbatch = int(manifest["global_batch_size"])

    # Per-rank trainer batch enables honest real-per-step throughput (vs nominal global).
    nominal_per_rank = None
    if manifest.get("orbit2_batch_size") is not None:
        nominal_per_rank = int(manifest["orbit2_batch_size"])

    foms = aggregate_foms(
        records,
        warmup_batches_per_epoch=args.warmup_batches_per_epoch,
        steady_epoch_start=args.steady_epoch_start,
        global_batch_size=gbatch,
        nominal_per_rank_batch=nominal_per_rank,
    )
    foms["job_id"] = job_id
    foms["log_path"] = str(log_path)

    # HBM reserved (peak) + effective-batch integrity. throughput now prefers the REAL
    # per-step batch dims (throughput_method=real_per_step_batch); partial_step_fraction and
    # steady_realized_batch_dims expose dataloader batch fragmentation so the orchestrator can
    # still reject artifact comparisons (loop overnight-3x-001 iter-1 + clean nw test 10504/10505).
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""
    hbm_vals = [float(x) for x in re.findall(r"memory_reserved:\s*([0-9.]+)\s*GB", log_text)]
    foms["hbm_reserved_GB"] = max(hbm_vals) if hbm_vals else None
    foms["hbm_reserved_pct_288"] = round(max(hbm_vals) / 288 * 100, 1) if hbm_vals else None
    if manifest:
        foms["manifest_parallelism"] = manifest.get("parallelism")
        foms["manifest_runtime_seconds"] = manifest.get("runtime_seconds")
        foms["manifest_data_type"] = manifest.get("data_type")

    _enrich_omnistat(foms, manifest, job_dir)

    # Energy placeholders until wired to rocm_avg_pwr like HydraGNN fom_extractor
    foms["energy_J"] = None
    foms["mean_power_W"] = None
    foms["energy_per_sample_J"] = None
    if foms.get("throughput_samples_per_s") and manifest.get("runtime_seconds"):
        tp = float(foms["throughput_samples_per_s"])
        rt = float(manifest["runtime_seconds"])
        if tp > 0 and rt > 0 and not math.isnan(tp):
            foms["samples_per_job_estimate"] = tp * rt

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
