"""Harvest AMD Omnistat GPU telemetry from a completed live training run.

After an 8-GPU HydraGNN training job finishes, its per-run VictoriaMetrics TSDB
(written by studio/telemetry/sbatch_train_telemetry_amd.sh) persists on disk. This
module spins up a short-lived READ-ONLY VictoriaMetrics against that DB, range-queries
the GPU metrics, downsamples them, and returns a compact JSON payload for the Analyze
"System Telemetry" panel.

Query patterns mirror material_science/models/HydraGNN/examples/run_fom_extractor.py
(the single source of truth for the gfx950 PromQL). Peaks are max-over-job; series are
downsampled to ~POINTS samples (max across the 8 GPUs per timestamp).
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import requests

POINTS = 93  # match the reference telemetry panel's time-series resolution

# ── PromQL ────────────────────────────────────────────────────────────────────
# The VictoriaMetrics DB is per-job (one omnistat-db per SLURM job), so we do NOT
# need the rmsjob_info join used in the multi-job perf-runs DB — every series in
# this DB already belongs to the one job. Metric names verified against a real
# gfx950 run (smoke test 17733). fp64 GFLOP/s follows the config-template formula;
# we keep the MFMA term (dominant for GEMM-heavy ML training).
#
# {jobid} kept in the format string for API symmetry but unused in the exprs.
_EXPR = {
    "gpu_util_pct": 'rocm_utilization_percentage',
    "power_w": 'rocm_average_socket_power_watts',
    "temp_c": 'rocm_temperature_celsius',
    "vram_gb": 'rocm_vram_used_percentage * rocm_vram_total_bytes / 100 / 1073741824',
    "fp64_tflops": (
        'rate(omnistat_hardware_counter{{name="SQ_INSTS_VALU_MFMA_MOPS_F64"}}[30s]) '
        '* 512 / 1e12'
    ),
    "hbm_read_gbs": 'rate(omnistat_hardware_counter{{name="FETCH_SIZE"}}[30s]) * 1024 / 1e9',
}

# Series metrics rendered as interactive charts (max across GPUs per timestamp).
_SERIES_KEYS = ("gpu_util_pct", "power_w", "temp_c")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start_read_vm(vm_bin: str, db_path: str, port: int, log_path: Path):
    """Start a read-only VictoriaMetrics against an existing on-disk TSDB.

    -fs.disableMmap avoids the stale-flock crash when re-opening a DB that another
    vmstorage held; retries because omnistat's --stopserver may not have released
    the lock the instant SLURM reports COMPLETED.
    """
    proc = subprocess.Popen(
        [
            vm_bin,
            f"-storageDataPath={db_path}",
            f"-httpListenAddr=127.0.0.1:{port}",
            "-retentionPeriod=100y",
            "-search.disableCache",
            "-search.latencyOffset=0",
            "-search.maxPointsPerTimeseries=90000",
            "-fs.disableMmap",
        ],
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(25):
        time.sleep(1)
        try:
            requests.get(f"http://127.0.0.1:{port}/api/v1/status/tsdb", timeout=2).raise_for_status()
            return proc
        except requests.RequestException:
            if proc.poll() is not None:  # VM died (likely flock) — let caller retry
                return proc
    return proc


def _tsdb_window(db_path: str, url: str) -> tuple[int, int]:
    """Return (start, end) unix seconds spanned by the TSDB, for range queries."""
    try:
        r = requests.get(f"{url}/api/v1/status/tsdb", timeout=10)
        r.raise_for_status()
        meta = r.json().get("data", {})
        mn, mx = meta.get("minTimestamp"), meta.get("maxTimestamp")
        if mn and mx:
            return int(mn) // 1000, int(mx) // 1000
    except (requests.RequestException, KeyError, ValueError):
        pass
    now = int(time.time())
    return now - 600, now


def _instant(url: str, promql: str, t: int) -> float | None:
    try:
        r = requests.get(f"{url}/api/v1/query", params={"query": promql, "time": t}, timeout=30)
        r.raise_for_status()
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else None
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None


def _range_max(url: str, base_expr: str, start: int, end: int, step: int) -> list:
    """query_range of max(<expr>) across GPUs; return list of (t_rel, value)."""
    promql = f"max({base_expr})"
    try:
        r = requests.get(
            f"{url}/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": f"{step}s"},
            timeout=60,
        )
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if not res:
            return []
        vals = res[0]["values"]  # [[ts, "v"], ...]
        return [(int(ts) - start, round(float(v), 3)) for ts, v in vals]
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return []


def _range_sum(url: str, base_expr: str, start: int, end: int, step: int) -> list:
    """query_range of sum(<expr>) across GPUs; return list of (t_rel, value)."""
    promql = f"sum({base_expr})"
    try:
        r = requests.get(
            f"{url}/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": f"{step}s"},
            timeout=60,
        )
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if not res:
            return []
        return [(int(ts) - start, round(float(v), 3)) for ts, v in res[0]["values"]]
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return []


def _range_avg(url: str, base_expr: str, start: int, end: int, step: int) -> list:
    """query_range of avg(<expr>) across GPUs; return list of (t_rel, value)."""
    promql = f"avg({base_expr})"
    try:
        r = requests.get(
            f"{url}/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": f"{step}s"},
            timeout=60,
        )
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if not res:
            return []
        return [(int(ts) - start, round(float(v), 3)) for ts, v in res[0]["values"]]
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return []


def _peak(url: str, key: str, jobid: str, start: int, end: int) -> float | None:
    """Peak (max across GPUs and time) of a metric over the job window.

    Uses query_range + Python max rather than a max_over_time subquery: the latter
    silently returns empty for rate()-based counter expressions in this VM build.
    """
    expr = _EXPR[key].format(jobid=jobid)
    step = max(1, (end - start) // POINTS)
    pts = _range_max(url, expr, start, end, step)
    vals = [v for _, v in pts if v is not None]
    return round(max(vals), 3) if vals else None


def _mean(url: str, key: str, jobid: str, start: int, end: int) -> float | None:
    """Mean over time of the per-timestamp AVERAGE across GPUs (the 'typical' value).

    Peak alone is misleading (a single GPU spiking to 100% shows as 100%); the mean
    of the cross-GPU average is the honest headline companion, matching the reference
    panel's 'peak 58% / mean 2.457%' framing.
    """
    expr = _EXPR[key].format(jobid=jobid)
    step = max(1, (end - start) // POINTS)
    pts = _range_avg(url, expr, start, end, step)
    vals = [v for _, v in pts if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def harvest(manifest_path: Path) -> dict | None:
    """Read a run manifest, query its omnistat DB, return the telemetry payload.

    Returns None if telemetry was disabled or the DB is unreadable. Always tears
    down the read-only VictoriaMetrics it starts.
    """
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError):
        return None
    if not manifest.get("telemetry_enabled"):
        return None

    db_path = manifest.get("omnistat_db_path", "")
    vm_bin = manifest.get("victoria_binary", "")
    jobid = str(manifest.get("jobid", ""))
    runtime_s = int(manifest.get("runtime_s") or 0)
    if not (db_path and vm_bin and Path(db_path).exists() and Path(vm_bin).exists()):
        return None

    log_path = Path(db_path).parent / "vm_read_query.log"
    proc = None
    # Retry the VM start a few times: omnistat's --stopserver may still hold the flock.
    for attempt in range(3):
        port = _free_port()
        proc = _start_read_vm(vm_bin, db_path, port, log_path)
        url = f"http://127.0.0.1:{port}"
        try:
            requests.get(f"{url}/api/v1/status/tsdb", timeout=2).raise_for_status()
            break
        except requests.RequestException:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            proc = None
            time.sleep(3)
    if proc is None:
        return None

    try:
        start, end = _tsdb_window(db_path, url)
        if runtime_s <= 0:
            runtime_s = max(end - start, 1)
        peaks = {k: _peak(url, k, jobid, start, end) for k in _EXPR}
        means = {k: _mean(url, k, jobid, start, end) for k in _EXPR}
        # Energy (kJ) = mean total socket power (summed across GPUs) * runtime / 1000.
        step = max(1, (end - start) // POINTS)
        _pwr_sum = _range_sum(url, _EXPR["power_w"], start, end, step)
        _pvals = [v for _, v in _pwr_sum if v is not None]
        mean_power = (sum(_pvals) / len(_pvals)) if _pvals else None
        peaks["energy_kj"] = round(mean_power * runtime_s / 1000.0, 2) if mean_power else None
        means["energy_kj"] = peaks["energy_kj"]  # energy is cumulative; no separate mean
        series: dict[str, list] = {"t_s": []}
        first = True
        for key in _SERIES_KEYS:
            pts = _range_max(url, _EXPR[key].format(jobid=jobid), start, end, step)
            if first and pts:
                series["t_s"] = [t for t, _ in pts]
                first = False
            series[key] = [v for _, v in pts]

        return {
            "n_gpus": int(manifest.get("n_gpus", 8)),
            "epochs": int(manifest.get("epochs", 0)),
            "runtime_s": runtime_s,
            "peaks": peaks,
            "means": means,
            "units": {
                "gpu_util_pct": "%", "power_w": "W", "temp_c": "C", "vram_gb": "GB",
                "energy_kj": "kJ", "fp64_tflops": "TFLOP/s", "hbm_read_gbs": "GB/s",
            },
            "series": series,
        }
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
