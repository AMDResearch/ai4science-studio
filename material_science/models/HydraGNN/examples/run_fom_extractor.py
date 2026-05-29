#!/usr/bin/env python3
"""run_fom_extractor.py — Compute FOMs + TraceLens↔Omnistat kernel_correlation.csv.

Implements material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/fom_extractor.md
for login-node post-processing of a completed perf run.

Usage:
    export AI4S_SHARED_DIR=/shared/aaji
    export REPO_ROOT=/home/aaji/git/ai4science-studio
    python run_fom_extractor.py --manifest /path/to/perf-runs/<jobid>/manifest.json \\
        [--vm-port 8432] [--tsdb-url http://127.0.0.1:8432]
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

MFMA_Q = (
    'rate(omnistat_hardware_counter{{name="SQ_INSTS_VALU_MFMA_MOPS_F64"}}[10s]) '
    '* on(instance) group_left() max by(instance)(rmsjob_info{{jobid="{jobid}"}}) * 4 / 1e12'
)
HBM_Q = (
    'rate(omnistat_hardware_counter{{name="FETCH_SIZE"}}[10s]) '
    '* on(instance) group_left() max by(instance)(rmsjob_info{{jobid="{jobid}"}}) * 1024 / 1e9'
)
PWR_Q = (
    'rocm_average_socket_power_watts '
    '* on(instance) group_left() max by(instance)(rmsjob_info{{jobid="{jobid}"}})'
)
JOB_MFMA_Q = (
    'avg(rate(omnistat_hardware_counter{{name="SQ_INSTS_VALU_MFMA_MOPS_F64"}}[10s]) '
    '* on(instance) group_left() max by(instance)(rmsjob_info{{jobid="{jobid}"}})) * 4 / 1e12'
)
JOB_PWR_Q = (
    'avg_over_time((rocm_average_socket_power_watts '
    '* on(instance) group_left() max by(instance)(rmsjob_info{{jobid="{jobid}"}}))'
    '[{runtime}s:])'
)


def _open_trace(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def trace_anchor_epoch_us(trace_path: str, trace: dict) -> int:
    """Return unix-epoch microseconds for kineto ts=0."""
    base_ns = trace.get("baseTimeNanoseconds")
    if base_ns and int(base_ns) > 1_700_000_000_000_000_000:
        return int(base_ns) // 1000
    m = re.search(r"\.(\d+)\.pt\.trace\.json", trace_path)
    if m:
        return int(m.group(1)) // 1000
    return 0


def load_kernels(trace_path: str) -> tuple[list[tuple[int, int, str]], int]:
    with _open_trace(trace_path) as f:
        data = json.load(f)
    anchor_us = trace_anchor_epoch_us(trace_path, data)
    kernels = [
        (int(e["ts"]), int(e["dur"]), e["name"])
        for e in data.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("cat") == "kernel" and "ts" in e and "dur" in e
    ]
    return kernels, anchor_us


def bucket_windows(
    kernels: list[tuple[int, int, str]], anchor_us: int
) -> dict[int, Counter]:
    windows: dict[int, Counter] = defaultdict(Counter)
    for ts, dur, name in kernels:
        win = (anchor_us + ts) // 1_000_000
        windows[win][name] += dur
    return windows


def window_top(counter: Counter) -> tuple[str, float, str, float]:
    if not counter:
        return "", 0.0, "", 0.0
    top2 = counter.most_common(2)
    top_name, top_dur = top2[0]
    top_frac = min(top_dur / 1_000_000.0, 1.0)
    if len(top2) > 1:
        second_name, second_dur = top2[1]
        second_frac = min(second_dur / 1_000_000.0, 1.0)
    else:
        second_name, second_frac = "", 0.0
    return top_name, top_frac, second_name, second_frac


def promql_vector(tsdb_url: str, promql: str, t: int) -> dict[str, float]:
    r = requests.get(
        f"{tsdb_url}/api/v1/query",
        params={"query": promql, "time": t},
        timeout=30,
    )
    r.raise_for_status()
    out: dict[str, float] = {}
    for row in r.json()["data"]["result"]:
        inst = row["metric"].get("instance", "")
        out[inst] = float(row["value"][1])
    return out


def promql_scalar(tsdb_url: str, promql: str, t: int) -> float | None:
    r = requests.get(
        f"{tsdb_url}/api/v1/query",
        params={"query": promql, "time": t},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()["data"]["result"]
    if not res:
        return None
    return float(res[0]["value"][1])


def start_vm(db_path: str, port: int, log_path: Path) -> None:
    vm = os.environ.get(
        "VICTORIA_METRICS_BIN",
        "/shared/omnihub/tools/victoriametrics/victoria-metrics-prod",
    )
    subprocess.Popen(
        [
            vm,
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
    for _ in range(20):
        time.sleep(1)
        try:
            requests.get(f"http://127.0.0.1:{port}/api/v1/status/tsdb", timeout=2).raise_for_status()
            return
        except requests.RequestException:
            continue
    raise RuntimeError(f"VictoriaMetrics did not become ready on port {port}")


def attribution_quality(windows: dict[int, Counter]) -> str:
    if not windows:
        return "poor"
    busy = 0
    for win in windows:
        _, frac, _, _ = window_top(windows[win])
        if frac > 0.5:
            busy += 1
    ratio = busy / len(windows)
    if ratio > 0.5:
        return "good"
    if ratio >= 0.25:
        return "fair"
    return "poor"


def short_kernel_family(name: str) -> str:
    if "nccl" in name.lower():
        return "nccl"
    if "aten::bmm" in name or "bmm" in name.lower()[:20]:
        return "aten::bmm"
    if "aten::mm" in name or name.startswith("Cijk_"):
        return "aten::mm"
    if "triton" in name.lower():
        return "triton"
    return name[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="perf-run manifest.json")
    parser.add_argument("--tsdb-url", default=None, help="VictoriaMetrics base URL")
    parser.add_argument("--vm-port", type=int, default=8432, help="Port if starting VM")
    parser.add_argument("--no-start-vm", action="store_true", help="Require --tsdb-url")
    args = parser.parse_args()

    shared = os.environ.get("AI4S_SHARED_DIR", "/shared/aaji")
    repo = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[3]))

    with open(args.manifest) as f:
        manifest = json.load(f)

    perf_run = Path(manifest["perf_run_dir"])
    jobid = str(manifest["jobid"])
    runtime_s = float(manifest.get("runtime_seconds", manifest.get("runtime_s", 300)))
    trace_paths = manifest.get("trace_paths") or []
    if not trace_paths and manifest.get("trace_json"):
        trace_paths = [manifest["trace_json"]]
    trace_path = trace_paths[0] if trace_paths else ""

    log_path = manifest.get("training_log_path") or str(perf_run / f"hydragnn-train-{jobid}.out")
    if not os.path.isfile(log_path):
        alt = perf_run / "logs" / f"hydragnn-train-{jobid}-N2" / "run.log"
        if alt.is_file():
            log_path = str(alt)

    tsdb_url = args.tsdb_url
    if not tsdb_url:
        if args.no_start_vm:
            print("ERROR: --tsdb-url or VM start required", file=sys.stderr)
            return 1
        port = args.vm_port
        db = manifest.get("omnistat_db_path", str(perf_run / "omnistat-db"))
        start_vm(db, port, perf_run / "vm_for_foms.log")
        tsdb_url = f"http://127.0.0.1:{port}"

    # Time-based FOMs from log when available
    conv_foms: dict[str, Any] = {}
    parse_script = Path(repo) / "material_science/models/HydraGNN/examples/parse_convergence.py"
    conv_json = perf_run / "_convergence_foms.json"
    if os.path.isfile(log_path) and parse_script.is_file():
        subprocess.run(
            [
                sys.executable,
                str(parse_script),
                "--log",
                log_path,
                "--output",
                str(perf_run / "convergence.csv"),
                "--json-foms",
                str(conv_json),
            ],
            check=False,
        )
        if conv_json.is_file():
            with open(conv_json) as f:
                conv_foms = json.load(f)

    epoch_time_s = conv_foms.get("mean_epoch_time_excluding_epoch_0")
    ranks = int(manifest.get("ranks", 16))
    batch = int(manifest.get("hg_batch_size", 400))
    max_batch = int(manifest.get("hydragnn_max_num_batch", 50))
    num_epochs = int(conv_foms.get("num_epochs_completed") or manifest.get("hg_num_epoch", 3))
    samples = num_epochs * max_batch * batch * ranks
    throughput = (samples / runtime_s) if runtime_s else None

    # Query counters near job mid-window (not profile-trace mid — trace is only epoch-2).
    t_query = int(time.time())
    db_meta = Path(manifest.get("omnistat_db_path", perf_run / "omnistat-db")) / "data" / "small"
    for meta_file in db_meta.rglob("metadata.json"):
        try:
            meta = json.loads(meta_file.read_text())
            t_query = int((meta["MinTimestamp"] + meta["MaxTimestamp"]) / 2000)
            break
        except (KeyError, json.JSONDecodeError, OSError):
            continue

    mfma_tflops = promql_scalar(tsdb_url, JOB_MFMA_Q.format(jobid=jobid), t_query)
    mean_power_W = promql_scalar(tsdb_url, JOB_PWR_Q.format(jobid=jobid, runtime=int(runtime_s)), t_query)
    energy_J = (mean_power_W * runtime_s) if mean_power_W is not None else None
    energy_per_sample = (energy_J / samples) if energy_J and samples else None

    windows: dict[int, Counter] = {}
    correlation_rows: list[dict[str, Any]] = []
    if trace_path and os.path.isfile(trace_path):
        size_gb = os.path.getsize(trace_path) / (1024**3)
        if size_gb > 4:
            attr_q = "skipped:trace_too_large"
        else:
            kernels, anchor_us = load_kernels(trace_path)
            windows = bucket_windows(kernels, anchor_us)
            attr_q = attribution_quality(windows)
            for win_ts in sorted(windows):
                top_name, top_frac, second_name, second_frac = window_top(windows[win_ts])
                mfma_by_inst = promql_vector(tsdb_url, MFMA_Q.format(jobid=jobid), win_ts)
                hbm_by_inst = promql_vector(tsdb_url, HBM_Q.format(jobid=jobid), win_ts)
                pwr_by_inst = promql_vector(tsdb_url, PWR_Q.format(jobid=jobid), win_ts)
                instances = sorted(set(mfma_by_inst) | set(hbm_by_inst) | set(pwr_by_inst))
                if not instances:
                    iso = datetime.fromtimestamp(win_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                    correlation_rows.append(
                        {
                            "window_start_iso": iso,
                            "window_start_ts": win_ts,
                            "instance": "lux-mi355x-b1",
                            "gpu_id": 0,
                            "top_kernel_name": top_name,
                            "top_kernel_busy_frac": f"{top_frac:.4f}",
                            "second_kernel_name": second_name,
                            "second_kernel_busy_frac": f"{second_frac:.4f}",
                            "mfma_tflops": "",
                            "hbm_read_GBps": "",
                            "mean_power_W": "",
                        }
                    )
                for inst in instances:
                    iso = datetime.fromtimestamp(win_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                    correlation_rows.append(
                        {
                            "window_start_iso": iso,
                            "window_start_ts": win_ts,
                            "instance": inst,
                            "gpu_id": 0,
                            "top_kernel_name": top_name,
                            "top_kernel_busy_frac": f"{top_frac:.4f}",
                            "second_kernel_name": second_name,
                            "second_kernel_busy_frac": f"{second_frac:.4f}",
                            "mfma_tflops": mfma_by_inst.get(inst, ""),
                            "hbm_read_GBps": hbm_by_inst.get(inst, ""),
                            "mean_power_W": pwr_by_inst.get(inst, ""),
                        }
                    )
    else:
        attr_q = "poor"

    top_dominants = Counter(short_kernel_family(window_top(windows[w])[0]) for w in windows).most_common(5)
    busy_gt_half = sum(1 for w in windows if window_top(windows[w])[1] > 0.5)

    mm_mfma: list[float] = []
    bmm_mfma: list[float] = []
    for row in correlation_rows:
        name = row["top_kernel_name"]
        mfma = row.get("mfma_tflops")
        if mfma == "" or mfma is None:
            continue
        val = float(mfma)
        fam = short_kernel_family(name)
        if fam == "aten::mm":
            mm_mfma.append(val)
        elif fam == "aten::bmm" or "bmm" in name.lower():
            bmm_mfma.append(val)

    existing: dict[str, Any] = {}
    foms_path = perf_run / "foms.json"
    if foms_path.is_file():
        with open(foms_path) as f:
            existing = json.load(f)

    prev_foms = existing.get("foms") or {}

    def _keep_counter(new_val: float | None, old_val: Any, min_sane: float) -> Any:
        if new_val is None:
            return old_val
        if old_val is not None and float(new_val) < min_sane and float(old_val) >= min_sane:
            return old_val
        return new_val

    foms_doc = {
        **existing,
        "jobid": jobid,
        "runtime_s": runtime_s,
        "num_epochs_completed": num_epochs,
        "samples_processed": samples,
        "foms": {
            **prev_foms,
            "epoch_time_s": epoch_time_s or prev_foms.get("epoch_time_s"),
            "throughput_samples_per_s": throughput or prev_foms.get("throughput_samples_per_s"),
            "mfma_tflops": _keep_counter(mfma_tflops, prev_foms.get("mfma_tflops"), 0.5),
            "energy_J": _keep_counter(energy_J, prev_foms.get("energy_J"), 1e5),
            "mean_power_W": _keep_counter(mean_power_W, prev_foms.get("mean_power_W"), 500.0),
            "energy_per_sample_J": energy_per_sample or prev_foms.get("energy_per_sample_J"),
            "final_loss": conv_foms.get("final_train_loss") or prev_foms.get("final_loss"),
        },
        "kernel_correlation_summary": {
            "windows_examined": len(windows),
            "windows_with_top_kernel_busy_frac_gt_0p5": busy_gt_half,
            "top_kernel_dominants": top_dominants,
            "median_mfma_tflops_when_aten_mm_dominant": statistics.median(mm_mfma) if mm_mfma else None,
            "median_mfma_tflops_when_aten_bmm_dominant": statistics.median(bmm_mfma) if bmm_mfma else None,
            "median_hbm_read_GBps_when_mul_dominant": None,
            "attribution_quality": attr_q,
            "trace_path": trace_path,
            "omnistat_join": "ok" if correlation_rows and correlation_rows[0].get("mfma_tflops") != "" else "trace_only_or_partial",
            "tracelens_report": str(perf_run / "tracelens" / "report.xlsx"),
        },
    }

    csv_path = perf_run / "kernel_correlation.csv"
    fields = [
        "window_start_iso",
        "window_start_ts",
        "instance",
        "gpu_id",
        "top_kernel_name",
        "top_kernel_busy_frac",
        "second_kernel_name",
        "second_kernel_busy_frac",
        "mfma_tflops",
        "hbm_read_GBps",
        "mean_power_W",
    ]
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(correlation_rows)
    tmp_csv.rename(csv_path)

    tmp_foms = foms_path.with_suffix(".json.tmp")
    with open(tmp_foms, "w") as f:
        json.dump(foms_doc, f, indent=2)
    tmp_foms.rename(foms_path)

    print(
        f"STATUS=ok; reason=foms epoch_time={foms_doc['foms'].get('epoch_time_s')}s "
        f"correlation={attr_q} windows={len(windows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
