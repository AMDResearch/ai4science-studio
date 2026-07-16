"""Local job-state store + SLURM status polling + log tail.

Job records are kept in a per-user JSON file (gitignored). squeue is polled in
one batched call; when a jobid drops out of squeue it's marked terminal.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .paths import jobs_state_file


def _load() -> list[dict[str, Any]]:
    p = jobs_state_file()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return []


def _save(records: list[dict[str, Any]]) -> None:
    p = jobs_state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2))


def add_job(record: dict[str, Any]) -> None:
    records = _load()
    records.insert(0, record)
    _save(records)


def list_jobs() -> list[dict[str, Any]]:
    return _load()


def remove_job(jobid: str) -> None:
    _save([r for r in _load() if r.get("jobid") != jobid])


def _squeue_states(jobids: list[str]) -> dict[str, str]:
    if not jobids or not shutil.which("squeue"):
        return {}
    try:
        proc = subprocess.run(
            ["squeue", "-j", ",".join(jobids), "-h", "-o", "%i|%T"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:  # noqa: BLE001
        return {}
    states: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "|" in line:
            jid, state = line.split("|", 1)
            states[jid.strip()] = state.strip()
    return states


_TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
             "NODE_FAIL")


def _sacct_final(jobid: str) -> tuple[str, str]:
    """Return (state, exit_code) for a finished job from sacct, or ('', '')."""
    if not shutil.which("sacct"):
        return ("", "")
    try:
        proc = subprocess.run(
            ["sacct", "-j", jobid, "-n", "-P", "-X",
             "--format=State,ExitCode"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:  # noqa: BLE001
        return ("", "")
    for line in proc.stdout.splitlines():
        if "|" in line:
            state, code = (line.split("|", 1) + [""])[:2]
            # State can be "CANCELLED by 1234" -> take first word
            return (state.strip().split()[0] if state.strip() else "", code.strip())
    return ("", "")


def poll_states() -> dict[str, str]:
    """Return {jobid: state}; jobs absent from squeue are resolved via sacct.

    When a job leaves the queue we ask sacct for its real terminal state
    (COMPLETED / FAILED / ...) and exit code instead of a meaningless "GONE".
    Persists last_state and exit_code back into the store.
    """
    records = _load()
    active = [r["jobid"] for r in records
              if r.get("jobid") and r.get("last_state") not in _TERMINAL]
    live = _squeue_states(active)
    result: dict[str, str] = {}
    changed = False
    for r in records:
        jid = r.get("jobid")
        if not jid:
            continue
        if jid in live:
            state = live[jid]
        elif jid in active:
            # left the queue -> ask sacct for the real outcome
            final, code = _sacct_final(jid)
            state = final or "COMPLETED"
            if code and r.get("exit_code") != code:
                r["exit_code"] = code
                changed = True
        else:
            state = r.get("last_state", "UNKNOWN")
        result[jid] = state
        if r.get("last_state") != state:
            r["last_state"] = state
            changed = True
    if changed:
        _save(records)
    return result


def cancel(jobid: str) -> bool:
    if not shutil.which("scancel"):
        return False
    try:
        subprocess.run(["scancel", jobid], timeout=15)
        return True
    except Exception:  # noqa: BLE001
        return False


def tail_log(outfile: str, max_bytes: int = 60_000) -> str:
    p = Path(outfile)
    if not p.exists():
        return ""
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            if size > max_bytes:
                fh.seek(-max_bytes, os.SEEK_END)
            data = fh.read()
        return data.decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"<could not read log: {exc}>"


# Headline metrics to surface from a completed run's log. Each entry:
# (label, compiled regex with one capture group). Order = display order.
import re as _re

_METRIC_PATTERNS = [
    ("PSNR", _re.compile(r"PSNR[ :=]+([0-9.]+)")),
    ("SSIM", _re.compile(r"SSIM[ :=]+([0-9.]+)")),
    ("Final loss", _re.compile(r"(?:val[_ ]?loss|final loss|test loss)[ :=]+([0-9.]+)", _re.I)),
    ("Throughput (samp/s)", _re.compile(r"([0-9.]+)\s*samples?/s", _re.I)),
]


def extract_metrics(outfile: str) -> dict[str, str]:
    """Pull headline metrics (PSNR/SSIM/loss/throughput) from a run's log.

    Returns the LAST match for each pattern (final epoch / final eval), so a
    completed job shows a compact result summary instead of a raw log wall.
    """
    log = tail_log(outfile, max_bytes=200_000)
    if not log:
        return {}
    found: dict[str, str] = {}
    for label, rx in _METRIC_PATTERNS:
        matches = rx.findall(log)
        if matches:
            found[label] = matches[-1]
    return found
