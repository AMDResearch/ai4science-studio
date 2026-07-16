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


def poll_states() -> dict[str, str]:
    """Return {jobid: state}; jobs absent from squeue are marked COMPLETED/GONE.

    Also persists last_state back into the store.
    """
    records = _load()
    active = [r["jobid"] for r in records
              if r.get("jobid") and r.get("last_state") not in ("COMPLETED", "GONE", "CANCELLED", "FAILED")]
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
            # was active, now gone from squeue -> terminal
            state = "GONE"
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
