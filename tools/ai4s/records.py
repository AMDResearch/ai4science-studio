"""Run and result record helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def new_run_record(
    *,
    run_id: str,
    name: str,
    kind: str,
    resolved_inputs: dict[str, Any],
    repo: Path,
    parent_run_id: str | None = None,
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "name": name,
        "kind": kind,
        "version": "0.1.0",
        "state": "running",
        "user": os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown",
        "slurm_job_ids": [job_id] if job_id else [],
        "start_time": utc_now(),
        "end_time": None,
        "git_commit": git_commit(repo),
        "resolved_inputs": resolved_inputs,
        "nodes": {},
        "llm": llm or {},
        "error": None,
    }


def finalize_run(
    run: dict[str, Any],
    *,
    state: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(run)
    out["state"] = state
    out["end_time"] = utc_now()
    out["error"] = error
    return out


def new_result(
    *,
    run_id: str,
    name: str,
    status: str,
    checks: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "name": name,
        "status": status,
        "checks": checks or [],
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "error": error,
    }


def artifact_entry(path: Path, *, required_for_next: bool = False) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "required_for_next": required_for_next,
    }


def validate_required_artifacts(
    result: dict[str, Any],
    required_keys: list[str],
) -> tuple[bool, list[str]]:
    """Return (ok, missing_or_invalid)."""
    artifacts = result.get("artifacts") or {}
    problems: list[str] = []
    for key in required_keys:
        entry = artifacts.get(key)
        if not entry:
            problems.append(f"missing artifact: {key}")
            continue
        path = Path(entry.get("path", ""))
        if not path.is_file() and not path.is_dir():
            problems.append(f"artifact path missing: {key} -> {path}")
    return (not problems, problems)
