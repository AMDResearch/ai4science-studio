"""SQLite provenance store for Studio runs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "studio.db"


def init():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        slug TEXT,
        domain TEXT,
        task TEXT,
        mode TEXT,
        prompt TEXT,
        params TEXT,
        state TEXT,
        slurm_job_id TEXT,
        created_at TEXT,
        updated_at TEXT
    )""")
    con.commit()
    con.close()


def upsert_run(run: dict):
    import json
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT OR REPLACE INTO runs
        (id, slug, domain, task, mode, prompt, params, state, slurm_job_id, created_at, updated_at)
        VALUES (:id, :slug, :domain, :task, :mode, :prompt, :params, :state, :slurm_job_id, :created_at, :updated_at)
    """, {
        "id": run["id"],
        "slug": run.get("slug", ""),
        "domain": run.get("domain", ""),
        "task": run.get("task", "inference"),
        "mode": run.get("mode", "demo"),
        "prompt": run.get("prompt", ""),
        "params": json.dumps(run.get("params", {})),
        "state": run.get("state", "pending"),
        "slurm_job_id": run.get("slurm_job_id"),
        "created_at": run.get("created_at", ""),
        "updated_at": run.get("updated_at", ""),
    })
    con.commit()
    con.close()


def get_run(run_id: str) -> dict | None:
    import json
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    con.close()
    if not row:
        return None
    keys = ["id", "slug", "domain", "task", "mode", "prompt", "params",
            "state", "slurm_job_id", "created_at", "updated_at"]
    d = dict(zip(keys, row))
    d["params"] = json.loads(d["params"] or "{}")
    return d


def list_runs(limit: int = 50) -> list[dict]:
    import json
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    keys = ["id", "slug", "domain", "task", "mode", "prompt", "params",
            "state", "slurm_job_id", "created_at", "updated_at"]
    result = []
    for row in rows:
        d = dict(zip(keys, row))
        d["params"] = json.loads(d["params"] or "{}")
        result.append(d)
    return result
