"""Repo-root and well-known path resolution."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the AI4Science Studio repo root.

    The app lives at <repo>/studio/, so the root is two parents up from this
    file. Allow an override for tests / unusual layouts.
    """
    override = os.environ.get("AI4S_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def models_index() -> Path:
    return repo_root() / "models.yaml"


def template_dir() -> Path:
    return repo_root() / "_template"


def jobs_state_file() -> Path:
    """Per-user job-state file, kept out of git."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "default"
    return repo_root() / "studio" / f".studio-jobs.{user}.json"
