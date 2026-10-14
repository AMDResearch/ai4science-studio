"""Repo and cluster path helpers (no model-specific knowledge)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root_from(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "models.yaml").is_file() and (p / "schemas").is_dir():
            return p
    raise FileNotFoundError("could not locate AI4Science Studio repo root")


def schemas_dir(root: Path | None = None) -> Path:
    return (root or repo_root_from()) / "schemas"


def repo_agent_catalog(root: Path | None = None) -> Path:
    return (root or repo_root_from()) / "agents"


def load_cluster_config(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root_from()
    candidates = [
        root / ".cluster-config.yaml",
        Path.home() / ".config" / "ai4science-studio" / "cluster.yaml",
    ]
    for path in candidates:
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return data
    return {}


def cluster_registry_dir(root: Path | None = None) -> Path | None:
    """Return configured cluster registry directory, or None if unset.

    Expected location is ``${AI4S_SHARED_DIR}/registry`` when configured via
    ``registry_dir`` in cluster config, or via ``AI4S_REGISTRY_DIR``.
    """
    env = os.environ.get("AI4S_REGISTRY_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    cfg = load_cluster_config(root)
    raw = cfg.get("registry_dir") or (cfg.get("paths") or {}).get("registry_dir")
    if isinstance(raw, str) and raw.strip():
        return Path(os.path.expandvars(raw)).expanduser()
    shared = os.environ.get("AI4S_SHARED_DIR", "").strip()
    if shared:
        return Path(shared) / "registry"
    return None
