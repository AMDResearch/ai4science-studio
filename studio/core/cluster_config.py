"""Read / merge / write the site-specific cluster config.

Two locations, checked in order (matching the /init-cluster agent command):
  1. <repo>/.cluster-config.yaml         (gitignored, per-checkout)
  2. ~/.config/ai4science-studio/cluster.yaml   (user-level, shared)

Writes are read-merge-write: unknown / extra top-level blocks (omnihub, assets,
perf_tools, discovered_sifs) are preserved rather than clobbered.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import repo_root

REPO_CONFIG = "repo"
USER_CONFIG = "user"


def repo_config_path() -> Path:
    return repo_root() / ".cluster-config.yaml"


def user_config_path() -> Path:
    return Path.home() / ".config" / "ai4science-studio" / "cluster.yaml"


def active_config_path() -> Path | None:
    """First existing config path, or None."""
    for p in (repo_config_path(), user_config_path()):
        if p.exists():
            return p
    return None


def load_config() -> dict[str, Any]:
    """Load the active config, or {} if none exists."""
    p = active_config_path()
    if not p:
        return {}
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge `update` into `base`, returning a new dict."""
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_config(update: dict[str, Any], location: str = REPO_CONFIG) -> Path:
    """Merge `update` into the existing config and write it back.

    location: REPO_CONFIG (repo root) or USER_CONFIG (~/.config).
    Existing extra blocks (omnihub/assets/perf_tools) are preserved.
    """
    target = repo_config_path() if location == REPO_CONFIG else user_config_path()
    existing: dict[str, Any] = {}
    if target.exists():
        with open(target) as fh:
            existing = yaml.safe_load(fh) or {}
    merged = _deep_merge(existing, update)

    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# AI4Science Studio - Local Cluster Configuration\n"
        "# Managed by the Studio web GUI (Cluster setup page) and /init-cluster.\n"
        "# This file is gitignored and never committed.\n\n"
    )
    with open(target, "w") as fh:
        fh.write(header)
        yaml.safe_dump(merged, fh, sort_keys=False, default_flow_style=False)
    return target


# --- convenience accessors used by the Run engine ---------------------------

def scratch_dir(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    return (cfg.get("paths") or {}).get("scratch") or ""


def slurm_defaults(cfg: dict | None = None) -> dict[str, str]:
    cfg = cfg if cfg is not None else load_config()
    s = cfg.get("slurm") or {}
    return {
        "partition": s.get("partition") or "",
        "account": s.get("account") or "",
        "qos": s.get("qos") or "",
    }


def runtime(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    return (cfg.get("containers") or {}).get("runtime") or ""


def assets(cfg: dict | None = None) -> dict[str, Any]:
    """Cached per-model discovered paths (SIF/overlay/clone/weights), if present."""
    cfg = cfg if cfg is not None else load_config()
    return cfg.get("assets") or {}
