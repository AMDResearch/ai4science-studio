"""Suggest concrete values for path-ish env vars so the Run form is never a
blank required box.

Sources, in priority order:
  1. The cluster config `assets:` cache (paths discovered by init-cluster).
  2. Path conventions derived from the shared root:
       <root>/models/<Model>/{code/<Model>,overlays,weights}
       <root>/images/*.sif
Existing paths are offered before non-existing ones; everything is de-duped.
"""
from __future__ import annotations

import re
from pathlib import Path

from .catalog import Model
from . import cluster_config as cc


def shared_root(cfg: dict) -> str:
    """The AI4S_SHARED_DIR value: scratch, else projects, else parent of assets."""
    paths = cfg.get("paths") or {}
    for key in ("scratch", "projects"):
        v = paths.get(key)
        if v:
            return v
    # infer from any absolute asset path: take the two-level parent of a
    # .../models/<Model>/... entry.
    for val in (cc.assets(cfg) or {}).values():
        if isinstance(val, str) and "/models/" in val:
            return val.split("/models/")[0]
    return ""


def _model_token(model: Model) -> str:
    return re.sub(r"[^a-z0-9]", "", model.slug.lower())


def _kind(var_name: str) -> str:
    u = var_name.upper()
    if "SIF" in u:
        return "sif"
    if "OVERLAY" in u:
        return "overlay"
    # config before code: ORBIT2_CONFIG_TEMPLATE / HG_CONFIG are config files.
    if "CONFIG" in u or u.endswith("_YAML") or "_YAML" in u:
        return "config"
    if any(k in u for k in ("ROOT", "REPO", "CLONE", "_CODE")):
        return "code"
    if any(k in u for k in ("CHECKPOINT", "CKPT", "WEIGHT")):
        return "weights"
    if "DATA" in u:
        return "data"
    return "other"


def candidates(model: Model, var_name: str, cfg: dict) -> list[str]:
    """Ordered, de-duped candidate values for a path-ish env var.

    Existing paths first. May be empty for vars we can't infer (caller then
    falls back to a plain text box).
    """
    kind = _kind(var_name)
    token = _model_token(model)
    root = shared_root(cfg)
    base = f"{root}/models/{model.slug}" if root else ""
    assets = cc.assets(cfg) or {}

    out: list[str] = []

    # 1) assets cache — match by kind, preferring the model's own entries.
    def _asset_match(want_token: bool) -> None:
        for key, val in assets.items():
            if not isinstance(val, str) or not val:
                continue
            k = key.lower()
            tok_ok = (token in k) if want_token else (token not in k)
            if not tok_ok:
                continue
            if kind == "sif" and "sif" in k:
                out.append(val)
            elif kind == "overlay" and "overlay" in k:
                out.append(val)
            elif kind == "code" and any(x in k for x in ("clone", "code", "repo", "root")):
                out.append(val)
            elif kind == "weights" and any(x in k for x in ("weight", "ckpt", "checkpoint")):
                out.append(val)
            elif kind == "data" and any(x in k for x in ("data", "weight")):
                out.append(val)

    _asset_match(want_token=True)
    if kind == "sif":  # generic rocm image applies to any model
        _asset_match(want_token=False)

    # config files live in the model's examples/ dir (tracked in the repo).
    if kind == "config":
        ex = model.examples_dir
        if ex.is_dir():
            for y in sorted(ex.glob("*.yaml")) + sorted(ex.glob("*.yml")):
                # offer the bare basename (scripts resolve relative to examples/)
                out.append(y.name)

    # 2) conventions from the shared root.
    if base:
        if kind == "code":
            out.append(f"{base}/code/{model.slug}")
            if model.slug == "ORBIT-2":
                out.append(f"{base}/code/bayes-cast")
        elif kind == "overlay":
            out.append(f"{base}/overlays/{token}-overlay.img")
        elif kind == "weights":
            out.append(f"{base}/weights")
        elif kind == "data":
            out.append(f"{base}/weights")
            out.append(f"{base}/data")
        elif kind == "sif" and root:
            for s in sorted(Path(f"{root}/images").glob("*.sif")) if Path(f"{root}/images").is_dir() else []:
                out.append(str(s))

    # de-dupe preserving order
    seen: set[str] = set()
    uniq = [x for x in out if not (x in seen or seen.add(x))]
    # For absolute-path kinds, surface existing paths first. Config basenames are
    # resolved relative to examples/ at runtime, so don't rank them by existence.
    if kind != "config":
        uniq.sort(key=lambda p: not Path(p).expanduser().exists())
    return uniq


def path_exists(p: str) -> bool:
    return bool(p) and Path(p).expanduser().exists()
