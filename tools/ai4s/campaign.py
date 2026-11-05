"""Load and resolve campaign YAML into typed stage plans."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ai4s.paths import repo_root_from
from ai4s.schema_util import validate


class CampaignError(ValueError):
    pass


def load_campaign(path: Path, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root_from()
    path = path.resolve()
    if not path.is_file():
        raise CampaignError(f"campaign file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CampaignError("campaign.yaml must be a mapping")
    errors = validate(data, "campaign.schema.json", root)
    if errors:
        raise CampaignError("campaign schema errors:\n- " + "\n- ".join(errors))
    return data


def resolve_inputs(
    campaign: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve campaign inputs from defaults, env, and CLI overrides."""
    overrides = overrides or {}
    declared = campaign.get("inputs") or {}
    resolved: dict[str, Any] = {}
    missing: list[str] = []

    for name, spec in declared.items():
        if not isinstance(spec, dict):
            continue
        value: Any = None
        if name in overrides and overrides[name] is not None:
            value = overrides[name]
        else:
            env_name = spec.get("env")
            if isinstance(env_name, str) and env_name in os.environ:
                value = os.environ[env_name]
            elif "default" in spec:
                value = spec["default"]
        if value is None and spec.get("required", False):
            missing.append(name)
            continue
        if value is not None:
            resolved[name] = _coerce(value, str(spec.get("type", "string")))

    for key, value in overrides.items():
        if key not in resolved and value is not None:
            resolved[key] = value

    if missing:
        raise CampaignError(f"missing required campaign inputs: {', '.join(missing)}")
    return resolved


def _coerce(value: Any, typ: str) -> Any:
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    if typ == "integer":
        return int(value)
    if typ == "number":
        return float(value)
    if typ == "path":
        return str(Path(str(value)).expanduser())
    if typ == "object":
        return value
    return str(value)


def stage_plan(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    stages = campaign.get("stages") or []
    if not stages:
        raise CampaignError("campaign has no stages")
    ids = [s["id"] for s in stages]
    if len(ids) != len(set(ids)):
        raise CampaignError("campaign stage ids must be unique")
    return list(stages)
