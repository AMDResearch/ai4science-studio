"""Optional JSON-schema validation for model.yaml / cluster-config.yaml.

Validation is best-effort and non-fatal: if jsonschema isn't installed or the
schema can't be read, validate_* returns an empty error list so the app still
runs. Errors are returned as short human-readable strings for the UI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import repo_root

_SCHEMA_DIR = repo_root() / "studio" / "schema"


def _load_schema(name: str) -> dict | None:
    p = _SCHEMA_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _validate(data: Any, schema_name: str) -> list[str]:
    schema = _load_schema(schema_name)
    if schema is None:
        return []
    try:
        import jsonschema
    except Exception:  # noqa: BLE001
        return []
    validator = jsonschema.Draft7Validator(schema)
    errs = []
    for e in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in e.path) or "(root)"
        errs.append(f"{loc}: {e.message}")
    return errs


def validate_model_yaml(data: dict) -> list[str]:
    return _validate(data, "model.schema.json")


def validate_cluster_config(data: dict) -> list[str]:
    return _validate(data, "cluster-config.schema.json")
