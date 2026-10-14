"""JSON Schema loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ai4s.paths import schemas_dir


def load_schema(name: str, root: Path | None = None) -> dict[str, Any]:
    path = schemas_dir(root) / name
    return json.loads(path.read_text(encoding="utf-8"))


def validate(instance: Any, schema_name: str, root: Path | None = None) -> list[str]:
    schema = load_schema(schema_name, root)
    validator = Draft202012Validator(schema)
    return sorted(e.message for e in validator.iter_errors(instance))
