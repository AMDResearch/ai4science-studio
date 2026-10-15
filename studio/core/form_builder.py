"""Map a model's env_vars{} spec to form-widget descriptors.

Kept UI-agnostic: this module decides *what kind* of input each env var needs;
the Run page turns descriptors into Streamlit widgets. That split keeps the
heuristics unit-testable without importing streamlit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PATH_HINT = re.compile(
    r"(SIF|OVERLAY|ROOT|_DIR$|_DIR_|PATH|CHECKPOINT|CKPT|CONFIG|YAML|DATA_DIR|DATA_ROOT|OUTPUT_FILE)"
)


@dataclass
class Field:
    name: str
    widget: str  # "checkbox" | "number" | "select" | "text"
    default: str
    description: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)
    is_path: bool = False


def _parse_enum(description: str) -> list[str]:
    """Pull a small allow-list out of a description like 'fp32 | fp64 | bf16'
    or 'fp32, fp64 or bf16'. Returns [] if nothing enumerable is found."""
    # token | token | token
    if "|" in description:
        cand = [t.strip() for t in description.split("|")]
        cand = [c for c in cand if re.fullmatch(r"[A-Za-z0-9_.\-]+", c)]
        if len(cand) >= 2:
            return cand
    # "A, B or C" / "A, B, C"
    m = re.search(r"\b([\w.\-]+(?:\s*,\s*[\w.\-]+)+(?:\s+or\s+[\w.\-]+)?)", description)
    if m:
        chunk = m.group(1).replace(" or ", ", ")
        cand = [t.strip() for t in chunk.split(",")]
        cand = [c for c in cand if re.fullmatch(r"[A-Za-z0-9_.\-]+", c)]
        # only treat as enum if they look like short flag-ish tokens
        if len(cand) >= 2 and all(len(c) <= 12 for c in cand):
            return cand
    return []


def _looks_bool(default: str, description: str) -> bool:
    # Only treat 0/1 as a boolean toggle when the description gives a clear cue.
    # A bare count like NUM_BATCHES=1 must stay a number.
    if default in ("0", "1"):
        cue = re.search(
            r"(1=|0=|set to 1|set 1|enable|disable|\bflag\b|toggle|\bon\b|\boff\b|=on|=off)",
            description, re.I,
        )
        return bool(cue)
    return False


def build_field(name: str, spec: dict[str, Any]) -> Field:
    default = spec.get("default")
    default = "" if default is None else str(default)
    desc = str(spec.get("description", ""))
    required = bool(spec.get("required", False))

    is_path = bool(PATH_HINT.search(name.upper()))

    # Path-ish vars are always free text (with a discover affordance); never try
    # to infer an enum/number from their prose descriptions. An explicit enum:
    # field (future normalized schema) still wins for non-path vars.
    if is_path:
        enum: list[str] = list(spec.get("enum") or [])
        widget = "select" if enum else "text"
    else:
        enum = spec.get("enum") or _parse_enum(desc)
        if _looks_bool(default, desc):
            widget = "checkbox"
        elif enum:
            widget = "select"
        elif re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", default or ""):
            widget = "number"
        else:
            widget = "text"

    return Field(
        name=name,
        widget=widget,
        default=default,
        description=desc,
        required=required,
        options=list(enum),
        is_path=is_path,
    )


def build_fields(env_vars: dict[str, dict]) -> list[Field]:
    """Return fields ordered required-first, then alphabetical."""
    fields = [build_field(n, s or {}) for n, s in env_vars.items()]
    fields.sort(key=lambda f: (not f.required, f.name))
    return fields
