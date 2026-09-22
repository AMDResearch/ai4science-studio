from __future__ import annotations

import re
from pathlib import Path

from ai4s_validate.discover import ModelEntry
from ai4s_validate.findings import Finding, Result

# Docker-style name:tag (registry optional). No digest, no pull — syntax only.
_IMAGE_REF = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)*"
    r":[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_SIF_REF = re.compile(r"^[\w./-]+\.sif$")


def _iter_images(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _ok_image(ref: str) -> bool:
    return bool(_IMAGE_REF.fullmatch(ref) or _SIF_REF.fullmatch(ref))


def check_container_images(root: Path, models: list[ModelEntry]) -> Result:
    """CPU check: container_image strings look like name:tag or *.sif. Does not pull."""
    del root  # reserved for a future optional skopeo inspect
    result = Result()
    for model in models:
        rel = str(model.manifest_path)
        refs = _iter_images(model.data.get("container_image"))
        if not refs:
            result.add(
                Finding(
                    "warning",
                    "missing-container-image",
                    "model.yaml has no container_image (name:tag or .sif)",
                    model=model.slug,
                    file=rel,
                )
            )
            continue
        for ref in refs:
            if not _ok_image(ref):
                result.add(
                    Finding(
                        "error",
                        "container-image-format",
                        f"container_image {ref!r} must be name:tag or a .sif path",
                        model=model.slug,
                        file=rel,
                    )
                )
    return result
