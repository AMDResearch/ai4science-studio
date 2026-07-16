"""Scaffold a new model into the repo from _template/.

Writes:
  <domain>/models/<slug>/README.md
  <domain>/models/<slug>/model.yaml
  <domain>/models/<slug>/recipes/.gitkeep
  <domain>/models/<slug>/examples/.gitkeep
and appends an entry to the root models.yaml `models:` list.

Follows CLAUDE.md conventions: link don't vendor, no large artifacts, slug rule
org/model -> org__model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import models_index, repo_root

DOMAINS = [
    "earth_science",
    "material_science",
    "protein_folding",
    "healthcare",
    "physics_simulation",
]


@dataclass
class NewModel:
    slug: str
    domain: str
    hf_id: str
    license: str
    task: str
    upstream_code: str = ""
    paper: str = ""
    container_image: str = ""


def slug_from_hf(hf_id: str) -> str:
    """org/model -> org__model. Public names pass through unchanged."""
    if "/" in hf_id and hf_id != "N/A":
        return hf_id.replace("/", "__")
    return hf_id


def validate(nm: NewModel) -> list[str]:
    errs: list[str] = []
    if not nm.slug or not re.fullmatch(r"[A-Za-z0-9._\-]+", nm.slug):
        errs.append("slug must be non-empty and contain only [A-Za-z0-9._-]")
    if nm.domain not in DOMAINS:
        errs.append(f"domain must be one of {DOMAINS}")
    if not nm.task:
        errs.append("task is required")
    if not nm.license:
        errs.append("license is required")
    dest = repo_root() / nm.domain / "models" / nm.slug
    if dest.exists():
        errs.append(f"model folder already exists: {dest}")
    return errs


def _render_readme(nm: NewModel) -> str:
    hf_line = (
        f"**Hugging Face:** [`{nm.hf_id}`](https://huggingface.co/{nm.hf_id})"
        if nm.hf_id and nm.hf_id != "N/A"
        else "**Hugging Face:** N/A (document weight source below)"
    )
    return f"""# {nm.slug}

{hf_line}
**Code:** {nm.upstream_code or "TODO"}
**Paper:** {nm.paper or "TODO"}

{nm.task}

## License and attribution

License: {nm.license}

Cite the upstream repository and paper. Add an entry to
[`ACKNOWLEDGEMENTS.md`](../../../ACKNOWLEDGEMENTS.md) following the existing
per-model format before opening a PR.

## Recipes

Add how-to docs under `recipes/` (one subfolder per task) and ready-to-run
scripts under `examples/`. Do not commit large checkpoints or datasets;
document how to obtain them instead.
"""


def _model_yaml_dict(nm: NewModel) -> dict:
    d: dict = {
        "name": nm.slug,
        "hf_id": nm.hf_id or "N/A",
        "license": nm.license,
        "task": nm.task,
        "domain": nm.domain,
        "upstream_code": nm.upstream_code,
        "paper": nm.paper,
        "container_image": nm.container_image
        or "rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0",
        "vram_gb": None,
        "validated_hardware": [],
        "recipes": [],
        "env_vars": {},
    }
    return d


def create(nm: NewModel) -> Path:
    """Create the model folder + files, and register in models.yaml. Returns dest."""
    errs = validate(nm)
    if errs:
        raise ValueError("; ".join(errs))

    dest = repo_root() / nm.domain / "models" / nm.slug
    (dest / "recipes").mkdir(parents=True, exist_ok=True)
    (dest / "examples").mkdir(parents=True, exist_ok=True)
    (dest / "recipes" / ".gitkeep").touch()
    (dest / "examples" / ".gitkeep").touch()

    (dest / "README.md").write_text(_render_readme(nm))
    with open(dest / "model.yaml", "w") as fh:
        yaml.safe_dump(_model_yaml_dict(nm), fh, sort_keys=False)

    _register_in_index(nm)
    return dest


def _register_in_index(nm: NewModel) -> None:
    """Append a new entry to models.yaml, preserving the file's comments/format.

    We append a text block rather than re-dumping the whole document so the
    existing section headers and hand-written comments survive.
    """
    idx = models_index()
    text = idx.read_text()

    block = (
        f"\n  - slug: {nm.slug}\n"
        f"    domain: {nm.domain}\n"
        f"    hf_id: {nm.hf_id or 'N/A'}\n"
        f"    license: {_yaml_scalar(nm.license)}\n"
        f"    task: {_yaml_scalar(nm.task)}\n"
        f"    tasks_available: []\n"
        f"    path: {nm.domain}/models/{nm.slug}\n"
    )
    if not text.endswith("\n"):
        text += "\n"
    idx.write_text(text + block)


def _yaml_scalar(s: str) -> str:
    """Quote a scalar if it contains YAML-special characters."""
    if s and re.search(r"[:#\[\]{}&*!|>'\"%@`,]", s):
        return '"' + s.replace('"', '\\"') + '"'
    return s
