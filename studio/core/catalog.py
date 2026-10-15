"""Load and normalize model metadata from models.yaml + per-model model.yaml.

Normalization handled here so the rest of the app sees one consistent shape:
- container_image -> always a list[str]
- recipe SLURM script field -> aliased from either `slurm:` or `sbatch_script:`
  into `sbatch_script`, tolerating a recipe with neither.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import models_index, repo_root


@dataclass
class Recipe:
    task: str
    description: str = ""
    recipe_path: str = ""
    script: str | None = None
    sbatch_script: str | None = None  # aliased from slurm|sbatch_script
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def runnable(self) -> bool:
        """A recipe is runnable by the simple engine if it has an sbatch script."""
        return bool(self.sbatch_script)


@dataclass
class Model:
    slug: str
    domain: str
    path: str  # relative to repo root
    name: str = ""
    hf_id: str = ""
    license: str = ""
    task: str = ""
    tasks_available: list[str] = field(default_factory=list)
    upstream_code: str = ""
    paper: str = ""
    container_image: list[str] = field(default_factory=list)
    vram_gb: Any = None
    validated_hardware: list[str] = field(default_factory=list)
    disclaimer: str = ""
    data_source: str = ""
    weight_source: str = ""
    model_variants: list[dict] = field(default_factory=list)
    recipes: list[Recipe] = field(default_factory=list)
    env_vars: dict[str, dict] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def abs_path(self) -> Path:
        return repo_root() / self.path

    @property
    def examples_dir(self) -> Path:
        return self.abs_path / "examples"

    def recipe_for(self, task: str) -> Recipe | None:
        for r in self.recipes:
            if r.task == task:
                return r
        return None


def _as_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    return [str(val)]


def _normalize_recipe(raw: dict) -> Recipe:
    sbatch = raw.get("sbatch_script") or raw.get("slurm")
    return Recipe(
        task=str(raw.get("task", "")),
        description=str(raw.get("description", "")),
        recipe_path=str(raw.get("recipe_path", "")),
        script=raw.get("script"),
        sbatch_script=sbatch,
        raw=raw,
    )


def _load_model_yaml(index_entry: dict) -> Model:
    rel_path = index_entry["path"]
    my_path = repo_root() / rel_path / "model.yaml"
    data: dict[str, Any] = {}
    if my_path.exists():
        with open(my_path) as fh:
            data = yaml.safe_load(fh) or {}

    recipes = [_normalize_recipe(r) for r in (data.get("recipes") or [])]

    return Model(
        slug=index_entry["slug"],
        domain=str(data.get("domain") or index_entry.get("domain", "")),
        path=rel_path,
        name=str(data.get("name") or index_entry["slug"]),
        hf_id=str(data.get("hf_id") or index_entry.get("hf_id", "")),
        license=str(data.get("license") or index_entry.get("license", "")),
        task=str(data.get("task") or index_entry.get("task", "")),
        tasks_available=_as_list(index_entry.get("tasks_available"))
        or [r.task for r in recipes],
        upstream_code=str(data.get("upstream_code", "")),
        paper=str(data.get("paper") or ""),
        container_image=_as_list(data.get("container_image")),
        vram_gb=data.get("vram_gb"),
        validated_hardware=_as_list(data.get("validated_hardware")),
        disclaimer=str(data.get("disclaimer") or "").strip(),
        data_source=str(data.get("data_source") or ""),
        weight_source=str(data.get("weight_source") or ""),
        model_variants=data.get("model_variants") or [],
        recipes=recipes,
        env_vars=data.get("env_vars") or {},
        raw=data,
    )


@functools.lru_cache(maxsize=1)
def _load_catalog_cached(_mtime: float) -> tuple[Model, ...]:
    with open(models_index()) as fh:
        raw = yaml.safe_load(fh) or {}
    # models.yaml wraps the list under a top-level `models:` key.
    index = raw.get("models", raw) if isinstance(raw, dict) else raw
    models = [_load_model_yaml(entry) for entry in index]
    return tuple(models)


def load_catalog() -> list[Model]:
    """Return all models, cached and invalidated when models.yaml changes."""
    idx = models_index()
    mtime = idx.stat().st_mtime if idx.exists() else 0.0
    return list(_load_catalog_cached(mtime))


def get_model(slug: str) -> Model | None:
    for m in load_catalog():
        if m.slug == slug:
            return m
    return None


def domains() -> list[str]:
    return sorted({m.domain for m in load_catalog() if m.domain})


def all_tasks() -> list[str]:
    tasks: set[str] = set()
    for m in load_catalog():
        tasks.update(m.tasks_available)
    return sorted(tasks)


def all_licenses() -> list[str]:
    return sorted({m.license for m in load_catalog() if m.license})
