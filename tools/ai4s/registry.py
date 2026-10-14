"""Reusable agent catalog (repo) + optional cluster registry merge.

Model-local tasks are NOT registered globally. They are resolved from
``*/models/*/model.yaml`` via ``model:recipe`` refs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai4s.paths import cluster_registry_dir, repo_agent_catalog, repo_root_from
from ai4s.schema_util import validate


class RegistryError(ValueError):
    pass


def _load_agent_yaml(path: Path, root: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RegistryError(f"agent.yaml must be a mapping: {path}")
    errors = validate(data, "agent.schema.json", root)
    if errors:
        raise RegistryError(f"{path}: " + "; ".join(errors))
    data["_path"] = str(path)
    data["_dir"] = str(path.parent)
    return data


def discover_repo_agents(root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = root or repo_root_from()
    catalog = repo_agent_catalog(root)
    agents: dict[str, dict[str, Any]] = {}
    if not catalog.is_dir():
        return agents
    for path in sorted(catalog.glob("*/agent.yaml")):
        agent = _load_agent_yaml(path, root)
        agents[agent["name"]] = agent
    return agents


def discover_cluster_agents(root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = root or repo_root_from()
    reg = cluster_registry_dir(root)
    agents: dict[str, dict[str, Any]] = {}
    if reg is None or not reg.is_dir():
        return agents
    for path in sorted(reg.glob("**/agent.yaml")):
        agent = _load_agent_yaml(path, root)
        agents[agent["name"]] = agent
    return agents


def merge_agent_registry(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Repo catalog first; cluster registry overrides same names."""
    root = root or repo_root_from()
    merged = discover_repo_agents(root)
    merged.update(discover_cluster_agents(root))
    return merged


def get_agent(name: str, root: Path | None = None) -> dict[str, Any]:
    agents = merge_agent_registry(root)
    if name not in agents:
        raise RegistryError(
            f"reusable agent not found: {name}. "
            "Place agent.yaml under agents/<name>/ or ${AI4S_SHARED_DIR}/registry/."
        )
    return agents[name]


def discover_model_recipes(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Map ``ModelName:task`` -> recipe metadata from model.yaml files."""
    root = root or repo_root_from()
    out: dict[str, dict[str, Any]] = {}
    for model_yaml in sorted(root.glob("*/models/*/model.yaml")):
        data = yaml.safe_load(model_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        model_name = str(data.get("name") or model_yaml.parent.name)
        model_dir = model_yaml.parent
        for recipe in data.get("recipes") or []:
            if not isinstance(recipe, dict) or "task" not in recipe:
                continue
            key = f"{model_name}:{recipe['task']}"
            out[key] = {
                "model": model_name,
                "task": recipe["task"],
                "description": recipe.get("description", ""),
                "recipe_path": recipe.get("recipe_path"),
                "script": recipe.get("script"),
                "slurm": recipe.get("slurm"),
                "model_dir": str(model_dir),
                "model_yaml": str(model_yaml),
            }
    return out


def get_task(ref: str, root: Path | None = None) -> dict[str, Any]:
    recipes = discover_model_recipes(root)
    if ref not in recipes:
        raise RegistryError(
            f"task not found: {ref}. Expected model-local recipe key like ORBIT-2:inference."
        )
    return recipes[ref]
