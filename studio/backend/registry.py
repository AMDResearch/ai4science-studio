"""Model registry — loads models.yaml + per-model model.yaml files."""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any

# Root of the cloned ai4science-studio repo (two levels up from studio/backend/)
REPO_ROOT = Path(__file__).resolve().parents[2]

_DOMAIN_TITLES = {
    "earth_science": "Earth Science",
    "material_science": "Material Science",
    "healthcare": "Healthcare & Life Sciences",
    "physics_simulation": "Physics Simulation",
    "protein_folding": "Protein Folding",
}

_DOMAIN_ICONS = {
    "earth_science": "🌍",
    "material_science": "⚗️",
    "healthcare": "🧬",
    "physics_simulation": "⚡",
    "protein_folding": "🔬",
}


def _load_models_index() -> list[dict]:
    idx_path = REPO_ROOT / "models.yaml"
    if not idx_path.exists():
        return []
    with open(idx_path) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def _load_model_yaml(entry: dict) -> dict:
    model_yaml = REPO_ROOT / entry["path"] / "model.yaml"
    if model_yaml.exists():
        with open(model_yaml) as f:
            detail = yaml.safe_load(f) or {}
    else:
        detail = {}
    return {**entry, **detail}


def _normalize_images(raw: Any) -> list[str]:
    """container_image may be a single string or a list in model.yaml."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw if x]


def list_domains() -> list[dict]:
    models = _load_models_index()
    domains: dict[str, dict] = {}
    for m in models:
        d = m.get("domain", "unknown")
        if d not in domains:
            domains[d] = {
                "slug": d,
                "title": _DOMAIN_TITLES.get(d, d.replace("_", " ").title()),
                "icon": _DOMAIN_ICONS.get(d, "🔬"),
                "model_count": 0,
            }
        domains[d]["model_count"] += 1
    return list(domains.values())


def list_models(domain: str) -> list[dict]:
    models = _load_models_index()
    result = []
    for entry in models:
        if entry.get("domain") != domain:
            continue
        full = _load_model_yaml(entry)
        result.append({
            "slug": full.get("slug") or full.get("name", "unknown"),
            "name": full.get("name") or full.get("slug", "Unknown"),
            "domain": domain,
            "task": full.get("task", ""),
            "hf_id": full.get("hf_id", ""),
            "license": full.get("license", ""),
            "vram_gb": full.get("vram_gb"),
            "validated_hardware": full.get("validated_hardware", []),
            "tasks_available": full.get("tasks_available", ["inference"]),
            "container_images": _normalize_images(full.get("container_image")),
            "recipes": full.get("recipes", []),
        })
    return result


def get_model(slug: str) -> dict | None:
    models = _load_models_index()
    for entry in models:
        if entry.get("slug") == slug:
            full = _load_model_yaml(entry)
            full["container_images"] = _normalize_images(full.get("container_image"))
            return full
    return None


def add_model(payload: dict) -> bool:
    """Write a new model.yaml and append to models.yaml index."""
    domain = payload.get("domain", "earth_science")
    slug = payload["slug"]
    model_dir = REPO_ROOT / domain / "models" / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    model_yaml = {
        "name": payload.get("name", slug),
        "hf_id": payload.get("hf_id", ""),
        "license": payload.get("license", "Unknown"),
        "task": payload.get("task", ""),
        "domain": domain,
        "container_image": _normalize_images(payload.get("container_image")),
        "vram_gb": payload.get("vram_gb"),
        "validated_hardware": payload.get("validated_hardware", []),
        "tasks_available": payload.get("tasks_available", ["inference"]),
        "curated_prompts": payload.get("curated_prompts", []),
    }
    with open(model_dir / "model.yaml", "w") as f:
        yaml.dump(model_yaml, f, default_flow_style=False)

    # Append to models.yaml index
    idx_path = REPO_ROOT / "models.yaml"
    with open(idx_path) as f:
        idx = yaml.safe_load(f) or {"models": []}
    # Remove existing entry for same slug if any
    idx["models"] = [m for m in idx["models"] if m.get("slug") != slug]
    idx["models"].append({
        "slug": slug,
        "domain": domain,
        "hf_id": payload.get("hf_id", ""),
        "license": payload.get("license", "Unknown"),
        "task": payload.get("task", ""),
        "tasks_available": payload.get("tasks_available", ["inference"]),
        "path": f"{domain}/models/{slug}",
    })
    with open(idx_path, "w") as f:
        yaml.dump(idx, f, default_flow_style=False)
    return True
