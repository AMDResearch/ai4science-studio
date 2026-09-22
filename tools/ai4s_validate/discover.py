from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DOMAINS = (
    "earth_science",
    "material_science",
    "protein_folding",
    "healthcare",
    "physics_simulation",
)

DOMAIN_LABELS = {
    "earth_science": "Earth Science",
    "material_science": "Material Science",
    "healthcare": "Healthcare & Life Sciences",
    "physics_simulation": "Physics Simulation",
    "protein_folding": "Protein Folding",
}


@dataclass(frozen=True)
class ModelEntry:
    slug: str
    domain: str
    path: Path  # relative to repo root
    manifest_path: Path
    data: dict[str, Any]

    @property
    def abs_path(self) -> Path:
        return self.path


def repo_root_from(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "tools" / "ai4s_validate").is_dir() and (candidate / "schemas" / "model.schema.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate AI4Science Studio repo root")


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover_models(root: Path) -> list[ModelEntry]:
    found: list[ModelEntry] = []
    for domain in DOMAINS:
        models_dir = root / domain / "models"
        if not models_dir.is_dir():
            continue
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest = child / "model.yaml"
            if not manifest.is_file():
                continue
            data = load_yaml(manifest) or {}
            if not isinstance(data, dict):
                data = {}
            found.append(
                ModelEntry(
                    slug=child.name,
                    domain=domain,
                    path=child.relative_to(root),
                    manifest_path=manifest.relative_to(root),
                    data=data,
                )
            )
    return found


def recipe_slurm_path(recipe: dict[str, Any]) -> str | None:
    """HydraGNN uses sbatch_script; everyone else uses slurm."""
    for key in ("slurm", "sbatch_script"):
        val = recipe.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
