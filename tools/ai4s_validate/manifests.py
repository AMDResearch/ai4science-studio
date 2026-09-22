from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ai4s_validate.discover import ModelEntry, recipe_slurm_path
from ai4s_validate.findings import Finding, Result


def load_schema(root: Path) -> dict:
    path = root / "schemas" / "model.schema.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _hf_id_ok(hf_id: str) -> bool:
    if hf_id == "N/A":
        return True
    # org/model, allowing extra path segments used by some hubs
    parts = hf_id.split("/")
    return len(parts) >= 2 and all(p.strip() for p in parts)


def check_manifests(root: Path, models: list[ModelEntry]) -> Result:
    result = Result()
    schema = load_schema(root)
    validator = Draft202012Validator(schema)

    seen_slugs: dict[str, str] = {}
    for model in models:
        rel = str(model.manifest_path)
        if model.slug in seen_slugs:
            result.add(
                Finding(
                    "error",
                    "duplicate-slug",
                    f"slug {model.slug!r} also used at {seen_slugs[model.slug]}",
                    model=model.slug,
                    file=rel,
                )
            )
        seen_slugs[model.slug] = rel

        declared_domain = model.data.get("domain")
        if declared_domain and declared_domain != model.domain:
            result.add(
                Finding(
                    "error",
                    "domain-mismatch",
                    f"manifest domain={declared_domain!r} but folder is under {model.domain}/",
                    model=model.slug,
                    file=rel,
                )
            )

        hf_id = model.data.get("hf_id")
        if isinstance(hf_id, str) and not _hf_id_ok(hf_id):
            result.add(
                Finding(
                    "error",
                    "hf-id-format",
                    f"hf_id {hf_id!r} must be N/A or org/model",
                    model=model.slug,
                    file=rel,
                )
            )

        for err in validator.iter_errors(model.data):
            path = "/".join(str(p) for p in err.absolute_path) or "(root)"
            result.add(
                Finding(
                    "error",
                    "schema",
                    f"{path}: {err.message}",
                    model=model.slug,
                    file=rel,
                )
            )

        recipes = model.data.get("recipes") or []
        if isinstance(recipes, list):
            for i, recipe in enumerate(recipes):
                if not isinstance(recipe, dict):
                    continue
                script = recipe.get("script")
                if isinstance(script, str) and script:
                    target = root / model.path / script
                    if not target.is_file():
                        result.add(
                            Finding(
                                "error",
                                "missing-script",
                                f"recipes[{i}].script {script!r} does not exist",
                                model=model.slug,
                                file=rel,
                            )
                        )
                slurm = recipe_slurm_path(recipe)
                if slurm:
                    target = root / model.path / slurm
                    if not target.is_file():
                        result.add(
                            Finding(
                                "error",
                                "missing-slurm",
                                f"recipes[{i}] slurm/sbatch_script {slurm!r} does not exist",
                                model=model.slug,
                                file=rel,
                            )
                        )
                recipe_path = recipe.get("recipe_path")
                if isinstance(recipe_path, str) and recipe_path:
                    rdir = root / model.path / recipe_path
                    if not rdir.is_dir():
                        result.add(
                            Finding(
                                "error",
                                "missing-recipe-dir",
                                f"recipes[{i}].recipe_path {recipe_path!r} is not a directory",
                                model=model.slug,
                                file=rel,
                            )
                        )
                    elif not (rdir / "README.md").is_file():
                        result.add(
                            Finding(
                                "error",
                                "missing-recipe-readme",
                                f"{recipe_path} has no README.md",
                                model=model.slug,
                                file=str((model.path / recipe_path / "README.md")),
                            )
                        )

        if model.domain == "healthcare":
            disclaimer = model.data.get("disclaimer") or ""
            if "research" not in str(disclaimer).lower():
                result.add(
                    Finding(
                        "error",
                        "healthcare-disclaimer",
                        "healthcare models must set a research/engineering disclaimer in model.yaml",
                        model=model.slug,
                        file=rel,
                    )
                )

    return result
