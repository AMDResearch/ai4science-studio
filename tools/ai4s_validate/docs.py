from __future__ import annotations

import re
from pathlib import Path

from ai4s_validate.discover import ModelEntry
from ai4s_validate.findings import Finding, Result

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DISCLAIMER_RE = re.compile(r"research\s*/\s*engineering use only", re.IGNORECASE)


def _is_external(href: str) -> bool:
    h = href.strip()
    return (
        h.startswith("http://")
        or h.startswith("https://")
        or h.startswith("mailto:")
        or h.startswith("#")
        or h.startswith("mailto:")
    )


def check_docs(root: Path, models: list[ModelEntry]) -> Result:
    result = Result()
    for model in models:
        model_dir = root / model.path
        readme = model_dir / "README.md"
        if not readme.is_file():
            result.add(
                Finding(
                    "error",
                    "missing-model-readme",
                    "model README.md is missing",
                    model=model.slug,
                    file=str(model.path / "README.md"),
                )
            )
            continue

        text = readme.read_text(encoding="utf-8", errors="replace")
        hf_id = model.data.get("hf_id")
        if hf_id == "N/A" and not re.search(r"obtaining model weights", text, re.IGNORECASE):
            # weight_source in the manifest is an acceptable substitute
            if not model.data.get("weight_source"):
                result.add(
                    Finding(
                        "warning",
                        "missing-weights-section",
                        'hf_id is N/A: add an "Obtaining model weights" README section or weight_source in model.yaml',
                        model=model.slug,
                        file=str(model.path / "README.md"),
                    )
                )

        if model.domain == "healthcare" and not DISCLAIMER_RE.search(text):
            result.add(
                Finding(
                    "error",
                    "healthcare-readme-disclaimer",
                    "healthcare model README must include the research/engineering-only disclaimer",
                    model=model.slug,
                    file=str(model.path / "README.md"),
                )
            )

        _check_md_links(root, model, readme, text, result)

        recipes_dir = model_dir / "recipes"
        if recipes_dir.is_dir():
            for child in sorted(recipes_dir.iterdir()):
                if child.is_dir() and not (child / "README.md").is_file():
                    result.add(
                        Finding(
                            "error",
                            "missing-recipe-readme",
                            f"recipes/{child.name}/ has no README.md",
                            model=model.slug,
                            file=str((model.path / "recipes" / child.name / "README.md")),
                        )
                    )

        for recipe in model.data.get("recipes") or []:
            if not isinstance(recipe, dict):
                continue
            runnable = bool(
                recipe.get("script") or recipe.get("slurm") or recipe.get("sbatch_script")
            )
            if not runnable:
                continue
            recipe_path = recipe.get("recipe_path")
            if not isinstance(recipe_path, str) or not recipe_path:
                continue
            readme = model_dir / recipe_path / "README.md"
            if not readme.is_file():
                continue
            rtext = readme.read_text(encoding="utf-8", errors="replace")
            if "examples" not in rtext.lower():
                result.add(
                    Finding(
                        "warning",
                        "recipe-no-examples",
                        "runnable recipe README should link to examples/",
                        model=model.slug,
                        file=str(readme.relative_to(root)),
                    )
                )

        examples_readme = model_dir / "examples" / "README.md"
        if examples_readme.is_file():
            _check_md_links(
                root,
                model,
                examples_readme,
                examples_readme.read_text(encoding="utf-8", errors="replace"),
                result,
            )

    return result


def _check_md_links(
    root: Path,
    model: ModelEntry,
    md_path: Path,
    text: str,
    result: Result,
) -> None:
    rel = str(md_path.relative_to(root))
    for i, line in enumerate(text.splitlines(), start=1):
        for match in MD_LINK.finditer(line):
            href = match.group(1).strip().strip("<>")
            href = href.split()[0] if href else href  # drop optional title
            if not href or _is_external(href):
                continue
            path_part = href.split("#", 1)[0]
            if not path_part:
                continue
            target = (md_path.parent / path_part).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                # link escaped the repo; still check existence
                pass
            if not target.exists():
                result.add(
                    Finding(
                        "warning",
                        "broken-link",
                        f"relative link does not resolve: {href}",
                        model=model.slug,
                        file=rel,
                        line=i,
                    )
                )
