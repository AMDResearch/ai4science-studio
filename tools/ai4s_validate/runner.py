from __future__ import annotations

from pathlib import Path

from ai4s_validate.discover import discover_models, repo_root_from
from ai4s_validate.docs import check_docs
from ai4s_validate.findings import Result
from ai4s_validate.index import check_index, write_index
from ai4s_validate.manifests import check_manifests
from ai4s_validate.preflight import check_preflight_dry_run
from ai4s_validate.scripts import check_scripts
from ai4s_validate.site_leaks import check_site_leaks


def _root(root: Path | None) -> Path:
    return root or repo_root_from()


def run_fast(root: Path | None = None) -> Result:
    """Manifest schema + generated index. Suitable for pre-commit."""
    root = _root(root)
    models = discover_models(root)
    result = Result()
    if not models:
        from ai4s_validate.findings import Finding

        result.add(Finding("error", "no-models", "no */models/*/model.yaml files found"))
        return result
    result.extend(check_manifests(root, models).findings)
    result.extend(check_index(root, models).findings)
    return result


def run_all(root: Path | None = None, *, preflight: bool = True) -> Result:
    root = _root(root)
    models = discover_models(root)
    result = run_fast(root)
    result.extend(check_scripts(root, models).findings)
    result.extend(check_site_leaks(root, models).findings)
    result.extend(check_docs(root, models).findings)
    if preflight:
        result.extend(check_preflight_dry_run(root, models).findings)
    return result


def fix(root: Path | None = None) -> Path:
    root = _root(root)
    models = discover_models(root)
    return write_index(root, models)
