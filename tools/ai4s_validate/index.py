from __future__ import annotations

import json
from pathlib import Path

from ai4s_validate.discover import DOMAIN_LABELS, DOMAINS, ModelEntry, load_yaml
from ai4s_validate.findings import Finding, Result


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    s = str(value)
    needs_quotes = (
        not s
        or s.strip() != s
        or any(c in s for c in ":{}[]#&*!|>%@`'\"")
        or "·" in s
        or s.lower() in {"true", "false", "null", "yes", "no", "on", "off"}
    )
    if needs_quotes:
        return json.dumps(s, ensure_ascii=False)
    return s

INDEX_HEADER = """# AI4Science Studio — Model Index
#
# GENERATED FILE — do not edit by hand.
# Source of truth: <domain>/models/<slug>/model.yaml
# Regenerate: make fix   (or: python3 -m ai4s_validate fix)
#
# Agents: read this file first to discover what's available.
# For full details on any model, read <domain>/models/<slug>/model.yaml.

"""

INDEX_KEYS = ("slug", "domain", "hf_id", "license", "summary", "task", "tasks_available", "path")


def build_index_entries(models: list[ModelEntry]) -> list[dict]:
    by_domain: dict[str, list[ModelEntry]] = {d: [] for d in DOMAINS}
    for m in models:
        by_domain.setdefault(m.domain, []).append(m)

    entries: list[dict] = []
    for domain in DOMAINS:
        for m in sorted(by_domain.get(domain, []), key=lambda x: x.slug.lower()):
            recipes = m.data.get("recipes") or []
            tasks = []
            if isinstance(recipes, list):
                for r in recipes:
                    if isinstance(r, dict) and r.get("task"):
                        tasks.append(str(r["task"]))
            entries.append(
                {
                    "slug": m.slug,
                    "domain": m.domain,
                    "hf_id": m.data.get("hf_id"),
                    "license": m.data.get("license"),
                    "summary": m.data.get("summary"),
                    "task": m.data.get("task"),
                    "tasks_available": tasks,
                    "path": str(m.path).replace("\\", "/"),
                }
            )
    return entries


def render_index(entries: list[dict]) -> str:
    lines = [INDEX_HEADER.rstrip(), "", "models:"]
    current_domain = None
    for entry in entries:
        domain = entry["domain"]
        if domain != current_domain:
            label = DOMAIN_LABELS.get(domain, domain)
            lines.append(f"  # --- {label} ---")
            current_domain = domain
        lines.append(f"  - slug: {entry['slug']}")
        lines.append(f"    domain: {_yaml_scalar(entry['domain'])}")
        lines.append(f"    hf_id: {_yaml_scalar(entry.get('hf_id'))}")
        lines.append(f"    license: {_yaml_scalar(entry.get('license'))}")
        lines.append(f"    summary: {_yaml_scalar(entry.get('summary'))}")
        lines.append(f"    task: {_yaml_scalar(entry.get('task'))}")
        tasks = entry.get("tasks_available") or []
        inner = ", ".join(tasks)
        lines.append(f"    tasks_available: [{inner}]")
        lines.append(f"    path: {entry['path']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def expected_index_text(models: list[ModelEntry]) -> str:
    return render_index(build_index_entries(models))


def write_index(root: Path, models: list[ModelEntry]) -> Path:
    path = root / "models.yaml"
    path.write_text(expected_index_text(models), encoding="utf-8")
    return path


def _normalize_loaded(data: dict) -> list[dict]:
    models = data.get("models") or []
    out = []
    for item in models:
        if not isinstance(item, dict):
            continue
        slim = {k: item.get(k) for k in INDEX_KEYS}
        out.append(slim)
    return out


def check_index(root: Path, models: list[ModelEntry]) -> Result:
    result = Result()
    path = root / "models.yaml"
    if not path.is_file():
        result.add(Finding("error", "missing-index", "models.yaml is missing", file="models.yaml"))
        return result

    on_disk = path.read_text(encoding="utf-8")
    expected = expected_index_text(models)
    if on_disk != expected:
        # Also compare semantically so a comment-only mismatch is still an error
        # (file is generated). Give a shorter semantic hint if keys differ.
        loaded = load_yaml(path) or {}
        got = _normalize_loaded(loaded if isinstance(loaded, dict) else {})
        want = build_index_entries(models)
        if got != want:
            result.add(
                Finding(
                    "error",
                    "index-drift",
                    "models.yaml does not match per-model manifests; run `make fix`",
                    file="models.yaml",
                )
            )
        else:
            result.add(
                Finding(
                    "error",
                    "index-format",
                    "models.yaml content matches semantically but formatting drifted; run `make fix`",
                    file="models.yaml",
                )
            )
    return result
