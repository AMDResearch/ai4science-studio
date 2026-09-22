"""Flag cluster-specific paths, accounts, node lists, and job IDs in committed recipes.

Only **active** lines in shell/Python/YAML count as errors. Comments, module
docstrings, and Markdown may document OLCF paths (e.g. ``/lustre/orion``) or
example log names without failing CI.
"""

from __future__ import annotations

import re
from pathlib import Path

from ai4s_validate.discover import ModelEntry
from ai4s_validate.findings import Finding, Result

SITE_MOUNT = re.compile(
    r"(?<![A-Za-z0-9_])(/(?:shared|scratch|lustre|gpfs|nethome|home)/)"
    r"(?!\$|\{)"
    r"([A-Za-z][A-Za-z0-9._-]{1,64})"
)

ALLOWED_MOUNT_NAMES = {
    "user",
    "username",
    "your",
    "users",
    "data",
    "opt",
    "tmp",
    "shared",
}

SBATCH_PARTITION = re.compile(r"^#SBATCH\s+--partition=(\S+)")
SBATCH_ACCOUNT = re.compile(r"^#SBATCH\s+--account=(\S+)")
SBATCH_NODELIST = re.compile(r"^#SBATCH\s+--nodelist=(\S+)")

JOB_ARTIFACT = re.compile(
    r"\b(?:[A-Za-z][\w.-]*-)?(?:train|infer|inference|vis|overlay-build|pairtune)-\d{4,}\b"
)
NUMERIC_JOB = re.compile(r"\b(?:sbatch|scancel|sacct)\s+[0-9]{5,}\b")
JOBID_ASSIGN = re.compile(r"\b(?:SLURM_)?JOB[_-]?ID\s*=\s*[0-9]{5,}\b", re.IGNORECASE)

CODE_SUFFIXES = {".sh", ".py", ".bash", ".yaml", ".yml", ".toml"}
SCAN_SUFFIXES = CODE_SUFFIXES | {".md"}
PLACEHOLDER_PREFIX = "YOUR_"
DOCSTRING_MARK = re.compile(r'"""|\'\'\'')


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip('"').strip("'")
    if not v:
        return True
    if v.startswith("$") or v.startswith("${") or "<" in v:
        return True
    return v.upper().startswith(PLACEHOLDER_PREFIX)


def _active_code_lines(text: str, suffix: str) -> set[int]:
    """1-based line numbers that are executable / live config, not comments or docs."""
    if suffix == ".md":
        return set()
    active: set[int] = set()
    in_py_doc = False
    doc_quote = ""
    for i, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip()
        if suffix == ".py":
            if in_py_doc:
                if doc_quote in raw:
                    in_py_doc = False
                continue
            m = DOCSTRING_MARK.search(stripped)
            if m and stripped.startswith(("'", '"')):
                quote = m.group(0)
                rest = stripped[m.end() :]
                if quote not in rest:
                    in_py_doc = True
                    doc_quote = quote
                continue
            if stripped.startswith("#"):
                continue
            active.add(i)
            continue
        if suffix in {".sh", ".bash"}:
            if stripped.startswith("#SBATCH"):
                active.add(i)
                continue
            if stripped.startswith("#") or stripped == "":
                continue
            active.add(i)
            continue
        if suffix in {".yaml", ".yml"}:
            if stripped.startswith("#") or stripped == "":
                continue
            active.add(i)
            continue
        if stripped.startswith("#"):
            continue
        active.add(i)
    return active


def scan_text(text: str, *, rel: str, model: str | None, suffix: str | None = None) -> list[Finding]:
    suffix = suffix if suffix is not None else Path(rel).suffix.lower()
    active = _active_code_lines(text, suffix)
    findings: list[Finding] = []

    for i, raw in enumerate(text.splitlines(), start=1):
        if i not in active:
            continue
        for match in SITE_MOUNT.finditer(raw):
            name = match.group(2)
            if name.lower() in ALLOWED_MOUNT_NAMES:
                continue
            findings.append(
                Finding(
                    "error",
                    "site-path",
                    f"site-specific path {match.group(1)}{name} — use $AI4S_SHARED_DIR / $HOME / $USER",
                    model=model,
                    file=rel,
                    line=i,
                )
            )
        m = SBATCH_PARTITION.match(raw.lstrip()) or (
            SBATCH_PARTITION.match(raw) if raw.startswith("#SBATCH") else None
        )
        if m and not _is_placeholder(m.group(1)):
            findings.append(
                Finding(
                    "error",
                    "site-partition",
                    f"#SBATCH --partition={m.group(1)} is site-specific; use YOUR_PARTITION_HERE",
                    model=model,
                    file=rel,
                    line=i,
                )
            )
        m = SBATCH_ACCOUNT.match(raw.lstrip()) if raw.lstrip().startswith("#SBATCH") else None
        if m is None and raw.startswith("#SBATCH"):
            m = SBATCH_ACCOUNT.match(raw)
        if m and not _is_placeholder(m.group(1)):
            findings.append(
                Finding(
                    "error",
                    "site-account",
                    f"#SBATCH --account={m.group(1)} is site-specific; use YOUR_ACCOUNT_HERE",
                    model=model,
                    file=rel,
                    line=i,
                )
            )
        m = SBATCH_NODELIST.match(raw.lstrip()) if "#SBATCH" in raw else None
        if m and not _is_placeholder(m.group(1)):
            findings.append(
                Finding(
                    "error",
                    "site-nodelist",
                    f"#SBATCH --nodelist={m.group(1)} is site-specific; take nodelist from env / cluster config",
                    model=model,
                    file=rel,
                    line=i,
                )
            )
        for rx, label in (
            (JOB_ARTIFACT, "looks like a recorded SLURM job id"),
            (NUMERIC_JOB, "hardcoded job id"),
            (JOBID_ASSIGN, "hardcoded job id"),
        ):
            for match in rx.finditer(raw):
                findings.append(
                    Finding(
                        "error",
                        "site-job-id",
                        f"{label}: {match.group(0)}",
                        model=model,
                        file=rel,
                        line=i,
                    )
                )
    return findings


def _iter_scan_files(model_dir: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("examples", "recipes"):
        base = model_dir / sub
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES:
                out.append(path)
    return out


def check_site_leaks(root: Path, models: list[ModelEntry]) -> Result:
    result = Result()
    for model in models:
        model_dir = root / model.path
        for path in _iter_scan_files(model_dir):
            rel = str(path.relative_to(root))
            text = path.read_text(encoding="utf-8", errors="replace")
            for finding in scan_text(text, rel=rel, model=model.slug, suffix=path.suffix.lower()):
                result.add(finding)
    return result
