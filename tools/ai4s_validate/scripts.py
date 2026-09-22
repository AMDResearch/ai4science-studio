from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ai4s_validate.discover import ModelEntry
from ai4s_validate.findings import Finding, Result

SHELL_GLOBS = ("*.sh",)
ENV_ASSIGN = re.compile(
    r"""(?:^|[\s;])(?:export\s+)?([A-Z][A-Z0-9_]{1,64})=\$\{([A-Z][A-Z0-9_]{1,64}):-"""
)
ENV_GET = re.compile(r"""os\.environ\.get\(\s*["']([A-Z][A-Z0-9_]+)["']""")
NV_FLAG = re.compile(r"""(?:^|[\s=])--nv(?:[\s"']|$)""")
_SHELLCHECK_MISSING = False


def _iter_example_files(model_dir: Path) -> list[Path]:
    examples = model_dir / "examples"
    if not examples.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(examples.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".sh", ".py", ".bash"}:
            files.append(path)
    return files


def _has_crlf(path: Path) -> bool:
    data = path.read_bytes()
    return b"\r\n" in data


def check_scripts(root: Path, models: list[ModelEntry]) -> Result:
    global _SHELLCHECK_MISSING
    _SHELLCHECK_MISSING = False
    result = Result()
    for model in models:
        model_dir = root / model.path
        examples = model_dir / "examples"
        rel_examples = str(model.path / "examples")

        if not examples.is_dir():
            result.add(
                Finding(
                    "error",
                    "missing-examples",
                    "examples/ directory is missing",
                    model=model.slug,
                    file=rel_examples,
                )
            )
            continue

        docker = examples / "docker_run.sh"
        if not docker.is_file():
            result.add(
                Finding(
                    "error",
                    "missing-docker-run",
                    "examples/docker_run.sh is missing",
                    model=model.slug,
                    file=str(model.path / "examples" / "docker_run.sh"),
                )
            )

        preflights = list(examples.glob("preflight_*.py"))
        if not preflights:
            result.add(
                Finding(
                    "error",
                    "missing-preflight",
                    "examples/preflight_*.py is missing",
                    model=model.slug,
                    file=rel_examples,
                )
            )
        else:
            for pf in preflights:
                text = pf.read_text(encoding="utf-8", errors="replace")
                if "--dry-run" not in text:
                    result.add(
                        Finding(
                            "warning",
                            "preflight-no-dry-run",
                            "preflight script should accept --dry-run (skip GPU/import checks)",
                            model=model.slug,
                            file=str(pf.relative_to(root)),
                        )
                    )

        declared_env = set()
        env_vars = model.data.get("env_vars") or {}
        if isinstance(env_vars, dict):
            declared_env = set(env_vars.keys())

        for path in _iter_example_files(model_dir):
            rel = str(path.relative_to(root))
            if _has_crlf(path):
                result.add(
                    Finding(
                        "error",
                        "crlf",
                        "CRLF line endings; convert to LF",
                        model=model.slug,
                        file=rel,
                    )
                )
            if not os.access(path, os.X_OK):
                result.add(
                    Finding(
                        "warning",
                        "not-executable",
                        "file is not executable (chmod +x)",
                        model=model.slug,
                        file=rel,
                    )
                )

            if path.suffix == ".sh":
                _check_shell(root, model, path, rel, declared_env, result)

    return result


def _check_shell(
    root: Path,
    model: ModelEntry,
    path: Path,
    rel: str,
    declared_env: set[str],
    result: Result,
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^set\s+-[^\n]*pipefail", text, re.MULTILINE):
        result.add(
            Finding(
                "warning",
                "missing-pipefail",
                "shell script should use `set -euo pipefail`",
                model=model.slug,
                file=rel,
            )
        )

    name = path.name
    is_sbatch = name.startswith("sbatch_")
    is_amd = "_amd.sh" in name or name.endswith("_amd.sh")

    if is_sbatch:
        if "scontrol" not in text or "SCRIPT_DIR" not in text:
            result.add(
                Finding(
                    "warning",
                    "missing-scontrol",
                    "sbatch scripts should resolve SCRIPT_DIR via scontrol when SLURM_JOB_ID is set",
                    model=model.slug,
                    file=rel,
                )
            )
        if is_amd and NV_FLAG.search(text):
            result.add(
                Finding(
                    "error",
                    "nv-flag",
                    "AMD Apptainer sbatch scripts must use --rocm, not --nv",
                    model=model.slug,
                    file=rel,
                )
            )
        if is_amd and "apptainer" in text and "--rocm" not in text and "docker" not in name:
            result.add(
                Finding(
                    "warning",
                    "missing-rocm-flag",
                    "Apptainer AMD sbatch script does not mention --rocm",
                    model=model.slug,
                    file=rel,
                )
            )
        if "_mi300" in name or "_mi250" in name or "_mi350" in name:
            result.add(
                Finding(
                    "warning",
                    "gpu-specific-filename",
                    "prefer sbatch_<task>_amd.sh over GPU-specific names",
                    model=model.slug,
                    file=rel,
                )
            )

    try:
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "bash -n failed").strip().splitlines()
            result.add(
                Finding(
                    "error",
                    "bash-syntax",
                    detail[0] if detail else "bash -n failed",
                    model=model.slug,
                    file=rel,
                )
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.add(
            Finding(
                "warning",
                "bash-n-unavailable",
                f"could not run bash -n: {exc}",
                model=model.slug,
                file=rel,
            )
        )

    _shellcheck(model, path, rel, result)

    if name.startswith("run_") and declared_env:
        used = {m.group(1) for m in ENV_ASSIGN.finditer(text)}
        used |= {m.group(2) for m in ENV_ASSIGN.finditer(text)}
        extra = sorted(v for v in used if v not in declared_env and not v.startswith("SLURM"))
        ignore = {
            "PATH",
            "HOME",
            "USER",
            "PWD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "HF_HOME",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "AI4S_SHARED_DIR",
        }
        extra = [v for v in extra if v not in ignore]
        if extra:
            result.add(
                Finding(
                    "warning",
                    "undeclared-env",
                    "run script env vars not listed in model.yaml env_vars: " + ", ".join(extra[:12]),
                    model=model.slug,
                    file=rel,
                )
            )


def _shellcheck(model, path: Path, rel: str, result: Result) -> None:
    """Error-severity shellcheck only (style stays out of CI). Missing binary is one warning."""
    global _SHELLCHECK_MISSING
    if _SHELLCHECK_MISSING:
        return
    try:
        proc = subprocess.run(
            ["shellcheck", "-S", "error", "-f", "gcc", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        _SHELLCHECK_MISSING = True
        result.add(
            Finding(
                "warning",
                "shellcheck-unavailable",
                "shellcheck not installed; CI installs it",
                model=model.slug,
                file=rel,
            )
        )
        return
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.add(
            Finding(
                "warning",
                "shellcheck-unavailable",
                f"could not run shellcheck: {exc}",
                model=model.slug,
                file=rel,
            )
        )
        return
    if proc.returncode == 0:
        return
    for line in (proc.stdout or proc.stderr or "").splitlines():
        line = line.strip()
        if not line:
            continue
        result.add(
            Finding(
                "error",
                "shellcheck",
                line,
                model=model.slug,
                file=rel,
            )
        )
