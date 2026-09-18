from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ai4s_validate.discover import ModelEntry
from ai4s_validate.findings import Finding, Result


def check_preflight_dry_run(root: Path, models: list[ModelEntry]) -> Result:
    """Run each preflight_*.py --dry-run on CPU. Failures are errors."""
    result = Result()
    for model in models:
        examples = root / model.path / "examples"
        if not examples.is_dir():
            continue
        for pf in sorted(examples.glob("preflight_*.py")):
            rel = str(pf.relative_to(root))
            try:
                proc = subprocess.run(
                    [sys.executable, str(pf), "--dry-run"],
                    cwd=str(examples),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result.add(
                    Finding(
                        "error",
                        "preflight-timeout",
                        "preflight --dry-run timed out",
                        model=model.slug,
                        file=rel,
                    )
                )
                continue
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                result.add(
                    Finding(
                        "error",
                        "preflight-dry-run",
                        (err[-1] if err else f"exit {proc.returncode}"),
                        model=model.slug,
                        file=rel,
                    )
                )
                continue
            combined = (proc.stdout or "") + (proc.stderr or "")
            if "dry-run" not in combined.lower():
                result.add(
                    Finding(
                        "error",
                        "preflight-dry-run-silent",
                        "preflight --dry-run must print that it skipped GPU/import checks",
                        model=model.slug,
                        file=rel,
                    )
                )
            body = pf.read_text(encoding="utf-8", errors="replace")
            if len(body.strip()) < 400:
                result.add(
                    Finding(
                        "error",
                        "preflight-stub",
                        "preflight is only a --dry-run stub; add real GPU/import checks for live use",
                        model=model.slug,
                        file=rel,
                    )
                )
    return result
