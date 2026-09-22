"""Importable validators for AI4Science Studio manifests, index, and scripts.

Callers (CLI, pre-commit, CI, and later evals/) should use `run_all` / `run_fast`
and inspect `Finding` objects rather than scraping stdout.
"""

from __future__ import annotations

from ai4s_validate.findings import Finding, Result
from ai4s_validate.runner import run_all, run_fast

__all__ = ["Finding", "Result", "run_all", "run_fast"]
