# ai4s_validate

Importable checks for AI4Science Studio. Used by `make check`, pre-commit, GitHub Actions, and (later) `evals/`.

```bash
PYTHONPATH=tools python3 -m ai4s_validate check        # full
PYTHONPATH=tools python3 -m ai4s_validate check-fast   # schema + models.yaml
PYTHONPATH=tools python3 -m ai4s_validate fix          # regenerate models.yaml
PYTHONPATH=tools python3 -m ai4s_validate check --json
```

`run_all()` / `run_fast()` return a `Result` of `Finding` dataclasses — do not scrape stdout from evals.

Full `check` also runs `bash -n`, **error-severity `shellcheck`** (style codes ignored), `container_image` syntax (no registry pull), and `preflight_*.py --dry-run` (must print that it skipped GPU checks and must not be a stub).
