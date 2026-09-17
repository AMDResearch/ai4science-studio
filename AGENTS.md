# Agent entry point (Cursor, Claude Code, Codex, and similar)

AI4Science Studio is an **agent-first** recipe collection. There is no application build for the science models themselves — the contract is YAML manifests plus scripts.

## Read first

1. Root [`models.yaml`](models.yaml) — **generated** index of every model. Do not edit by hand; change `<domain>/models/<slug>/model.yaml` and run `make fix`.
2. `<domain>/models/<slug>/model.yaml` — source of truth (HF id, license, recipes, env vars, hardware).
3. This file, then domain skills under `.cursor/skills/`.

## Layout

```
models.yaml
<domain>/models/<slug>/{model.yaml,README.md,recipes/,examples/}
```

Domains: `earth_science/`, `material_science/`, `protein_folding/`, `healthcare/`, `physics_simulation/`.

`protein_folding/` is an empty placeholder until a model is added — keep the domain.

## Validation (run this after any manifest or examples/ change)

```bash
make check        # schema, generated index, scripts, docs, preflight --dry-run
make check-fast   # schema + index only (pre-commit)
make fix          # regenerate models.yaml from */models/*/model.yaml
```

Validators live in `tools/ai4s_validate/` and return structured findings (importable). Future agent evals under `evals/` should grade by calling the same library, not by scraping logs.

## Adding a model

1. Copy `_template/` to `<domain>/models/<slug>/`.
2. Fill `README.md` and `model.yaml` (include a short `summary` for the generated index).
3. Add recipes and `examples/` scripts (`docker_run.sh`, `run_*`, `preflight_*.py` with `--dry-run`, `sbatch_*_amd.sh`).
4. Run `make fix && make check`.
5. Do not commit checkpoints, datasets, or secrets.

## Conventions

- Hugging Face id `org/model` → directory `org__model`, unless a public name is documented in the model README.
- **No site-specific paths, partitions, node lists, or job IDs** in committed `examples/` or `recipes/`. `make check` flags `/shared/<user>`, `/home/<user>`, `#SBATCH --partition=` that is not `YOUR_*`, and strings that look like recorded SLURM job ids. Site values belong in gitignored `.cluster-config.yaml`.
- Healthcare content is research/engineering only — no PHI, no clinical claims.
- Capture lessons in `.cursor/skills/` (and a rule if the mistake is procedural) in the same pass as the fix.

## Skills and commands

- Cursor: `.cursor/skills/`
- Claude Code: `.claude/commands/` plus [`CLAUDE.md`](CLAUDE.md) (pointer to this file)
