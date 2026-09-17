# Agentic evals (Tier 3) — scaffold

This directory is the future home of **headless agent evals**. It is not wired to CI yet.

The grading principle: **prefer deterministic assertions over LLM judges**.

The repo already has an importable grader: `tools/ai4s_validate` returns `Finding` objects. An eval case should produce a git worktree (or patch) and ask:

- Does `run_all(root)` report zero errors?
- Did the agent read `<model>/model.yaml` before writing scripts? (trace assertion, once a runner exists)
- Does the emitted sbatch command contain `--rocm` and not `--nv`?

LLM-as-judge is reserved for prose quality (recipe README clarity), not for “did the agent follow the contract?”

## Case format

See [`cases/`](cases/). Each YAML file is one case:

```yaml
id: run-orbit2-inference
prompt: |
  Run ORBIT-2 inference as a 1-node SLURM job using the repo recipes.
assertions:
  - type: validate_clean          # ai4s_validate.run_fast / run_all
  - type: file_exists
    path: earth_science/models/ORBIT-2/model.yaml
  - type: command_contains
    needle: --rocm
  - type: command_excludes
    needle: --nv
```

## Runner (not implemented)

Deferred until a headless driver is chosen (Cursor SDK / `cursor-agent`). When added, it should:

1. Check out a clean worktree at a pinned SHA.
2. Run the agent with the case prompt and a budget (steps / dollars).
3. Grade with the assertions above.
4. Write `evals/results/<case>-<timestamp>.json` (gitignored).

Do not add a GPU smoke-test runner here; that is Tier 2 and belongs in a self-hosted workflow gated on a `run-gpu` label.
