# perf-report playbook

Read-only agent. Inputs come from a prior instrumented science stage.

1. Resolve `job_dir` from prior artifacts (`manifest.json` parent) or inputs.
2. Run deterministic checks → candidate findings with evidence paths.
3. If an LLM endpoint is available (external or same-allocation local), rank/explain.
   Drop any finding that invents evidence paths.
4. Write `findings.json` and `perf_report.md`.
5. On LLM failure: keep deterministic findings, mark synthesis failed / status partial.

Never submit or cancel SLURM jobs. Never edit workload code.
