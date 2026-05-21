---
name: ai4science-perf-analysis
description: Runs the AMD AI agents bottleneck-analysis workflow on a HydraGNN (or other ai4science) training job using TraceLens + Omnistat user-mode + paired analyst/verifier subagents. Use when the user wants to "analyze a 2-node HydraGNN run", "find bottlenecks with TraceLens/Omnistat", or "run the AMD AI agents on a job".
---

# AI4Science perf-analysis (multi-subagent bottleneck workflow)

## When this skill applies

The user wants an automated bottleneck analysis of a multi-node training/inference run using AMD's open-source observability tooling (TraceLens, Omnistat). This is **distinct** from the `ai4science-run-models` skill — that one launches a model; this one diagnoses a model after it runs.

Default target: HydraGNN on Lux MI355X. The same pattern can extend to ORBIT-2 / StormCast / GP-MoLFormer in iteration 2.

## Repository entry points

- Recipe: [material_science/models/HydraGNN/recipes/perf-analysis/](../../../material_science/models/HydraGNN/recipes/perf-analysis/)
- Sbatch wrapper: [material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh](../../../material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh)
- Agent prompt files: `material_science/models/HydraGNN/recipes/perf-analysis/agents/*.md`

## Orchestration loop (what the main agent does)

1. **Read the recipe README** to refresh on the run topology and artifact layout.
2. **Dispatch the launcher subagent** (Task tool, `generalPurpose` or `shell`) with the prompt at `agents/launcher.md`. Wait for the job to complete and the manifest to be written.
3. **Dispatch the two analyst subagents in parallel** (one Task call per subagent in the same message):
   - `tracelens_analyst` — prompt at `agents/tracelens_analyst.md`
   - `omnistat_analyst` — prompt at `agents/omnistat_analyst.md`
4. **Dispatch the two verifier subagents in parallel** once analysts return:
   - `tracelens_verifier`
   - `omnistat_verifier`
5. **Dispatch the synthesizer subagent**, hand it both `verified_claims.json` files.
6. **Print the combined report** to the user with a short executive summary in chat.

## Subagent contract (every subagent must)

- Read **only** the files listed in its `## Inputs` section.
- Produce **only** the file(s) listed in its `## Outputs` section.
- Write a single line `STATUS=ok|partial|fail; reason=<short>` to stdout as the last line.
- Never escalate to multi-node `srun`; verifiers may use **at most one** 1-node `srun -p lux -A vultr_lux -N1 --time=00:05:00` interactive probe.
- Never edit files outside `/shared/aaji/models/HydraGNN/perf-runs/<jobid>/`.

## Cluster constraints (Lux)

- Partition: `lux`. Account: `vultr_lux`. Read from [.cluster-config.yaml](../../../.cluster-config.yaml).
- Central Prometheus at `atlvrmonad01:9090` is **unreachable from compute nodes** (TCP RST, IP allowlist). Do **not** point omnistat-query at it. Always use the user-mode VictoriaMetrics that the launcher started.
- System-mode Omnistat is running on every compute node at port 8001. Don't touch it; we run our own user-mode collector alongside.
- The login node (`rad-vultr-login`) has no GPU and no MPI. All analysis runs after the job — no live profiling on the login node.

## When something goes wrong

- If the launcher times out waiting for `sacct` to report a terminal state, it must `scontrol show job <id>` and write `state=TIMEOUT_WAITING` in the manifest; the orchestrator should surface this to the user and stop.
- If TraceLens or omnistat-inspect can't be installed (network, py-version), the launcher's install step writes a clear error to stderr and exits non-zero — the orchestrator must NOT fall back to "skip analysis", it must surface the error.
- If a verifier refutes an analyst's top claim, it stays in the report (with `verdict=refuted`) so future iterations don't re-derive the same wrong conclusion.

## Lessons captured (smoke-test on JOB 6762)

The first 2-node end-to-end run on Lux exposed five real bugs that are now all worked around in [`sbatch_train_perf_amd.sh`](../../../material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh) and the agent prompts:

1. **`%(SLURM_JOB_ID)s` in omnistat config is broken** — `configparser` doesn't interpolate `os.environ`. Use `@JOB_DIR@` placeholder and `sed` at submit time.
2. **`/tmp/omni_rmsjobinfo` permission collision with system-mode Omnistat** — override `job_detection_file` to a per-job path under `/shared/aaji/...`.
3. **VictoriaMetrics on login node needs `-fs.disableMmap`** to load even a 1.6 MB DB (cgroup mmap restriction).
4. **HydraGNN `Profile` block must be under `NeuralNetwork`**, not at the top level — `train_validate_test()` is called with `config["NeuralNetwork"]` so that's the scope its Profiler reads.
5. **PromQL with `jobid="..."` requires joining via `rmsjob_info`** — `rocm_*` metrics don't carry a `jobid` label directly. Verifier subagents must use:
   ```promql
   metric * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="..."}))
   ```

The full list (with cross-cutting context) is at the end of [.cursor/skills/ai4science-studio/SKILL.md](../ai4science-studio/SKILL.md). When fixing a bug here, check there first and propagate the lesson.
