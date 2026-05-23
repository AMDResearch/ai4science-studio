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
- **Node-specific mount faults are real and silent.** Any single compute node can land an allocation with a broken per-user autofs/NFS mount of `/home/$USER` or `/shared/$USER`, while sibling nodes in the same allocation are fine. Symptom in slurmd logs on the broken node: `Home Directory for <user> Not Found ... Please contact system admin` and `lstat /shared/...: no such file or directory`; the surviving nodes wedge in a `pmix_coll_ring` collective fence until SLURM kills the job at the wall.
  - **Confirmed bad nodes** (as of 2026-05-23, until cluster admin repairs them):
    - `lux-mi355x-a6` — `/home/aaji` missing, `/shared/aaji/images/*.sif` unreadable; sibling `lux-mi355x-a1` healthy in the same alloc. Loop-43b33ec1 iter-1 / job 7088.
    - `lux-mi355x-a10` — pmix ring collective times out at MPI init; `opal_mpi_init_ext3x_client_unreachable`. NOT a mount fault — broken PMIx/OpenMPI install on this node. Loop-c3e4df1c iter-1 / job 7161.
  - **Diagnostic rule:** never blacklist a node *class* (e.g. "avoid all a-nodes") based on a single failure. Always probe the *specific* allocated nodes and exclude only those that fail by name. Probe is a one-task-per-node srun that checks `/home/$USER`, `/shared/$USER`, and the SIF path; see the `Per-node mount-health probe` section of `sbatch_train_perf_amd.sh` (exit 42 on fail). The perf-optimizer-loop orchestrator handles exit 42 by re-submitting with `--exclude=<bad-node>` and adding the node to `loop-<uuid>/known_bad_nodes.txt`; the lever is NOT penalized.

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

## Lessons captured (perf-optimizer-loop pilot, loop-43b33ec1, 2026-05-22)

1. **Diagnose broken nodes individually, not by class** (see Cluster constraints above for full detail). The optimizer-loop's iter-1 wrongly concluded "a-nodes lack /home" from a single failed alloc — actually only `lux-mi355x-a6` was broken. The fix is a per-job mount-health probe (now in `sbatch_train_perf_amd.sh`, exit code 42) plus orchestrator step 2e-bis: on exit 42, parse `node_health_probe.txt`, append the bad node(s) to `loop-<uuid>/known_bad_nodes.txt`, and re-submit with `--exclude=<list>`. Lever is untouched.
2. **HydraGNN `torch.compile` hooks must wrap the model object, not a module path.** `import hydragnn.train.train` (the path the iter-4 hook tried) does not exist in the installed package. Wrap the constructed model directly: `model = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=False)` at the rank-script level, by patching the entrypoint script (`gfm_mlip_all_mpnn.py`) immediately after model construction. **(Now superseded by Lesson #6 below — torch.compile is BLOCKED for HydraGNN MLIP regardless of where it wraps.)**
3. **`parse_convergence.py` over-counts epochs by N_ranks** because each rank emits a `tqdm=100%` line and the parser counts every one. Always read epoch wall-time from rank-0 tqdm directly (`s/it × max_num_batch`), not from the parser's epoch count. Fixed in fom_extractor; long-term fix is to filter on `rank=0` lines in the parser.
4. **MFMA TFLOPS methodology must be declared explicitly.** Burst-rate via PromQL `rate([10s])×4` (iter-0/3 values ≈ 0.006-0.012 TFLOPS) and time-averaged `total_accumulated_ops / duration` (iter-4 value ≈ 0.75 TFLOPS) differ by ~60× for this workload and **must not be compared**. fom_extractor should standardize on time-averaged across the full job window, with the burst-rate as a separate `*_peak_burst` field.
5. **`epoch_time_s` is the correct primary FOM** for HydraGNN-style latency-bound GNN workloads. Throughput (samples/s) misleads when batch size is a lever, because epoch wall time scales with batches-per-epoch, not just per-sample work.

## Lessons captured (loop-c3e4df1c + compile-investigation-aa480554, 2026-05-22..23)

6. **`torch.compile` is BLOCKED for HydraGNN GFM-MLIP MACE** under PyTorch 2.10 / ROCm 7.2.2. Six hooks across three hypotheses (inner-model compile, outer + `dynamo.allow_in_graph(autograd.grad)`, single-submodule compile) all fail with the same verbatim PyTorch error: `RuntimeError: torch.compile with aot_autograd does not currently support double backward` (`torch/_functorch/_aot_autograd/runtime_wrappers.py:2356`). Root cause: HydraGNN's `energy_force_loss` (`hydragnn/models/create.py:718`) calls `torch.autograd.grad(graph_energy_pred, data.pos, create_graph=True)` to compute forces as derivatives of energy w.r.t. positions; with `create_graph=True`, training requires DOUBLE BACKWARD (loss.backward differentiates through forces). Every torch.compile mode (`default`, `reduce-overhead`, `max-autotune`) goes through AOT Autograd which pre-compiles the backward as a joint program and cannot itself be differentiated. **To unblock requires upstream HydraGNN source change**: move `energy_force_loss` INTO `EnhancedModelWrapper.forward()` so `autograd.grad` is inside the compiled forward (the canonical MACE/Allegro/NequIP pattern). ~30-50 LOC across `create.py:622-758` + `train_validate_test.py:725-735`, config-gated. The lever `torch_compile_e3nn` is marked `status: blocked` in `lever_catalog.yaml` with full evidence; the lever_picker drops blocked levers from its candidate set.
7. **`dynamo.allow_in_graph(autograd.grad)` is weaker than MACE upstream docs suggest.** It only prevents dynamo from TRACING THROUGH the call; it does NOT prevent dynamo's FakeTensor symbolic-execution pass from running `autograd.grad` against fake tensors (which fails with the `allow_unused=True` error because fake tensors have no real autograd graph). The MACE `prepare()` pattern works only when `autograd.grad` lives inside the compiled function's `forward()` — not when it's in a separate method called from outside.
8. **TunableOp live tuning (`PYTORCH_TUNABLEOP_TUNING=1`) is unsafe on MI355X / ROCm 7.2.2 with HydraGNN.** All 16 GPUs hit `Memory access fault by GPU node-N` during the hipBLASLt kernel autotuning phase (loop-c3e4df1c iter-3 / job 7188). Use a 2-phase pattern instead: dedicated warmup sbatch with `PYTORCH_TUNABLEOP_TUNING=1` to generate `tunableop_results_<N>.csv` files, then production runs with `PYTORCH_TUNABLEOP_ENABLED=1` only (no `_TUNING=1`). The `tunable_op_warmup_then_use` lever in the catalog encodes this. The original `tunable_op_live` lever is now marked `status: blocked`.
9. **Lever payoff is node-class-dependent.** `num_workers_12` improved a-nodes (-12.4% in loop-43b33ec1) but regressed b-nodes (+10.7% in loop-c3e4df1c). Lever_picker must read `SLURM_NODELIST` from the current best run's manifest and treat a-nodes vs b-nodes as separate state for at least `num_workers_*` and `batch_*` levers.
10. **b-nodes are ~17% faster baseline than a-nodes for HydraGNN.** Loop-43b33ec1's a-node baseline was 92.5 s/epoch; loop-c3e4df1c's b-node baseline (same code, same config, same fp64, same batch_size=400) was 76.0 s/epoch. The single biggest finding of the night came from a baseline shift, not a lever. The `launcher` subagent should report which node class was allocated, and cross-loop FOM comparisons must control for this.
11. **`hydragnn/train/__init__.py` re-export shadow** breaks naive `import hydragnn.train.train_validate_test as tvt`. The line `from .train_validate_test import train_validate_test, train, ...` makes `hydragnn.train.train_validate_test` resolve to the **function** `train_validate_test` (re-exported as attribute of the package), not the module. Use `importlib.import_module("hydragnn.train.train_validate_test")` instead. Applies to any rank hook that needs to monkey-patch `train_validate_test.train`.
12. **`sbatch --export=ALL,...` does NOT include shell vars that are not in `--export`'s explicit list.** When using `--export=ALL,K1=V1,K2=V2`, vars set in the calling shell but not listed here are propagated only if they're in the user's persistent env (login shell init). Always explicitly list cluster-pathing vars like `AI4S_SHARED_DIR=/shared/aaji` in `--export`.

The full list (with cross-cutting context) is at the end of [.cursor/skills/ai4science-studio/SKILL.md](../ai4science-studio/SKILL.md). When fixing a bug here, check there first and propagate the lesson.
