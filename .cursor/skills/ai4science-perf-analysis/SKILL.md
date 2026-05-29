---
name: ai4science-perf-analysis
description: Runs the AMD AI agents bottleneck-analysis workflow on a HydraGNN (or other ai4science) training job using TraceLens + Omnistat user-mode + paired analyst/verifier subagents. Use when the user wants to "analyze a 2-node HydraGNN run", "find bottlenecks with TraceLens/Omnistat", or "run the AMD AI agents on a job".
---

# AI4Science perf-analysis (multi-subagent bottleneck workflow)

## When this skill applies

The user wants an automated bottleneck analysis of a multi-node training/inference run using AMD's open-source observability tooling (TraceLens, Omnistat). This is **distinct** from the `ai4science-run-models` skill — that one launches a model; this one diagnoses a model after it runs.

Default target: HydraGNN on AMD MI355X. The same pattern can extend to ORBIT-2 / StormCast / GP-MoLFormer in iteration 2.

## Repository entry points

- Recipe: [material_science/models/HydraGNN/recipes/perf-analysis/](../../../material_science/models/HydraGNN/recipes/perf-analysis/)
- Iterative-loop recipe: [material_science/models/HydraGNN/recipes/perf-optimizer-loop/](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/) — **see [`HANDOFF.md`](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/HANDOFF.md) for current state, what works, what's blocked, and next-work priorities**
- Sbatch wrapper: [material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh](../../../material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh)
- Reusable node-health probe: [material_science/models/HydraGNN/examples/microbench_node_health.sh](../../../material_science/models/HydraGNN/examples/microbench_node_health.sh)
- Agent prompt files: `material_science/models/HydraGNN/recipes/perf-analysis/agents/*.md` and `material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/*.md`

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
- Never escalate to multi-node `srun`; verifiers may use **at most one** 1-node `srun -N1 --time=00:05:00` interactive probe (partition/account from `.cluster-config.yaml`).
- Never edit files outside `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/<jobid>/`.

## Cluster constraints (AMD Instinct MI355X)

- Partition / account: read from [.cluster-config.yaml](../../../.cluster-config.yaml).
- Central Prometheus may be **unreachable from compute nodes** (check your cluster's network policy). Always use the user-mode VictoriaMetrics that the launcher started.
- System-mode Omnistat may be running on every compute node at port 8001. Don't touch it; we run our own user-mode collector alongside.
- The login node typically has no GPU and no MPI. All analysis runs after the job — no live profiling on the login node.
- **Node-specific mount faults are real and silent.** Any single compute node can land an allocation with a broken per-user autofs/NFS mount of `/home/$USER` or `$AI4S_SHARED_DIR`, while sibling nodes in the same allocation are fine. Symptom in slurmd logs on the broken node: `Home Directory for <user> Not Found ... Please contact system admin` and `lstat ...: no such file or directory`; the surviving nodes wedge in a `pmix_coll_ring` collective fence until SLURM kills the job at the wall.
  - **Confirmed bad node patterns we hit (MI355X, May 2026):** intermittent dual-NUMA STREAM degradation on one node, broken container bind-mounts (`$HOME`/`$AI4S_SHARED_DIR`) on another, PMIx ring timeout on a third. Node names are cluster-specific; see the untracked microbench outputs for hostnames.
  - **Diagnostic rule:** never blacklist a node *class* based on a single failure. Always probe the *specific* allocated nodes and exclude only those that fail by name. Probe is a one-task-per-node srun that checks `/home/$USER`, `$AI4S_SHARED_DIR`, and the SIF path; see the `Per-node mount-health probe` section of `sbatch_train_perf_amd.sh` (exit 42 on fail). The perf-optimizer-loop orchestrator handles exit 42 by re-submitting with `--exclude=<bad-node>` and adding the node to `loop-<uuid>/known_bad_nodes.txt`; the lever is NOT penalized.

## When something goes wrong

- If the launcher times out waiting for `sacct` to report a terminal state, it must `scontrol show job <id>` and write `state=TIMEOUT_WAITING` in the manifest; the orchestrator should surface this to the user and stop.
- If TraceLens or omnistat-inspect can't be installed (network, py-version), the launcher's install step writes a clear error to stderr and exits non-zero — the orchestrator must NOT fall back to "skip analysis", it must surface the error.
- If a verifier refutes an analyst's top claim, it stays in the report (with `verdict=refuted`) so future iterations don't re-derive the same wrong conclusion.

## Lessons captured (smoke-test on JOB 6762)

The first 2-node end-to-end run exposed five real bugs that are now all worked around in [`sbatch_train_perf_amd.sh`](../../../material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh) and the agent prompts:

1. **`%(SLURM_JOB_ID)s` in omnistat config is broken** — `configparser` doesn't interpolate `os.environ`. Use `@JOB_DIR@` placeholder and `sed` at submit time.
2. **`/tmp/omni_rmsjobinfo` permission collision with system-mode Omnistat** — override `job_detection_file` to a per-job path under `$AI4S_SHARED_DIR/...`.
3. **VictoriaMetrics on login node needs `-fs.disableMmap`** to load even a 1.6 MB DB (cgroup mmap restriction).
4. **HydraGNN `Profile` block must be under `NeuralNetwork`**, not at the top level — `train_validate_test()` is called with `config["NeuralNetwork"]` so that's the scope its Profiler reads.
5. **PromQL with `jobid="..."` requires joining via `rmsjob_info`** — `rocm_*` metrics don't carry a `jobid` label directly. Verifier subagents must use:
   ```promql
   metric * on (instance) group_left() (max by (instance) (rmsjob_info{jobid="..."}))
   ```

## Lessons captured (perf-optimizer-loop pilot, loop-43b33ec1, 2026-05-22)

1. **Diagnose broken nodes individually, not by class** (see Cluster constraints above for full detail). The optimizer-loop's iter-1 wrongly concluded "a-class nodes lack /home" from a single failed alloc — only one specific node was broken. The fix is a per-job mount-health probe (now in `sbatch_train_perf_amd.sh`, exit code 42) plus orchestrator step 2e-bis: on exit 42, parse `node_health_probe.txt`, append the bad node(s) to `loop-<uuid>/known_bad_nodes.txt`, and re-submit with `--exclude=<list>`. Lever is untouched.
2. **HydraGNN `torch.compile` hooks must wrap the model object, not a module path.** `import hydragnn.train.train` (the path the iter-4 hook tried) does not exist in the installed package. Wrap the constructed model directly: `model = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=False)` at the rank-script level, by patching the entrypoint script (`gfm_mlip_all_mpnn.py`) immediately after model construction. **(Now superseded by Lesson #6 below — torch.compile is BLOCKED for HydraGNN MLIP regardless of where it wraps.)**
3. **`parse_convergence.py` over-counts epochs by N_ranks** because each rank emits a `tqdm=100%` line and the parser counts every one. Always read epoch wall-time from rank-0 tqdm directly (`s/it × max_num_batch`), not from the parser's epoch count. Fixed in fom_extractor; long-term fix is to filter on `rank=0` lines in the parser.
4. **MFMA TFLOPS methodology must be declared explicitly.** Burst-rate via PromQL `rate([10s])×4` (iter-0/3 values ≈ 0.006-0.012 TFLOPS) and time-averaged `total_accumulated_ops / duration` (iter-4 value ≈ 0.75 TFLOPS) differ by ~60× for this workload and **must not be compared**. fom_extractor should standardize on time-averaged across the full job window, with the burst-rate as a separate `*_peak_burst` field.
5. **`epoch_time_s` is the correct primary FOM** for HydraGNN-style latency-bound GNN workloads. Throughput (samples/s) misleads when batch size is a lever, because epoch wall time scales with batches-per-epoch, not just per-sample work.

## Lessons captured (loop-c3e4df1c + compile-investigation-aa480554, 2026-05-22..23)

6. **`torch.compile` is BLOCKED for HydraGNN GFM-MLIP MACE** under PyTorch 2.10 / ROCm 7.2.2. Six hooks across three hypotheses (inner-model compile, outer + `dynamo.allow_in_graph(autograd.grad)`, single-submodule compile) all fail with the same verbatim PyTorch error: `RuntimeError: torch.compile with aot_autograd does not currently support double backward` (`torch/_functorch/_aot_autograd/runtime_wrappers.py:2356`). Root cause: HydraGNN's `energy_force_loss` (`hydragnn/models/create.py:718`) calls `torch.autograd.grad(graph_energy_pred, data.pos, create_graph=True)` to compute forces as derivatives of energy w.r.t. positions; with `create_graph=True`, training requires DOUBLE BACKWARD (loss.backward differentiates through forces). Every torch.compile mode (`default`, `reduce-overhead`, `max-autotune`) goes through AOT Autograd which pre-compiles the backward as a joint program and cannot itself be differentiated. **To unblock requires upstream HydraGNN source change**: move `energy_force_loss` INTO `EnhancedModelWrapper.forward()` so `autograd.grad` is inside the compiled forward (the canonical MACE/Allegro/NequIP pattern). ~30-50 LOC across `create.py:622-758` + `train_validate_test.py:725-735`, config-gated. The lever `torch_compile_e3nn` is marked `status: blocked` in `lever_catalog.yaml` with full evidence; the lever_picker drops blocked levers from its candidate set.
7. **`dynamo.allow_in_graph(autograd.grad)` is weaker than MACE upstream docs suggest.** It only prevents dynamo from TRACING THROUGH the call; it does NOT prevent dynamo's FakeTensor symbolic-execution pass from running `autograd.grad` against fake tensors (which fails with the `allow_unused=True` error because fake tensors have no real autograd graph). The MACE `prepare()` pattern works only when `autograd.grad` lives inside the compiled function's `forward()` — not when it's in a separate method called from outside.
8. **TunableOp live tuning (`PYTORCH_TUNABLEOP_TUNING=1`) is unsafe on MI355X / ROCm 7.2.2 with HydraGNN.** All 16 GPUs hit `Memory access fault by GPU node-N` during the hipBLASLt kernel autotuning phase (loop-c3e4df1c iter-3 / job 7188). Use a 2-phase pattern instead: dedicated warmup sbatch with `PYTORCH_TUNABLEOP_TUNING=1` to generate `tunableop_results_<N>.csv` files, then production runs with `PYTORCH_TUNABLEOP_ENABLED=1` only (no `_TUNING=1`). The `tunable_op_warmup_then_use` lever in the catalog encodes this. The original `tunable_op_live` lever is now marked `status: blocked`.
9. **Lever payoff can be run-to-run noise, not node-class signal.** `num_workers_12` improved one run (-12.4%) and regressed another (+10.7%), both on the same node pair on different days. The lever picker should treat `num_workers_*` results as high-variance until replicated within ±5% on the same node pair.
10. **Always read `nodes_list` from `iter-0-baseline.json` before drawing any class conclusion from a FOM delta.** Two baselines that appear to "use different node classes" can in fact be the same physical nodes scheduled by SLURM on different days — especially when one iter-1 failed and the successful iter-0 landed wherever SLURM placed it.
11. **`hydragnn/train/__init__.py` re-export shadow** breaks naive `import hydragnn.train.train_validate_test as tvt`. The line `from .train_validate_test import train_validate_test, train, ...` makes `hydragnn.train.train_validate_test` resolve to the **function** `train_validate_test` (re-exported as attribute of the package), not the module. Use `importlib.import_module("hydragnn.train.train_validate_test")` instead. Applies to any rank hook that needs to monkey-patch `train_validate_test.train`.
12. **`sbatch --export=ALL,...` does NOT include shell vars that are not in `--export`'s explicit list.** When using `--export=ALL,K1=V1,K2=V2`, vars set in the calling shell but not listed here are propagated only if they're in the user's persistent env (login shell init). Always explicitly list cluster-pathing vars like `AI4S_SHARED_DIR` in `--export`.

## Lessons captured (pathc-pathb-a3b8f3f0 + loop-e3fac2af, 2026-05-23)

13. **fp32-vs-fp64 epoch-time comparison is the canonical low-cost dispatch-vs-compute discriminator.** On gfx950 (MI355X), the fp32 MFMA peak is **2× the fp64 MFMA peak**. A compute-bound workload should show ≥30% speedup when switching fp64→fp32. HydraGNN GFM-MLIP MACE showed ~0% speedup (78→79 s, +1.3% within noise, fp32 verified by `precision-diagnostic` rank-0 logs `model_param_dtype=torch.float32 first_batch_float_dtype=torch.float32`). This proves the workload is dispatch-bound. Run this comparison ONCE per workload-class as a single sbatch; the verdict shapes the entire subsequent lever strategy:
    - **fp32 ≈ fp64 → dispatch-bound** → compute-side levers (`batch_*`, `torch_compile_*`, `tunable_op_*`, `precision_fp32`) are dead ends. Focus on kernel-trace analysis to identify the dispatch culprit, then either upstream kernel fusion or reduce launch count (larger blocks, mega-kernels).
    - **fp32 ≪ fp64 → compute-bound** → standard compute-side levers apply. fp32 itself becomes a real lever (within numerical tolerance).
    - **fp32 > fp64 → memory-bound** → bandwidth-targeting levers (batch packing, fewer kernel launches over more data) apply.
14. **`torch.jit.script` is ALSO blocked for HydraGNN MLIP MACE** (in addition to `torch.compile`). Root cause: `MACEStack.py:397` uses `**conv_args` keyword-arg expansion when calling the interaction block; TorchScript JIT does not support `**kwargs` expansion (`NotSupportedError: keyword-arg expansion is not supported`). Despite the upstream `@compile_mode("script")` decorator from e3nn, the actual call site is not scriptable. **Both PyTorch compiler paths are now definitively dead** for HydraGNN MLIP MACE as shipped — unblocking either requires upstream source changes:
    - `torch.compile` path: move `energy_force_loss` into `EnhancedModelWrapper.forward()` (~30-50 LOC; addresses AOT-double-backward; see lesson #6)
    - `torch.jit.script` path: unpack `**conv_args` at `MACEStack.py:397` (small diff but in the read-only `/opt/hydragnn-pkgs` overlay)
15. **TunableOp `_TUNING=1` is unsafe regardless of phase** on MI355X / ROCm 7.2.2 / PyTorch 2.10. Both the live-tuning lever (`tunable_op_live`, loop-c3e4df1c iter-3) and the warmup-then-use 2-phase pattern (`tunable_op_warmup_then_use`, loop-e3fac2af iter-2) hit the same `Memory access fault by GPU node-N` on all 16 GPUs during the hipBLASLt tuning routine. The 2-phase split doesn't help — the fault is in the tuning routine itself, not in the production consumption. **Both levers are marked `status: blocked` in the catalog.** Re-test only after a ROCm stack update with a documented hipBLASLt fix.
16. **The "dispatch ceiling" conclusion is solid, but do NOT frame it as node-class-specific.** All three loop baselines ran on the same physical node pair; every compute-side lever tested (batch_800, num_workers_12, precision_fp32) regressed or was neutral. That confirms dispatch-bound character for this workload — but makes no statement about node classes. The microbench (lesson #18) later showed all tested nodes have identical HIP launch latency, so the ceiling is workload-driven, not hardware-class-driven.
17. **Cross-loop best is composed of a free finding + a small lever**: current best (75.0 s/epoch) = baseline-on-b1-b2 (76.0 s) + `rccl_high_priority` (-1.3%). The single accepted lever across 3 loops + 1 investigation is `rccl_high_priority` (`TORCH_NCCL_HIGH_PRIORITY=1 GPU_MAX_HW_QUEUES=2`); it should be baked into the baseline contract for HydraGNN GFM-MLIP going forward (now marked `status: accepted-baked-in` in the catalog).

## Lessons captured (a-vs-b parity microbench, 2026-05-26)

18. **A "node-class performance gap" narrative was debunked by a controlled microbench.** A 4-test microbench sweep (script: [`examples/microbench_node_health.sh`](../../../material_science/models/HydraGNN/examples/microbench_node_health.sh); raw outputs in untracked `$AI4S_SHARED_DIR/microbench-a-vs-b/`) found:
    - All tested nodes had identical HW at the host level (same CPU model, NUMA layout, GPUs, NICs, firmware).
    - HIP empty-kernel launch latency 3.5–3.8 µs/launch on every GPU of every node; NCCL single-node all-reduce and STREAM single-NUMA TRIAD were statistically equivalent across all nodes tested (one outlier node had degraded dual-NUMA STREAM COPY, flagged for sysadmin).
    - A controlled 2-node sanity HydraGNN run using nodes from each "class" showed < 2% wall-time difference, well inside noise.
    - **Root cause of the original gap narrative:** all three baseline loop runs landed on the same physical node pair. The 92.5 s → 76 s spread was run-to-run variance on the *same pair* on different days. The misattribution arose from labeling one loop as "class A" because its iter-1 failed on a broken-mount node — but the *successful* baseline iter-0 landed on a different pair via SLURM scheduling.
    - **Action for future loops:** always `jq .nodes_list $LOOP_DIR/iter-0-baseline.json` before drawing any class conclusion. The `launcher` and `synthesizer` subagents should log the assigned nodelist front-and-center in their output.
19. **Per-node health regressions are real and worth a 30-second prologue probe.** In 2 weeks we hit three different per-node failure modes: container bind-mount EIO on one node, PMIx/NCCL ring timeout on another, dual-NUMA bandwidth degradation on a third. Each manifested as a confusing application-level failure hours into a job. All three would have been caught at job start by a 30-s SLURM Prolog running [`microbench_node_health.sh`](../../../material_science/models/HydraGNN/examples/microbench_node_health.sh) (mount-write probe + STREAM dual-NUMA + NCCL single-node + HIP launch latency). Recommend adding this as a SLURM Prolog to your cluster admin.

## Lessons captured (attribution pass on job 7187, 2026-05-27)

20. **Attribution pass tooling:** Use [`examples/run_fom_extractor.py`](../../../material_science/models/HydraGNN/examples/run_fom_extractor.py) + TraceLens on `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/<jobid>/`. Backfill `manifest.json` if missing (loop jobs before launcher wrote it). **VictoriaMetrics PromQL on perf-run DBs requires `time=<unix>` at the job window** — instant queries on the login node return empty series even when `omnistat_hardware_counter` data exists. **`kernel_correlation_summary.attribution_quality=poor` on a ~6 s profile-epoch kineto slice is often a windowing artifact** (busy_frac &lt;0.5 in every 1 s bucket); trust TraceLens `ops_summary.csv` (`aten::mm` + `aten::bmm` ≈80% kernel time) before escalating to `OMNISTAT_KERNEL_TRACE=1`. See [`recipes/perf-optimizer-loop/dispatch-attribution.md`](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/dispatch-attribution.md).

The full list (with cross-cutting context) is at the end of [.cursor/skills/ai4science-studio/SKILL.md](../ai4science-studio/SKILL.md). When fixing a bug here, check there first and propagate the lesson.
