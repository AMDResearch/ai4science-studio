---
name: ai4science-perf-analysis
description: Runs the AMD AI agents bottleneck-analysis workflow on a HydraGNN or ORBIT-2 training job using TraceLens + Omnistat user-mode + paired analyst/verifier subagents, or drives the iterative perf-optimizer-loop (accept/revert on throughput or epoch time). Use when the user wants to analyze a perf run, analyze a 2-node HydraGNN run, find bottlenecks with TraceLens/Omnistat, run the AMD AI agents on a job, run the perf-optimizer-loop, tune ORBIT-2 throughput, or optimize HydraGNN training performance.
---

# AI4Science perf-analysis (multi-subagent bottleneck workflow)

## When this skill applies

The user wants an automated bottleneck analysis of a multi-node training/inference run using AMD's open-source observability tooling (TraceLens, Omnistat). This is **distinct** from the `ai4science-run-models` skill — that one launches a model; this one diagnoses a model after it runs.

Default target: HydraGNN on AMD MI355X. **ORBIT-2 training** uses the same perf-analysis pattern; see `earth_science/models/ORBIT-2/recipes/perf-analysis/`. **ORBIT-2 iterative sysopt** (throughput-primary FOM) lives in `earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/` (`run_optimizer_loop.sh`, `lever_catalog.yaml`).

## Repository entry points

### HydraGNN

- Recipe: [material_science/models/HydraGNN/recipes/perf-analysis/](../../../material_science/models/HydraGNN/recipes/perf-analysis/)
- Iterative-loop recipe: [material_science/models/HydraGNN/recipes/perf-optimizer-loop/](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/) — **see [`dispatch-attribution.md`](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/dispatch-attribution.md) (current-best attribution) and [`lever_catalog.yaml`](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/lever_catalog.yaml) (levers tried / blocked with evidence)**
- Sbatch wrapper: [material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh](../../../material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh)
- Reusable node-health probe: [material_science/models/HydraGNN/examples/microbench_node_health.sh](../../../material_science/models/HydraGNN/examples/microbench_node_health.sh)
- Agent prompt files: `material_science/models/HydraGNN/recipes/perf-analysis/agents/*.md` and `material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/*.md`

### ORBIT-2

- Recipe: [earth_science/models/ORBIT-2/recipes/perf-analysis/](../../../earth_science/models/ORBIT-2/recipes/perf-analysis/) — see [`README.md`](../../../earth_science/models/ORBIT-2/recipes/perf-analysis/README.md) (FOM contract, defaults, landmines) and [`gemm-attribution.md`](../../../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/gemm-attribution.md) (GEMM-time finding + levers)
- **Iterative sysopt loop:** [earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/](../../../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/) — primary accept/revert FOM **`throughput_samples_per_s`** (`run_fom_extractor.py` + `manifest.global_batch_size`)
- Sbatch: [earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh](../../../earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh)
- Plain training: [earth_science/models/ORBIT-2/examples/sbatch_train_amd.sh](../../../earth_science/models/ORBIT-2/examples/sbatch_train_amd.sh)
- Scaling: [run_scaling_study.sh](../../../earth_science/models/ORBIT-2/examples/run_scaling_study.sh) + [collate_scaling_study.py](../../../earth_science/models/ORBIT-2/examples/collate_scaling_study.py)
- FOM parser: [parse_training_log.py](../../../earth_science/models/ORBIT-2/examples/parse_training_log.py) — latency **`steady_batch_time_s`**; throughput when `global_batch_size` is known; **`loss_sanity_pass`** requires strictly decreasing epoch losses
- FOM extractor: [run_fom_extractor.py](../../../earth_science/models/ORBIT-2/examples/run_fom_extractor.py) — writes `foms.json`; optional PromQL via `ORBIT2_TSDB_URL`; omnistat template defaults to **bf16 MFMA** profile `hbm_flops_bf16` (`FETCH_SIZE` + `SQ_INSTS_VALU_MFMA_MOPS_BF16`); set **`OMNISTAT_ROCPROF_PROFILE=hbm_flops_f64`** to revert to fp64 HydraGNN-style counters
- Artifacts: `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/`
- Workload: `intermediate_downscaling.py` via [run_orbit2_train.py](../../../earth_science/models/ORBIT-2/examples/run_orbit2_train.py), or Bayes-CAST **`train_edm.py`** / **`launch_diffusion.sh`** when present (see `sbatch_train_amd.sh` / `sbatch_train_perf_amd.sh`). Studio `orbit2_rank_hook_runner.py` applies `ORBIT2_RANK_PRE_TRAIN_HOOK` before the launcher (set hook path **before** `sbatch` so the wrapper bakes it into the rank script).
- Data: 10.0_arcmin PRISM same-dir (`interm_8m_prism.yaml`) or ERA5 1.0° same-dir (`interm_8m_era5.yaml`, `ORBIT2_CONFIG_TEMPLATE`) — timing only until true downscaling targets are staged; for HBM saturation see [STAGING_ERA5_FOR_HBM.md](../../../earth_science/models/ORBIT-2/recipes/perf-optimizer-loop/STAGING_ERA5_FOR_HBM.md)
- Profiler: [orbit2_profiler_hook.py](../../../earth_science/models/ORBIT-2/examples/orbit2_profiler_hook.py) via `ORBIT2_RANK_PRE_TRAIN_HOOK` (res_slimvit path; Bayes EDM defaults to no hook unless overridden)

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

## Historical lessons

Dated job-history notes (HydraGNN/ORBIT-2 loops, rocprofiler, attribution): [reference/lessons.md](reference/lessons.md). Shared Omnistat/TraceLens pitfalls: [ai4science-studio/reference/rocm.md](../ai4science-studio/reference/rocm.md).
