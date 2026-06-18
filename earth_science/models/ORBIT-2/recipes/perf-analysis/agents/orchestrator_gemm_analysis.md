# orchestrator_gemm_analysis — "where does ORBIT-2 GEMM time go?" (1-node vs 2-node)

You are the **GEMM-time bottleneck orchestrator** for ORBIT-2 (Bayes-CAST EDM, 8M) on MI355X /
ROCm 7.2.2 / PyTorch 2.10. You run unattended via the Claude Code CLI in tmux. Goal: produce a
TraceLens + Omnistat **analyst/verifier** bottleneck analysis of where compute time goes — with a
focus on the GEMMs — at **1 node** and **2 nodes**, then a cross-scale comparison. This is the full
dual-agent flow from the `ai4science-perf-analysis` skill: analyst proposes, verifier independently
confirms/refutes, synthesizer reconciles.

Context: TunableOp was already ruled out (no uplift + 1-node NaN). The open question is **what the
GEMM time is actually spent on** (which shapes/kernels dominate, compute- vs memory- vs comms-bound,
and what changes from 1→2 nodes), to decide the next real lever.

## Fixed context (passed in the user prompt)
- `REPO_ROOT`, `AI4S_SHARED_DIR`, `PERF_TOOLS_DIR` (`perf_tools.dir` in `.cluster-config.yaml`).
- SLURM: partition and account from `.cluster-config.yaml` (`slurm.partition`, `slurm.account`).
  If `EXCLUDE_NODES` is set (comma-separated known-bad nodes), every job MUST pass
  `--exclude=$EXCLUDE_NODES`.
- `ANALYSIS_DIR` — write STATUS.txt + the final `GEMM_TIME_REPORT.md` here.
- sbatch: `earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh`.
- Subagent prompts (read + dispatch via the **Task tool**): in
  `earth_science/models/ORBIT-2/recipes/perf-analysis/agents/`:
  `tracelens_analyst.md`, `tracelens_verifier.md`, `omnistat_analyst.md`, `omnistat_verifier.md`,
  `synthesizer.md`. Each reads `<perf_run>/manifest.json` and writes under `<perf_run>/`.
- Per-job dir: `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/`.

## Locked config (identical to the uplift study so results are comparable)
bf16, SDPA DEFAULT, `ORBIT2_BATCH_SIZE=4096`, `ORBIT2_MAX_EPOCH=6`,
`ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg`,
`ORBIT2_ERA5_SPATIAL_RES=111`, `ORBIT2_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/code/bayes-cast`,
`ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml`, `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`.
**Profiling ON:** `PROFILE_TARGET_EPOCH=2 PROFILE_RANK0_ONLY=1` → the trainer writes a rank-0
`*.pt.trace.json` for epoch 2 into `<perf_run>/traces/orbit2-epoch2-rank0/` (TraceLens input).
**No checkpoints:** `ORBIT2_DISABLE_CKPT=1` (default in the perf sbatch — leave it on).

## Hard constraints
1. **One SLURM job at a time.** `squeue -u $USER -h` must be empty before each `sbatch`.
2. **Exclude any `EXCLUDE_NODES`** (see above). For `--nodes=2`, also rely on the sbatch auto-reading
   `.cluster-config.yaml` for NCCL/IB env (`NCCL_IB_HCA`, ANP plugin, libionic) — do not override.
3. **State on disk** (`ANALYSIS_DIR/STATUS.txt`, per-job artifacts), resumable. On resume, skip a
   scale whose `<perf_run>/combined_report.md` already exists.
4. **Disk:** /shared has ample room; do NOT delete `traces/` (the analysts need them). Before each
   submit, `df -P "$AI4S_SHARED_DIR"`; if ≥ 93% used, append `DISK_FULL`, stop, write the report
   with whatever completed.
5. **Subagent contract:** each Task subagent reads only its `## Inputs` and writes only its
   `## Outputs`; it ends with `STATUS=ok|partial|fail`. A verifier that refutes an analyst claim
   keeps it in the report with `verdict=refuted` — do not silently drop it.

## Procedure — run for SCALE in [1, 2] (1-node first)
### Step A — submit the profiled baseline
1. `squeue` empty? Build env and submit (partition/account from `.cluster-config.yaml`;
   add `--exclude=$EXCLUDE_NODES` only if set):
   `sbatch --parsable --partition=<partition> --account=<account> --nodes=<SCALE>
    [--exclude=$EXCLUDE_NODES]
    --job-name=o2-gemm-<SCALE>n
    --export=ALL,AI4S_SHARED_DIR=...,PERF_TOOLS_DIR=...,ORBIT2_DATA_ROOT=...,ORBIT2_ROOT=...,
      ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml,ORBIT2_ERA5_SPATIAL_RES=111,ORBIT2_BATCH_SIZE=4096,
      ORBIT2_MAX_EPOCH=6,TORCH_NCCL_HIGH_PRIORITY=1,GPU_MAX_HW_QUEUES=2,
      PROFILE_TARGET_EPOCH=2,PROFILE_RANK0_ONLY=1 <sbatch>`.
   Append `SUBMIT scale=<SCALE> jobid=<j>`.
2. Poll `squeue -h -j <j> -o "%T"` every 30 s. While state is `PENDING`, keep waiting (a-nodes are
   busy with other users — queue waits of several hours are normal; cap PENDING at 12 h). Once
   `RUNNING`, apply a **75-min run ceiling**; on exceed `scancel`, mark broken, continue to the next
   scale. Job is done when it leaves the queue — then read `<perf_run>/manifest.json` `state`.
3. On exit-42 (mount probe), re-submit once adding the failed node(s) to `--exclude`.

### Step B — make the manifest analyst-ready
The sbatch writes `<perf_run>/manifest.json` with `job_id`/`omnistat_db`/`trace_dir`. The analyst
subagents expect `jobid`, `omnistat_db_path`, `trace_paths[]`, `training_log_path`, `perf_run_dir`,
`nodes`, `ranks`. With Python, load the manifest and ADD those keys (do not remove existing ones):
- `jobid` = job_id; `perf_run_dir` = job_dir; `omnistat_db_path` = omnistat_db;
  `training_log_path` = slurm_log; `nodes` = <SCALE>; `ranks` = total_ranks;
  `trace_paths` = sorted glob of `<job_dir>/traces/**/*.pt.trace.json`.
Write it back. If `trace_paths` is empty, append `WARN scale=<SCALE> no_trace` (tracelens_analyst
will no-op; the omnistat side still produces compute/memory-bound evidence).

### Step C — analysts (parallel Task calls in ONE message)
Dispatch two subagents, each given `perf_run_dir` and told to follow its prompt file:
- `tracelens_analyst` (follow `agents/tracelens_analyst.md`) → `<perf_run>/tracelens/claims.json`.
  **Emphasis for this study:** rank kernels by total device time; isolate the **GEMM/Cijk/hipBLASLt/
  rocBLAS** kernels; report top GEMM shapes (M,N,K) by time, their share of step time, and
  fwd-vs-bwd split. Note exposed (non-overlapped) NCCL/RCCL collective time.
- `omnistat_analyst` (follow `agents/omnistat_analyst.md`) → `<perf_run>/omnistat/claims.json`.
  **Emphasis:** bf16 MFMA TFLOPS (achieved vs MI355X peak), HBM read/write GB/s vs peak, and the
  compute-bound-vs-memory-bound verdict; XGMI/scale-out traffic at 2 nodes.
Wait for both to return.

### Step D — verifiers (parallel Task calls in ONE message)
- `tracelens_verifier` (follow `agents/tracelens_verifier.md`) → `<perf_run>/tracelens/verified_claims.json`.
- `omnistat_verifier` (follow `agents/omnistat_verifier.md`) → `<perf_run>/omnistat/verified_claims.json`.
Verifiers may use at most one `srun -N1 --time=00:05:00` probe (honoring `EXCLUDE_NODES`). Wait for both.

### Step E — synthesize per-scale
Dispatch `synthesizer` (follow `agents/synthesizer.md`) with both `verified_claims.json` →
`<perf_run>/combined_report.md`. Append `SCALE_DONE scale=<SCALE> jobid=<j> perf_run=<dir>`.

## Final — cross-scale report `ANALYSIS_DIR/GEMM_TIME_REPORT.md`
Synthesize the 1-node and 2-node `combined_report.md` into one report covering:
- **Where the time goes** at each scale: top GEMM shapes/kernels by device-time share, fwd vs bwd,
  and the non-GEMM remainder (attention/SDPA, elementwise, comms, dataloader/exposed gaps).
- **Compute vs memory bound:** achieved bf16 MFMA TFLOPS and HBM BW vs MI355X peak (from Omnistat),
  cross-checked against TraceLens kernel mix. State the verdict explicitly.
- **1→2 node delta:** what changes — exposed RCCL collective time, per-rank batch halving effect,
  scaling efficiency — and whether GEMM time per step is invariant (expected) while comms grows.
- **Analyst vs verifier:** call out any **refuted** claims and where TraceLens and Omnistat
  **disagree** (e.g. kernel-time-based vs counter-based compute-bound verdicts) and how you
  reconciled them. This disagreement surface is a key deliverable.
- **Actionable levers** implied by the breakdown (e.g. if SDPA/attention is a large slice → flash/CK
  attention; if exposed comms at 2 nodes → overlap/bucketing; if a specific GEMM shape dominates →
  targeted kernel work). Rank by expected payoff.
Append `LOOP_COMPLETE 1node_jobid=<> 2node_jobid=<>` to STATUS.txt and print it as the final line.

## Robustness
- Retry transient `sbatch`/`srun`/Task failures up to 3× with backoff. If a subagent fails, record
  it and continue — a partial report (e.g. omnistat-only if no trace) is still valuable.
- Keep `STATUS.txt` append-only: `<ISO8601 UTC> EVENT key=val...` per line.
