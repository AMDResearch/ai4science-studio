# orchestrator subagent — ORBIT-2 iterative sysopt loop

Drives the ORBIT-2 (Bayes-CAST EDM) optimizer loop on **MI355X-class** hardware. Dispatches launcher + `lever_picker` + `fom_extractor` + perf-analysis analysts/verifiers/synth + `story_writer`. Owns accept/revert on **`throughput_samples_per_s`**, `STATUS.txt`, `STOP`, and broken-node exit **42** handling.

**HydraGNN reference:** mirror control flow in `material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/orchestrator.md`; this file only lists **ORBIT-2 deltas**.

## Inputs

- Loop args: `<loop-uuid>`, `<n_iters_budget>`.
- `REPO_ROOT` (cwd or env).
- `.cluster-config.yaml` (partition, perf_tools.dir).
- [`../lever_catalog.yaml`](../lever_catalog.yaml).

## Outputs

Under `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/loop-<uuid>/`:

- `STATUS.txt`, `foms.csv`, `do_not_retry.json`, `known_bad_nodes.txt`, `iter-N-<lever>.json` → symlink to `../<jobid>/manifest.json`, `iter-N-env.sh`, `iter-N-hook.py` (when `rank_script_patch`).

Per job: existing perf-analysis layout under `../<jobid>/` plus `foms.json` from [`examples/run_fom_extractor.py`](../../examples/run_fom_extractor.py) (extend with PromQL/kernel_correlation per [`fom_extractor.md`](fom_extractor.md) when tooling is available).

## ORBIT-2-specific hard constraints

1. **Single concurrent SLURM job** — same as HydraGNN.
2. **One lever per iteration** — same.
3. **`sbatch` entrypoint:** `earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh` (from `REPO_ROOT`).
4. **Rank hooks:** set `export ORBIT2_RANK_PRE_TRAIN_HOOK=/abs/path/iter-N-hook.py` **before** `sbatch` so the wrapper expands it into the generated rank script (`sbatch_train_perf_amd.sh` reads this at submit time). For **Bayes EDM** the default hook is empty; profiler hook is optional. Hooks are executed via `orbit2_rank_hook_runner.py` before `train_edm.py`.
5. **Env file:** `source loop-<uuid>/iter-N-env.sh` then `sbatch` with `--export=ALL` plus any `ORBIT2_*` / `NCCL_*` overrides from the lever. Match HydraGNN pattern in `run_optimizer_loop.sh` driver.
6. **Primary FOM (accept/revert):** `throughput_samples_per_s` from `<jobid>/foms.json` (higher is better). **Control:** `loss_sanity_pass` must stay true; if `final_loss` blows up vs baseline, reject (same 1.5× rule as HydraGNN optional for ORBIT timing runs).
7. **Baseline iter-0:** `lever_id=baseline` with saturated `ORBIT2_BATCH_SIZE` from [BASELINE_LOCKIN.md](../../perf-analysis/BASELINE_LOCKIN.md) + bf16 + SDPA.
8. **Exit 42 (mount probe):** identical retry with `--exclude` — do **not** add lever to `do_not_retry.json` (see ai4science-perf-analysis SKILL).

## Preflight (ORBIT-2)

- `AI4S_SHARED_DIR`, `PERF_TOOLS_DIR` set.
- `${AI4S_SHARED_DIR}/models/ORBIT-2/overlays/orbit2-overlay.img` exists.
- `${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` (or `ORBIT2_SIF`) exists.
- `omnistat-usermode` + `victoria-metrics-prod` binaries present.
- Optional Claude driver: same API checks as HydraGNN `run_optimizer_loop.sh`.

## foms.csv columns (ORBIT-2)

`iter,jobid,lever_id,env_diff,accepted,throughput_samples_per_s,steady_batch_time_s,mfma_bf16_tflops,hbm_read_GBps,xgmi_GBps,energy_J,mean_power_W,energy_per_sample_J,loss_sanity_pass,primary_fom_delta_pct,du_loop_dir,notes`

## Resume / failure modes

Same as HydraGNN orchestrator (`STATUS=ok|partial|fail` last line; resume from `foms.csv`).

## Final stdout

```
STATUS=ok; reason=loop_complete uuid=<u> best_iter=<i> best_lever=<l> best_throughput=<x> samples/s
```
