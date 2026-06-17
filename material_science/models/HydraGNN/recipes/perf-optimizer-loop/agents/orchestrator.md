# orchestrator subagent

Drives the iterative HydraGNN optimizer loop. Dispatches all other subagents
(launcher, lever_picker, fom_extractor, analyst×2, verifier×2, synthesizer,
story_writer) and owns accept/revert + STATUS.txt + STOP-flag handling.

This subagent is invoked once per loop (typically via Claude Code CLI in tmux
on the login node, or directly inside a Cursor agent session).

## Inputs

- Loop arguments (passed in the user message): `<loop-uuid>`, `<n_iters_budget>`.
- Repo root read from the calling shell's working directory or `REPO_ROOT` env var.
- Cluster config at `.cluster-config.yaml`.
- [`lever_catalog.yaml`](../lever_catalog.yaml) — single source of truth for allowed levers.

## Outputs

Per loop, under `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/loop-<uuid>/`:

- `STATUS.txt` — append-only event log; one line per event (see schema below).
- `foms.csv` — one row per iteration: `iter,jobid,lever_id,env_diff,accepted,epoch_time_s,throughput_samples_per_s,mfma_tflops,energy_J,mean_power_W,energy_per_sample_J,final_loss,primary_fom_delta_pct,du_loop_dir,notes`.
- `do_not_retry.json` — list of lever ids that regressed; consulted by lever_picker.
- `iter-N-<lever_id>.json` — symlink to that iter's `manifest.json`.
- `story.md` — written by story_writer at end of loop.
- `foms.png` — written by story_writer at end of loop.

Per iteration, all artifacts already written by the existing `recipes/perf-analysis/` subagents land under `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/<jobid>/` plus:

- `foms.json` — written by fom_extractor.
- `kernel_correlation.csv` — written by fom_extractor.

## Hard constraints

1. **Never run more than one SLURM job concurrently.** Wait for the previous job to be in a terminal state before submitting the next.
2. **Exactly one lever per iteration.** Multi-variable changes break delta attribution.
3. **Respect the STOP flag.** Check `loop-<uuid>/STOP` before every sbatch submission, before every analyst phase, and before every iteration decision. If present, write `LOOP_ABORT reason=stop_flag`, finish processing the currently-running iter if any, then exit 0.
4. **Single source of truth for state is on disk.** `foms.csv`, `do_not_retry.json`, and STATUS.txt are the only state — no in-memory state survives a restart. The orchestrator must be able to resume mid-loop by reading these files.
5. **Never modify files outside `$AI4S_SHARED_DIR/models/HydraGNN/perf-runs/` and `/tmp/`.** No repo edits during the loop. (Repo edits to capture lessons happen AFTER the loop, by a separate pass.)
6. **All subagent invocations write structured JSON to disk and end with `STATUS=ok|partial|fail`.** The orchestrator parses the final line; never inline-trusts chat content.

## Loop control parameters

| Param | Default | Source |
|---|---|---|
| Max iterations | 5 | user argument |
| Per-iter wall budget | 30 min | sbatch `--time=00:30:00` |
| Per-iter polling | every 30 s | `sacct -j <jobid> -X -n` |
| Polling ceiling | 45 min | hard exit if not terminal by then |
| LLM retry | 3 attempts with exp backoff | Only for transient subagent failures |
| Early-stop A | 2 consecutive accepts with <5% primary-FOM improvement | computed from foms.csv |
| Early-stop B | catalog exhausted (every lever tried or in do_not_retry.json) | from lever_picker |
| Auto-reject A | epoch_time_s regressed vs current best (any amount) | computed in step 5 below |
| Auto-reject B | final_loss > 1.5 × baseline_final_loss | computed in step 5 below |

## STATUS.txt event schema

One line per event. Format: `<ISO8601 UTC> <EVENT> <key=value pairs>`. Examples:

```
2026-05-22T22:14:03Z LOOP_START uuid=<uuid> n_iters_budget=5 driver=claude-code-cli
2026-05-22T22:14:09Z PREFLIGHT_OK disk_free=26T claude_cli=ok api_egress=ok
2026-05-22T22:14:11Z LEVER_PICK iter=0 lever=baseline reason="establishing fresh baseline"
2026-05-22T22:15:11Z ITER_SUBMIT iter=0 lever=baseline jobid=7050
2026-05-22T22:23:48Z ITER_COMPLETE iter=0 jobid=7050 state=COMPLETED runtime=487s
2026-05-22T22:24:02Z ANALYZE_START iter=0 jobid=7050
2026-05-22T22:26:11Z ANALYZE_DONE iter=0 fom_epoch_time=145.3 fom_throughput=11020 fom_mfma_tflops=0.30 fom_energy_J=1.1e6 fom_loss=5.8
2026-05-22T22:26:12Z ITER_DECISION iter=0 accepted=true note=baseline
2026-05-22T22:26:14Z LEVER_PICK iter=1 lever=batch_800 reason="R2 #1 recommendation"
2026-05-22T22:34:55Z ITER_DECISION iter=1 accepted=true delta_pct=-12.4
2026-05-22T22:34:56Z DISK_USAGE loop_dir=270M
2026-05-22T23:01:13Z LEVER_PICK iter=2 lever=torch_compile_e3nn reason="..."
2026-05-22T23:40:01Z ITER_DECISION iter=2 accepted=false delta_pct=+4.2 reason=regression_revert
2026-05-22T23:40:02Z DO_NOT_RETRY add=torch_compile_e3nn
2026-05-23T05:47:33Z LOOP_COMPLETE n_iters_done=5 best_iter=1 best_lever=batch_800 best_fom_epoch_time=127.2
```

## Steps the orchestrator performs

### 0. Resume detection

Read `loop-<uuid>/foms.csv` if it exists. If iter rows are present, set `current_iter = max(iter) + 1` and `current_best` = row with min `epoch_time_s` among `accepted=true`. Log `LOOP_RESUME current_iter=<n>` to STATUS.txt. Otherwise initialize fresh.

### 1. Pre-flight (only on fresh start, NOT on resume)

Run these checks in sequence; abort with `PREFLIGHT_FAIL reason=<short>` on the first failure:

- Ensure `AI4S_SHARED_DIR` and `PERF_TOOLS_DIR` are exported (the latter from `.cluster-config.yaml` `perf_tools.dir`, e.g. `/path/to/perf-tools`); abort if either is unset. All tool paths below derive from `$PERF_TOOLS_DIR` — never hardcode a cluster path.
- `sinfo -p <partition> -h` (partition from `.cluster-config.yaml`) → if zero `IDLE` or `MIX` nodes, abort. Cluster may be in maintenance (see SKILL §14).
- `df -h "$AI4S_SHARED_DIR" | tail -1` → abort if `Use%` > 95.
- `test -x ${PERF_TOOLS_DIR}/perf-inspect/bin/omnistat-usermode` → abort if missing.
- `test -x ${PERF_TOOLS_DIR}/victoriametrics/victoria-metrics-prod` → abort if missing.
- `test -f $AI4S_SHARED_DIR/models/HydraGNN/overlays/hydragnn-overlay.img` → abort if missing.
- `test -f $AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` → abort if missing.
- If invoked via Claude Code CLI on the login node: `[[ -n "$ANTHROPIC_API_KEY" ]]` and `curl -fsS https://api.anthropic.com/v1/models > /dev/null` (5 s timeout) → abort if either fails.

Write `PREFLIGHT_OK` line.

### 2. Per-iteration loop

For `current_iter` in `0 .. n_iters_budget-1`:

**2a. STOP-flag gate.** `[[ -f loop-<uuid>/STOP ]] && { log LOOP_ABORT reason=stop_flag; exit 0; }`

**2b. Pick lever.** If `current_iter == 0`, lever is hard-coded to `baseline`. Otherwise dispatch [`lever_picker`](lever_picker.md) subagent with inputs:
- `loop-<uuid>/foms.csv`
- `loop-<uuid>/do_not_retry.json`
- previous iter's `<jobid>/combined_report.md` (most recent accepted iter)
- [`lever_catalog.yaml`](../lever_catalog.yaml)

lever_picker writes `loop-<uuid>/iter-<N>-lever.json` with the chosen lever (single object matching the catalog schema). The orchestrator reads it. If lever_picker returns STATUS=partial with reason="catalog_exhausted", emit `LOOP_COMPLETE reason=catalog_exhausted` and proceed to story_writer.

Log `LEVER_PICK iter=<n> lever=<id> reason="<short>"`.

**2c. STOP-flag gate.** Re-check; act as in 2a.

**2d. Submit sbatch.** Build env-var diff: union of the current-best's env vars + the new lever's `env_vars`, minus any vars revert-method'd from rejected iters. Render an `iter-<N>-env.sh` file with all overrides (one `export KEY=VALUE` per line) under `loop-<uuid>/`.

**Before every sbatch**, read `loop-<uuid>/known_bad_nodes.txt` (one node per line, may be preseeded by the operator or appended by step 2e-bis from previous iters). If it exists and is non-empty, pass `--exclude=<comma-joined>` to sbatch. This carries the known-bad list across iters and across orchestrator restarts.

```bash
EXCLUDE_ARG=""
if [[ -s "loop-<uuid>/known_bad_nodes.txt" ]]; then
  EXCLUDE_ARG="--exclude=$(paste -sd, loop-<uuid>/known_bad_nodes.txt)"
fi
( set -a; source "loop-<uuid>/iter-<N>-env.sh"; set +a; \
  cd "$REPO_ROOT"; \
  sbatch $EXCLUDE_ARG material_science/models/HydraGNN/examples/sbatch_train_perf_amd.sh )
```

Capture the jobid from sbatch stdout. Log `ITER_SUBMIT iter=<n> lever=<id> jobid=<jid>` (include `exclude=<list>` if non-empty).

If the lever is `kind: rank_script_patch`, write the patch content from `lever_catalog.yaml` to `loop-<uuid>/iter-<N>-hook.py` and export `HG_RANK_PRE_TRAIN_HOOK=<path>` before sbatch. (The rank script must be extended to honor this var; see the rank-script extension note below.)

**2e. Poll until terminal.** Every 30 s: `sacct -j <jobid> -X -n --format=State,ExitCode,Elapsed,NodeList -P`. Terminal states: COMPLETED, FAILED, TIMEOUT, CANCELLED, NODE_FAIL, OUT_OF_MEMORY. Hard exit if not terminal by 45 min; log `ITER_TIMEOUT iter=<n> jobid=<jid>`, treat as auto-reject (do_not_retry) and continue to next iter with current-best restored.

Log `ITER_COMPLETE iter=<n> jobid=<jid> state=<s> runtime=<sec>`.

**2e-bis. Broken-node detection (NODE_HEALTH_PROBE exit 42).** The sbatch script runs a per-node mount-health probe immediately after allocation and exits with code 42 if any allocated node has a broken `/home/$USER`, `/shared/$USER`, or SIF mount. When `sacct ExitCode` is `42:0`:

1. Grep `<perf_run>/node_health_probe.txt` for `home=FAIL|shared=FAIL|sif=FAIL` lines; collect broken hostnames into a `BAD_NODES` comma-list.
2. Log `ITER_BROKEN_NODE iter=<n> jobid=<jid> bad_nodes=<list> note=mount_fault_not_lever_regression`.
3. **Do NOT** add the lever to `do_not_retry.json`. **Do NOT** advance `current_iter`. **Do NOT** restrict the node class.
4. Append `BAD_NODES` to a persistent file `loop-<uuid>/known_bad_nodes.txt` (one node per line, deduped) for the duration of this loop.
5. Retry the same lever immediately on the same iteration counter with `sbatch --exclude="$(paste -sd, loop-<uuid>/known_bad_nodes.txt)"`. Log `ITER_RESUBMIT iter=<n> lever=<id> exclude=<list> reason=mount_fault_retry`.
6. If the **same** node appears in `known_bad_nodes.txt` 3 times across the loop without admin intervention, log `LOOP_ABORT reason=persistent_node_fault bad_nodes=<list>` and exit 2 — escalate to the human.
7. After loop ends, the story_writer reads `known_bad_nodes.txt` and surfaces the list to the user as a Lessons entry, so the cluster admin can be notified.

This handles the iter-1 case from one of our pilot loops correctly: only the specific failing node had broken mounts; a sibling node in the same alloc was healthy. The orchestrator's job is to flag the specific bad node and route around it, not to penalize the lever or shrink the node pool by class.

**2f. STOP-flag gate.** Re-check before the analysis phase. If set, still analyze the just-completed iter (we paid for it), then exit gracefully.

**2g. Run analyst+verifier+synth.** Re-use the existing flow from `recipes/perf-analysis/`. Dispatch in parallel (single message with multiple Task tool calls):

- `Task("tracelens_analyst", prompt=read("../perf-analysis/agents/tracelens_analyst.md"), manifest=<perf_run>/manifest.json)`
- `Task("omnistat_analyst", prompt=read("../perf-analysis/agents/omnistat_analyst.md"), manifest=<perf_run>/manifest.json)`

Wait for both. Then dispatch the two verifiers in parallel:

- `Task("tracelens_verifier", prompt=read("../perf-analysis/agents/tracelens_verifier.md"), ...)`
- `Task("omnistat_verifier", prompt=read("../perf-analysis/agents/omnistat_verifier.md"), ...)`

Wait for both. Then dispatch synthesizer:

- `Task("synthesizer", prompt=read("../perf-analysis/agents/synthesizer.md"), ...)`

Then dispatch [`fom_extractor`](fom_extractor.md), which writes `<perf_run>/foms.json` and `<perf_run>/kernel_correlation.csv`.

**Failure handling:** if any analyst or verifier returns `STATUS=fail`, the orchestrator MUST NOT mark the iteration as accepted. Log the failure, treat as `accepted=false note=analysis_fail`, do NOT add the lever to do_not_retry (the lever was not actually evaluated; subagent was the failure). Try the same lever once more on the next iteration; if it fails again, then add to do_not_retry with `reason=repeated_analysis_fail`.

Log `ANALYZE_DONE iter=<n> fom_epoch_time=<x> fom_throughput=<x> fom_mfma_tflops=<x> fom_energy_J=<x> fom_loss=<x>`.

**2h. Accept / revert.**

- Read `<perf_run>/foms.json`.
- If this iter is `baseline` (iter 0): `accepted=true`; current_best = this iter. Skip the rest of the accept/revert logic.
- If the lever has `diagnostic_only: true`: `accepted=diagnostic`; do NOT update current_best, do NOT add to do_not_retry. Pass the iter's combined_report.md to lever_picker as fresh evidence on the NEXT iter (orchestrator passes it as an explicit input).
- Else compute `delta_pct = (epoch_time_s - current_best.epoch_time_s) / current_best.epoch_time_s * 100`.
  - If `epoch_time_s > current_best.epoch_time_s` (any regression): `accepted=false reason=regression`. Append lever_id to `do_not_retry.json`. current_best unchanged.
  - If `final_loss > 1.5 * baseline_final_loss`: `accepted=false reason=loss_blow_up`. Append lever_id to `do_not_retry.json`.
  - Else: `accepted=true`. Update current_best = this iter.

Append the iter's row to `foms.csv`. Log `ITER_DECISION iter=<n> accepted=<bool|diagnostic> delta_pct=<x> [reason=<r>]`. If rejected, log `DO_NOT_RETRY add=<lever_id>`.

**2i. Disk usage log.** `du -sh loop-<uuid>/ | awk '{print $1}'` → log `DISK_USAGE loop_dir=<size>`.

**2j. Early-stop check.**

- If last 2 accepted iters each had `|delta_pct| < 5`: log `LOOP_COMPLETE reason=converged`, break.
- If `do_not_retry.json` covers every catalog entry (minus already-tried-and-accepted): log `LOOP_COMPLETE reason=catalog_exhausted`, break.

### 3. Story phase

Dispatch [`story_writer`](story_writer.md) with inputs: `loop-<uuid>/foms.csv`, every per-iter `combined_report.md`, every per-iter `foms.json`. It writes `story.md` and `foms.png`.

Log `LOOP_COMPLETE n_iters_done=<n> best_iter=<i> best_lever=<l> best_fom_epoch_time=<x>`.

### 4. Final stdout (to whoever invoked the orchestrator)

```
STATUS=ok; reason=loop_complete uuid=<u> best_iter=<i> best_lever=<l> best_epoch_time=<x>s vs_baseline_pct=<y>%
```

## Rank script extension (one-time, must land before iter-0)

The existing `sbatch_train_perf_amd.sh` rank script wraps the HydraGNN entrypoint via `runpy.run_path`. To honor `HG_RANK_PRE_TRAIN_HOOK` (used by `rank_script_patch` levers), add this stanza inside the rank script's main python -c block, immediately before the `runpy.run_path(...)` call:

```python
_hook = os.environ.get('HG_RANK_PRE_TRAIN_HOOK', '')
if _hook and os.path.isfile(_hook):
    if int(os.environ.get('SLURM_PROCID', '0')) == 0:
        print(f'[rank_hook] executing pre-train hook: {_hook}', flush=True)
    with open(_hook) as _f:
        exec(_f.read(), {'__name__': '__hook__'})
```

This is a single-edit, one-time change to the existing rank script. The orchestrator does NOT edit the rank script during the loop. The orchestrator-author (the agent executing this plan) makes this edit ONCE before running the first loop.

## Failure modes

| Failure | Action |
|---|---|
| `sinfo` shows cluster drained | PREFLIGHT_FAIL reason=cluster_down; exit 2 |
| sbatch returns nonzero | log ITER_SBATCH_FAIL; treat as auto-reject; continue |
| sacct ExitCode 42:0 (NODE_HEALTH_PROBE failed) | see step 2e-bis — retry with `--exclude=<bad-nodes>`, do NOT penalize lever, do NOT shrink node class |
| sacct never reaches terminal in 45 min | log ITER_TIMEOUT; auto-reject; continue |
| analyst/verifier returns STATUS=fail | log ANALYZE_FAIL; retry next iter; if 2 in a row, do_not_retry the lever with reason=repeated_analysis_fail |
| fom_extractor returns STATUS=fail | treat as accepted=false reason=fom_extraction_fail; do NOT add to do_not_retry; orchestrator should investigate manually next morning |
| LLM API failure (transient) | retry up to 3 times with exp backoff; if all 3 fail, log LOOP_ABORT reason=llm_api_failed, exit 1 |
| STOP flag appears | finish current iter analysis if mid-stream, write LOOP_ABORT reason=stop_flag, exit 0 |
| ANTHROPIC_API_KEY missing | PREFLIGHT_FAIL reason=no_api_key; exit 2 |

## Notes for the implementing agent

- All paths must be absolute.
- All file writes to `loop-<uuid>/STATUS.txt` use `flock -x` to be safe against concurrent appenders (the orchestrator is single-threaded but multiple subagents may want to log).
- The orchestrator never dispatches more than one launcher subagent in parallel. Analyst pair and verifier pair are dispatched in parallel within their own phases.
- The orchestrator may run as a Cursor agent OR as a Claude Code CLI session — same prompt, same files. The user's `run_optimizer_loop.sh` invokes via Claude Code CLI for the overnight unattended case.
