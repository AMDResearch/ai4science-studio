# lever_picker subagent

Proposes the SINGLE next lever for the orchestrator to apply, given the loop
history so far. Emits a JSON file matching the catalog entry schema; STATUS=ok
on success or STATUS=partial reason=catalog_exhausted when no candidates remain.

## Inputs

- `loop-<uuid>/foms.csv` — every iter so far, including accepted/rejected.
- `loop-<uuid>/do_not_retry.json` — array of lever ids already tried-and-rejected.
- `<current_best_perf_run>/combined_report.md` — the most recent accepted iter's synthesizer report. This is the primary evidence source for lever ranking.
- `<current_best_perf_run>/foms.json` — quantitative FOM values for the current best.
- `<current_best_perf_run>/kernel_correlation.csv` — 1-s window TraceLens↔Omnistat join (from fom_extractor). Used to disambiguate compute-bound vs memory-bound vs dispatch-bound.
- [`../lever_catalog.yaml`](../lever_catalog.yaml) — full catalog.
- (Optional) `<previous_iter_perf_run>/combined_report.md` if previous iter was a `diagnostic_only` lever (e.g. `kernel_trace_diag_only`).

## Outputs

- `loop-<uuid>/iter-<N>-lever.json` — single object matching the catalog schema, with two extra fields:
  - `picked_by`: "lever_picker"
  - `reason`: short string (1-2 sentences) explaining the choice.
- Final stdout: `STATUS=ok; reason=lever=<id>` OR `STATUS=partial; reason=catalog_exhausted`.

## Decision algorithm

### 1. Build candidate set

Read catalog; drop:
- `baseline` (iter-0 only; never picked again).
- Any lever id in `do_not_retry.json`.
- Any lever id already accepted in `foms.csv` (we don't re-pick already-applied levers; the new "best" already has them baked in).
- Any lever with `status: blocked` in the catalog (these are permanently broken on this stack; the catalog entry exists only as a cautionary record — never pick them). Read each lever's `blocked_reason` and `blocked_evidence` fields when explaining the candidate set; this is how we avoid wasting iterations on known-dead levers like `torch_compile_e3nn` (double-backward + AOT Autograd) or `tunable_op_live` (GPU mem fault during live hipBLASLt tuning).
- `kernel_trace_diag_only` — see special gate below.

**Do NOT** drop levers that appear in `foms.csv` with `accepted=false` if the failure was infrastructural (e.g. `ITER_BROKEN_NODE`, `ITER_SBATCH_FAIL`, `ANALYZE_FAIL`). Those failures are not evidence about the lever — the orchestrator handles them by retry. Cross-check by reading STATUS.txt: if the most recent ITER_DECISION for a lever was preceded by `ITER_BROKEN_NODE` for the same jobid, treat the lever as untried.

If the candidate set is empty: emit `STATUS=partial; reason=catalog_exhausted`.

### 2. Score remaining candidates

For each candidate, compute a heuristic score combining:

- **`expected_payoff` from catalog**: high=3, medium=2, low=1.
- **Evidence support from `combined_report.md`**: +2 if the report's "Next steps" or "Remedies proposed but not tried" explicitly names this lever (e.g. "torch.compile" → matches `torch_compile_e3nn`). +1 if it implicitly relates (e.g. "still dispatch-bound" → +1 for `batch_800`, `torch_compile_e3nn`).
- **Bottleneck-class alignment**: parse the report's top-3 bottleneck classes (`cpu_dispatch`, `gpu_compute`, `comm_*`, `dataloader`, ...). Match against the lever's expected target:
  - `cpu_dispatch` → `torch_compile_e3nn`, `batch_800` (+2 each)
  - `gpu_compute` + low TFLOP/s → `tunable_op`, `precision_fp32` (+2 each)
  - `gpu_memory_hbm` saturated → no lever directly helps; +0
  - `comm_xgmi` / `comm_scaleout` → `rccl_high_priority`, `nccl_minchannels` (+1 each at N=2; +2 if N≥4)
  - `dataloader` → `num_workers_12` (+2)
- **Risk penalty**: low=−0, medium=−1, high=−2.
- **Cardinality penalty**: −5 if it would be the second `diagnostic_only` lever in the same loop (we cap diagnostic iters at 1 per loop).

Top-scoring lever wins. Ties broken by lower `catalog_order`.

### 3. Special gate: `kernel_trace_diag_only`

Only pick `kernel_trace_diag_only` when ALL of:
1. The `combined_report.md` contains language indicating per-kernel attribution is unclear (e.g. "cannot attribute", "fused kernel", "unknown", "see kernel trace"), OR all other improvement-levers in the catalog are in `do_not_retry.json`.
2. AND the most recent `kernel_correlation.csv` has been examined and is insufficient (e.g. `top_kernel_busy_frac` < 0.4 in >50% of steady-state windows, indicating no single kernel dominates the 1-s windows).
3. AND no previous iter in the loop already used `kernel_trace_diag_only` (1-per-loop cap).

When picked, the orchestrator runs the iter, gets a `combined_report.md` with per-kernel-name evidence, and on the NEXT iter `lever_picker` uses that to choose a real improvement lever (likely `torch_compile_e3nn` if a specific fused kernel is dispatch-bound, or `tunable_op` if a specific GEMM shape is under-tuned).

### 4. fp32 precision gate

Only pick `precision_fp32` when the current_best `foms.json` shows the workload is compute-bound, defined as:
- `mfma_tflops > 0.3 * 78.6` (i.e. > 24 TFLOP/s per card, meaning fp64 MFMA is doing real work), OR
- `combined_report.md` Executive summary names `gpu_compute` as the #1 bottleneck.

Rationale: if the workload is still dispatch-bound (R2 state: mean util 12%, fp64 0.87% of peak), switching fp64→fp32 doubles peak FLOP/s but the dispatch ceiling stays the same. Save the lever for after `batch_800` and `torch_compile_e3nn` have lifted the dispatch ceiling enough to expose compute as the bottleneck.

### 5. Emit JSON

Write `loop-<uuid>/iter-<N>-lever.json` as a single object copying the catalog entry verbatim, with two added fields:

```json
{
  "id": "batch_800",
  "description": "...",
  "catalog_order": 1,
  "kind": "env_only",
  "env_vars": {"HG_BATCH_SIZE": "800"},
  "diagnostic_only": false,
  "expected_payoff": "high",
  "risk": "low",
  "revert_method": "drop_env",
  "citation": "...",
  "notes": "...",
  "picked_by": "lever_picker",
  "reason": "R2 combined_report names HG_BATCH_SIZE=800 as #1 next-lever; current_best is still dispatch-bound (mean util 12.65%); VRAM headroom 24/288 GB."
}
```

### 6. Final stdout

```
STATUS=ok; reason=lever=<id>
```

or

```
STATUS=partial; reason=catalog_exhausted
```

## Hallucination guardrails

- **Never invent a lever.** If you think there's a missing knob worth trying, write your reasoning to `loop-<uuid>/lever_proposal-<N>.md` and emit `STATUS=partial; reason=catalog_proposal`. The orchestrator will treat this as catalog_exhausted (since you didn't pick from the catalog) and the user/maintainer can review the proposal next morning to add it to the catalog.
- **Cite evidence by path.** Every `reason` field must point at a specific `combined_report.md` line or `foms.json` field. No vague justifications.
- **Single lever, single change.** The JSON output's `env_vars` field copies from the catalog verbatim — do NOT merge in additional env vars from other levers, even if they look complementary. Multi-knob changes are tracked across iterations, not within one.

## Failure modes

- Empty candidate set → STATUS=partial; reason=catalog_exhausted (terminal for the loop).
- Cannot read previous combined_report.md → STATUS=partial; reason=missing_report; pick lever purely by catalog_order from candidates.
- Both above conditions → STATUS=fail; reason=no_state_no_catalog. Orchestrator must log and exit.

## Notes for the implementing agent

- This subagent does NOT submit jobs or read raw traces. It is a pure-reasoning subagent over the existing reports.
- It MAY consult ROCm blog URLs from the catalog's `citation` field if the reason needs grounding, but should not fetch new pages — the reasoning lives in the catalog already.
- Reads at most 5 files (catalog + foms.csv + do_not_retry + combined_report + foms.json + kernel_correlation.csv). Bounded.
