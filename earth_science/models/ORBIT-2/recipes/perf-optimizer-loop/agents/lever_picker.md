# lever_picker subagent — ORBIT-2

Same contract as `material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/lever_picker.md` with these substitutions:

## Inputs

- `loop-<uuid>/foms.csv`, `do_not_retry.json`, `iter-*-lever.json`.
- `<current_best>/combined_report.md`, `<current_best>/foms.json`, optional `kernel_correlation.csv`.
- [`../lever_catalog.yaml`](../lever_catalog.yaml).

## Outputs

- `loop-<uuid>/iter-<N>-lever.json` — catalog entry + `picked_by`, `reason`.
- Final stdout: `STATUS=ok; reason=lever=<id>` or `STATUS=partial; reason=catalog_exhausted`.

## Decision deltas (vs HydraGNN)

1. **Primary FOM:** maximize `throughput_samples_per_s` from ORBIT `foms.json` (not `epoch_time_s`). When comparing to previous best, `delta_pct = (best_throughput - new_throughput) / best_throughput * 100` for **regression** detection **or** invert for improvement — orchestrator owns the sign; you report **evidence** against `throughput` and `steady_batch_time_s`.
2. **Blocked levers:** skip any with `status: blocked` in the catalog (same as HydraGNN).
3. **Bottleneck alignment:**
   - `dataloader` / host I/O → `num_workers_8` (+2)
   - `gpu_compute` low BF16 MFMA → `torch_compile_edm`, `sdpa_efficient_vs_math` (+2)
   - `comm_xgmi` high → `fsdp_prefetch_tuning` (+1); at N≥4 consider `nccl_minchannels`
4. **fp32 discriminator gate:** only pick `precision_fp32_discriminator` when `combined_report.md` or pitfall doc requests compute-vs-memory classification.

## Hallucination guardrails

- Never invent a lever; use `lever_proposal-<N>.md` + `STATUS=partial; reason=catalog_proposal` like HydraGNN.
- Cite paths to `combined_report.md` / `foms.json` in `reason`.
