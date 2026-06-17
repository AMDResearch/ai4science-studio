# story_writer subagent — ORBIT-2

Produces `story.md` + `foms.png` under `loop-<uuid>/` after the ORBIT-2 optimizer loop completes.

## Inputs

Same file list as HydraGNN `story_writer.md`, but paths under `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/`.

## Outputs

- `loop-<uuid>/story.md`
- `loop-<uuid>/foms.png` (matplotlib Agg; throughput on primary axis)

## `story.md` layout deltas

- **Hardware:** `1 node × 8 GPUs MI355X` (or N×8 after scale-out).
- **Primary FOM:** `throughput_samples_per_s` (higher is better); include `steady_batch_time_s` as secondary.
- **TL;DR table columns:** `Iter | Lever | throughput_samples_per_s | steady_batch_time_s | Δ vs prev best | Accepted? | Notes`
- **Chart:** top panel `throughput_samples_per_s` + `steady_batch_time_s` (twin axis, inverted scale for batch time *lower is better*); bottom `mfma_bf16_tflops` + `hbm_read_GBps` if present in `foms.csv`.

## matplotlib sketch

Use `throughput` from `foms.csv` as primary; mark rejects with red `x` like HydraGNN.

## Final stdout

`STATUS=ok; reason=story written ...`
