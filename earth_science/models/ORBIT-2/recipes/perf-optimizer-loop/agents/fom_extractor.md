# fom_extractor subagent — ORBIT-2

Computes per-iteration FOMs for the ORBIT-2 loop and optionally TraceLens↔Omnistat correlation (reuse HydraGNN §4 algorithm when traces exist).

## Inputs

- `<perf_run>/manifest.json` — must include `job_id`, `runtime_seconds`, `global_batch_size`, `parallelism`, `data_type`.
- `<perf_run>/orbit2-train-<jobid>.out`
- `<perf_run>/omnistat-db/` + live VM URL (or start VM with `-fs.disableMmap`)
- `<perf_run>/traces/*.pt.trace.json` (rank-0 kineto)

## Outputs

- `<perf_run>/foms.json` — schema below
- `<perf_run>/kernel_correlation.csv` — optional; same column idea as HydraGNN `fom_extractor.md`

## `foms.json` schema (ORBIT-2)

```json
{
  "job_id": "12345",
  "primary_fom": "throughput_samples_per_s",
  "throughput_samples_per_s": 1.23e4,
  "steady_batch_time_s": 0.41,
  "global_batch_size": 2048,
  "mfma_bf16_tflops_per_card_avg": null,
  "hbm_read_GBps_per_card_avg": null,
  "xgmi_GBps_avg": null,
  "loss_sanity_pass": true,
  "final_loss": 0.05
}
```

## Step 1 — Log FOMs (required)

Run the repo extractor (writes `foms.json` base fields):

```bash
python3 "$REPO_ROOT/earth_science/models/ORBIT-2/examples/run_fom_extractor.py" --job-dir "$PERF_RUN"
```

If `manifest.json` lacks `global_batch_size`, pass `--global-batch-size` explicitly.

**Effective-batch integrity (read before trusting a throughput delta):** `foms.json` includes
`hbm_reserved_GB`, `hbm_reserved_pct_288`, `max_batches_per_epoch`, **`throughput_method`**,
**`partial_step_fraction`**, and **`steady_realized_batch_dims`**. Throughput now prefers the
**real per-step batch dim** (`throughput_method=real_per_step_batch`, read from the EDM `y.shape`
line) instead of the nominal `global_batch_size`, so partial trailing batches no longer inflate it.
Still **reject any cross-run comparison where `partial_step_fraction`, `steady_realized_batch_dims`,
`hbm_reserved_pct_288`, or `max_batches_per_epoch` deviates materially from the baseline** — even a
correct number can hide a different work mix.

Two artifacts caught this way, both from `num_workers` × the Bayes-CAST IterableDataset
(each worker shards files via `per_worker = n_files // (num_workers × data_par_size)` and batches
its own shard **independently**):
- **loop overnight-3x-001 iter-1** (60 files, num_workers=4): per_worker `60//32 = 1` → realized
  batch collapsed 4096→1704, HBM 236→98 GB, batches/epoch 3→4 → nominal "+129%" was fake.
- **clean nw test (jobs 10504 nw=1 vs 10505 nw=4, 100 files)**: staged to `per_worker = 3` so full
  batches refill to 4096, but each worker still emits a partial **1016**-sample trailing batch →
  `steady_realized_batch_dims=[1016, 4096]`, `partial_step_fraction=0.57`. Nominal throughput said
  +80% (2993→5379 s/s); **real-per-step throughput showed +2.7% (2987→3068 s/s) = noise.**

**Conclusion: `num_workers` does not help ORBIT-2 throughput — the workload is compute-bound.**
The full-4096 step time is ~equal (11.0 s nw=1 vs 10.8 s nw=4). Cap at `num_workers ≤ 4` for I/O
overlap only; never use it as a throughput lever, and never compare runs with different
`steady_realized_batch_dims`. Baseline at batch 4096 on 3× data = ~82% HBM (236 GB).

**Log format (do not "fix" the parser):** the perf-loop default trainer is Bayes-CAST
`launch/train_edm.py`, which does **not** emit `Batch N: X seconds` / `Epoch N completed. Loss:`.
It emits per step (rank 0): `epoch:  N batch_idx M world_rank 0  loss  tensor(L, ...)` for loss
and `my rank 0. tic4-tic1 in X seconds` for wall time. `parse_training_log.py` reads **both** this
EDM format and the older `intermediate_downscaling.py` format; per-epoch loss is synthesized from
per-step losses (mean) for the loss-sanity gate. Crashed jobs (no `batch_idx` lines) correctly
yield `epoch_records=0`, null throughput, and `loss_sanity_pass=null` — that is the signal a job
never reached steady state, not a parser bug.

Optional: `export ORBIT2_TSDB_URL=http://127.0.0.1:8428` during login-node enrichment so the Python tool queries BF16 MFMA + FETCH + XGMI (metric names may need site tuning).

## Step 2 — PromQL (BF16 MFMA + HBM read)

Use **job-windowed** instant queries with `time=<unix>` (see ai4science-perf-analysis SKILL).

| Field | PromQL sketch |
|-------|----------------|
| BF16 TFLOP/s | `avg(rate(SQ_INSTS_VALU_MFMA_MOPS_BF16[30s]) * on(instance) group_left() max by(instance)(rmsjob_info{jobid="JOBID"})) * 512 / 1e12` |
| HBM read GB/s | `avg(rate(FETCH_SIZE[30s]) * on(instance) group_left() max by(instance)(rmsjob_info{jobid="JOBID"})) * 1024 / 1e9` |

Merge results into `foms.json` (overwrite `null`s from the Python tool if queries succeed).

## Step 3 — kernel_correlation.csv

Follow HydraGNN `material_science/models/HydraGNN/recipes/perf-optimizer-loop/agents/fom_extractor.md` §4 (1-second windows), substituting ORBIT trace glob `traces/orbit2-epoch*-rank0.pt.trace.json`.

## Final stdout

```
STATUS=ok; reason=foms throughput=<x> steady_batch=<y>s mfma_bf16=<z|n/a>
```
