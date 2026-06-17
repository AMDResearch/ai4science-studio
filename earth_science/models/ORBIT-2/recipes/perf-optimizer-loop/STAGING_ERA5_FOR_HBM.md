# Staging ERA5 for HBM saturation (ORBIT-2 / Bayes-CAST EDM)

**Goal:** Break the **effective per-step batch cap** seen on small ERA5 staging trees (e.g. rank-0 `y.shape` batch dim plateauing near ~1704 regardless of larger `trainer.batch_size`, and VRAM plateauing far below full HBM). See [BASELINE_LOCKIN.md](../perf-analysis/BASELINE_LOCKIN.md) calibration notes.

**Knowledge base / provenance**

- ORBIT-2 curated dataset and staging workflow: [Constellation / ORBIT-2 dataset](https://doi.ccs.ornl.gov/dataset/e4c2db1f-e88c-5ad0-bb96-59be0ef7c772) (DOI `10.13139/OLCF/2589526`) — use the official **Globus** link from that page; do not hard-code stale endpoint IDs.
- Cluster filesystem guidance: [earth_science/models/ORBIT-2/recipes/data/README.md](../data/README.md).
- ERA5 licensing: Copernicus CDS terms apply if you build your own NPZ tree from ERA5; keep citations in your runbook.

## Recommended layout

Stage under a **shared** path visible on compute nodes (same as `ORBIT2_DATA_ROOT`):

```text
$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/<your_staging_name>/
```

Point `ORBIT2_DATA_ROOT` at that directory. `render_orbit2_config.py` injects `__DATA_ROOT__` into the Bayes-CAST template.

## Levers that increase memory / work per step

1. **More time samples / longer calendar coverage** — larger dataset → more batches without repeating the same shard layout; reduces pathological sampler caps on tiny corpora.
2. **Higher spatial resolution** — set `ORBIT2_ERA5_SPATIAL_RES` (template token `__ERA5_1_SPATIAL_RES__`) to match your staged grid (e.g. 111 for 1.0°, higher for finer grids). Larger `spatial_resolution` increases tokens per sample → more activation memory and MFMA opportunity.
3. **Keep `trainer.batch_size` as the VRAM knob** — after staging changes, re-run the bf16 binary-search sweep ([`examples/sweep_orbit2_batch_bf16_amd.sh`](../../examples/sweep_orbit2_batch_bf16_amd.sh)) toward **~85–90%** `memory_reserved` on MI355-class HBM.

## Bayes-CAST hotfix checklist

- **`train_edm.py` + `ERA5_1`:** ensure `std_delta` (or equivalent normalization branch) exists for `data_key == "ERA5_1"` so training does not hit `UnboundLocalError` (see [HANDOFF.md](../perf-analysis/HANDOFF.md) landmine #7).
- **Attention path:** keep **`ORBIT2_FUSED_ATTN=DEFAULT`** (PyTorch SDPA) on ROCm 7.2.x + current xFormers wheels; CK MEA is **not** safe on this stack (HANDOFF xFormers probe).

## After staging

1. Run a short smoke (`ORBIT2_MAX_EPOCH=3`, `ORBIT2_MAX_BATCHES=20`) and confirm rank-0 logs show the expected **batch dimension** and **VRAM** trend with `ORBIT2_BATCH_SIZE`.
2. Update [BASELINE_LOCKIN.md](../perf-analysis/BASELINE_LOCKIN.md) **Reference job** with the new data path, `ORBIT2_ERA5_SPATIAL_RES`, and locked bf16 batch.
