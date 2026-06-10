# ORBIT-2 perf-analysis HANDOFF

_Last updated: 2026-06._

## What works

| Item | Status | Notes |
|------|--------|-------|
| 1-node × 8-GPU training on 10.0_arcmin same-dir | **Green** | Representative steady batch time ~0.5–2 s after warmup (workload- and I/O-dependent) |
| `sbatch_train_amd.sh` + `run_orbit2_train.py` | **Green** | gptl4py stub, FusedAttn fallback when `xformers.ops` unavailable, batch cap |
| Node health probe (exit 42) | **Green** | Catches broken home/shared/SIF/data mounts before training |
| `superres_mag: 1` same-dir config | **Required** | Without it, 4× superres vs same-resolution targets causes tensor shape mismatch |

## Landmines

1. **`max_epochs` must be ≥ 2** — upstream loop is `while (epoch_start + 1) < max_epochs`; `max_epochs=1` runs zero epochs.
2. **`xformers.ops` missing in overlay** — `run_orbit2_train.py` can fall back to `FusedAttn.DEFAULT` (PyTorch SDPA). Rebuild overlay per `build_overlay_amd.sh` if you need the CK attention path.
3. **`bfloat16` + CK attn** needs working `xformers.ops`; use `ORBIT2_DATA_TYPE=float32` until the overlay provides `ops`.
4. **Same-dir data** — both `low_res_dir` and `high_res_dir` point at the same grid; set `superres_mag: 1` in `interm_8m_lux.yaml` / `interm_8m_lux_era5.yaml`.
5. **Batch cap** — must patch `intermediate_downscaling` before `main()` (not via `runpy.run_path` reload).

## FOM contract (steady-state + loss sanity)

**Primary FOM:** `steady_batch_time_s` — mean per-batch wall time from **epoch ≥ 2**, excluding **batch index &lt; 1** within each epoch (skips compile/warmup batch 0). Epochs 0–1 are warmup.

**Loss sanity:** epoch-completed losses must **strictly decrease** epoch-over-epoch (`loss_sanity_pass`). Same-dir data makes absolute loss non-physical; use as crash/hang detector only.

**Scaling runs:** use `ORBIT2_MAX_EPOCH=6` (trains epochs 0–4) so the steady window spans epochs 2–4. Parse with:

```bash
python3 parse_training_log.py --log ... --steady-epoch-start 2 --warmup-batches-per-epoch 1
python3 collate_scaling_study.py --log-dir .../logs --jobs ... --steady-epoch-start 2 --require-loss-sanity
```

## Scaling / wall-clock lessons

- Long runs on full 180×360 grids need **enough wall time** (multi-epoch × many batches can exceed 1 h on some sites).
- Point SLURM `--output` / `--error` at a **shared project filesystem** your compute nodes can always read during outages (not only the submission directory on a login node).
- **Multi-node hangs** are often **infrastructure** (drained nodes, stuck scheduler states, shared filesystem hiccups) — confirm node state with `sinfo` / site ops before assuming a code bug.
- For exclusive GPU jobs, **avoid stacking two jobs on the same node**. Pin nodes with `--nodelist=...` only when your site policy allows it and you have confirmed the nodes are healthy.

## Next work (after scaling baseline)

- Port `perf-optimizer-loop/` + `lever_catalog.yaml` from the HydraGNN reference recipe.
- Rebuild overlay; re-test `bfloat16` + CK attn when `xformers.ops` is available.
- Stage **2.5_arcmin** PRISM targets for scientifically meaningful loss (not same-dir timing).

## Key paths (parameterize with `AI4S_SHARED_DIR`)

- Data: `$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin` (PRISM same-dir) or `.../era5/1.0_deg` (ERA5 same-dir sanity template).
- Overlay: `$AI4S_SHARED_DIR/models/ORBIT-2/overlays/orbit2-overlay.img`
- Perf runs: `$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/`
- Training logs (recommended): `$AI4S_SHARED_DIR/models/ORBIT-2/outputs/train/logs/`
- Omnistat / TraceLens tools: set **`OMNIHUB_TOOLS_DIR`** to your site-local checkout (see root `.cluster-config.yaml` key `omnihub.tools_dir` if your site maintains one).
