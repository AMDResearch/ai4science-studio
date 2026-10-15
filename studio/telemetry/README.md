# Studio Telemetry — Live 8-GPU HydraGNN training with Omnistat

Adds a **LIVE 8-GPU HydraGNN training** case to AI4Science Studio that captures AMD
Omnistat GPU telemetry during the run and renders it in an interactive
"System Telemetry" panel on the Analyze page. Also adds a configurable epochs slider
and a demo (baked-asset) replay of the same panel.

## Self-owned (no dependency on anyone else's tree)

All perf tooling and data live under your own site directories, configured
through environment variables (see `studio/.env.example` and `studio/SERVING.md`):

| Thing | Default path | Provenance |
|-------|--------------|-----------|
| Omnistat venv | `$PERF_TOOLS_DIR/omnistat-venv` | built fresh from source (`build_perf_tools.sh`) |
| VictoriaMetrics | `$PERF_TOOLS_DIR/victoriametrics/victoria-metrics-prod` | staged portable Go binary (`stage_victoriametrics.sh`, from `VM_BINARY_SRC`) |
| Alexandria dataset | `$HG_DATA_DIR/Alexandria-v2.bp` | copied 23 GB (`copy_dataset.sh`, from `HG_DATASET_SRC`) |
| Kernel-trace library | `$PERF_TOOLS_DIR/omnistat-src/build-trace/libomnistat_trace.so` | built once on a compute node (`build_kernel_trace_amd.sh`) |

`PERF_TOOLS_DIR` defaults to `$AI4S_SHARED_DIR/perf-tools`, `HG_BASE` to
`$AI4S_SHARED_DIR/models/HydraGNN`, and `HG_DATA_DIR` to `$HG_BASE/weights`.

## Isolation guarantee — the live run cannot corrupt the demos

The demos replay static data and load the production checkpoint
`train_work/results/hg_model_ddp_v3.pk`. The live training run is fully isolated:

- **Writes only to a per-run dir** `train_work/perf-runs/<jobid>/`
  (`hg_model.pk`, `validation.json`, `omnistat-db/`, `manifest.json`). It does **not**
  write `results/hg_model_ddp_v3.pk` — unlike the original `hg_ddp8_v3.sbatch`, which
  did. The inference demo/live checkpoint is never touched.
- **Reads** the dataset and `hg_train_config_v3.json` read-only; overlay mounted `:ro`.
- Uses a distinct rank launcher `tele8_rank.sh` (not the production `ddp8_rank.sh`).
- Demo replay assets (`backend/assets/*.json`) are read-only; training writes nothing there.

## Files

- `CHECKLIST.md` — ordered execution checklist.
- `build_perf_tools.sh` — build the Omnistat venv from source.
- `stage_victoriametrics.sh` — stage the VictoriaMetrics binary.
- `copy_dataset.sh` — copy Alexandria-v2.bp into your own storage.
- `sbatch_train_telemetry_amd.sh` — the 8-GPU training + Omnistat sbatch (per-run isolated).
- `bake_telemetry_asset.py` — turn a real run's telemetry into the demo asset.
- `build_kernel_trace_amd.sh` — build the optional kernel-trace library.
- `omnistat.config.template` — self-contained Omnistat config the training
  sbatch renders per run (found beside the script, also under a direct sbatch).

## Run

```bash
set -a; . studio/.env; set +a                   # from the repo root: AI4S_SHARED_DIR, SBATCH_PARTITION, ...
cd studio/telemetry                             # logs/ lands here
sbatch sbatch_train_telemetry_amd.sh                    # full 200-epoch run
HG_NUM_EPOCH=30 sbatch sbatch_train_telemetry_amd.sh    # short demo run
OMNISTAT_KERNEL_TRACE=1 sbatch sbatch_train_telemetry_amd.sh   # add per-kernel trace
```
Telemetry + manifest land in `train_work/perf-runs/<jobid>/`; the studio backend
range-queries the omnistat DB after completion to populate the Analyze panel.
