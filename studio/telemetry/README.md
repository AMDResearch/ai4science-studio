# Studio Telemetry — Live 8-GPU HydraGNN training with Omnistat

Adds a **LIVE 8-GPU HydraGNN training** case to AI4Science Studio that captures AMD
Omnistat GPU telemetry during the run and renders it in an interactive
"System Telemetry" panel on the Analyze page. Also adds a configurable epochs slider
and a demo (baked-asset) replay of the same panel.

## Self-owned (no dependency on colleagues)

All perf tooling and data are reproduced under `/shared/spannala`:

| Thing | Path | Provenance |
|-------|------|-----------|
| Omnistat venv | `/shared/spannala/perf-tools/omnistat-venv` | built fresh from source (`build_perf_tools.sh`) |
| VictoriaMetrics | `/shared/spannala/perf-tools/victoriametrics/victoria-metrics-prod` | copied portable Go binary (`stage_victoriametrics.sh`) |
| Alexandria dataset | `/shared/spannala/models/HydraGNN/weights/Alexandria-v2.bp` | copied 23 GB (`copy_dataset.sh`) |

Independence check: `grep -rn '/shared/aaji\|/shared/omnihub' studio/telemetry/ studio/backend/jobs.py`
should return nothing at runtime.

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
- `copy_dataset.sh` — copy Alexandria-v2.bp into spannala-owned storage.
- `sbatch_train_telemetry_amd.sh` — the 8-GPU training + Omnistat sbatch (per-run isolated).
- `bake_telemetry_asset.py` — turn a real run's telemetry into the demo asset.
- The omnistat config template is reused in-repo at
  `material_science/models/HydraGNN/recipes/perf-analysis/omnistat.config.template`.

## Run

```bash
sbatch studio/telemetry/sbatch_train_telemetry_amd.sh          # full 200-epoch run
HG_NUM_EPOCH=30 sbatch studio/telemetry/sbatch_train_telemetry_amd.sh   # short demo run
```
Telemetry + manifest land in `train_work/perf-runs/<jobid>/`; the studio backend
range-queries the omnistat DB after completion to populate the Analyze panel.
