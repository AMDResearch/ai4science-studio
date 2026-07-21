# Telemetry Feature — Execution Checklist

Live 8-GPU HydraGNN training case with self-owned AMD Omnistat telemetry, an
interactive Analyze panel, a configurable epochs slider, and a 4K demo video.

All perf tooling is reproduced under `/shared/spannala` so this feature does NOT
depend on `/shared/aaji` or `/shared/omnihub`.

## Actions

1. [ ] **Build perf tools** (parallel): `./build_perf_tools.sh`
   - Fresh Omnistat venv at `/shared/spannala/perf-tools/omnistat-venv`
   - VictoriaMetrics binary at `/shared/spannala/perf-tools/victoriametrics/`
   - Verify: `omnistat-venv/bin/omnistat-usermode --help`, `victoria-metrics-prod --version`
2. [ ] **Copy dataset** (parallel): `./copy_dataset.sh`
   - `Alexandria-v2.bp` (23 GB) → `/shared/spannala/models/HydraGNN/weights/`
   - Verify: size matches source; `trainset/` + `valset/` groups present
3. [ ] **Smoke test**: submit `sbatch_train_perf_amd.sh` with 3 epochs + self-owned paths
   - `PERF_TOOLS_DIR=/shared/spannala/perf-tools`
   - `HG_DATA_DIR=/shared/spannala/models/HydraGNN/weights`
   - `HG_NUM_EPOCH=3`
   - Verify: per-job VictoriaMetrics DB written + `foms.json` produced
4. [ ] **Confirm PromQL metric labels** against the smoke-run TSDB
   (gpu util / power / temp / vram / fp64 / hbm read exact names)
5. [ ] **Backend**: 8-GPU SBATCH header + perf-sbatch selection + epochs env in
   `_build_slurm_script` (`studio/backend/jobs.py`)
6. [ ] **Backend**: `_harvest_telemetry` + read-only VM range queries + telemetry JSON
7. [ ] **Bake asset**: `./bake_telemetry_asset.py` → `studio/backend/assets/hydragnn_telemetry_8gpu.json`;
   add synthetic replay branch + curated prompt
8. [ ] **Frontend**: `TelemetryPanel` (7 tiles + 3 interactive charts) + Analyze switch case
   (`studio/frontend/src/steps/4_Analyze.jsx`)
9. [ ] **Frontend**: epochs slider in Configure for train task
   (`studio/frontend/src/steps/2_Configure.jsx`)
10. [ ] **Restart** backend/frontend per `studio/SERVING.md`; test demo + live end-to-end
11. [ ] **Record** 4K 3-min video → `~/transfer/hydragnn_training_4k_demo.mp4`

## Independence check

```
grep -rn '/shared/aaji\|/shared/omnihub' studio/telemetry/ studio/backend/jobs.py
```
Should return nothing (all perf paths point at `/shared/spannala`).
