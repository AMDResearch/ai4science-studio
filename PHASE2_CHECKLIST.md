# AI4Science Studio — Phase 2 Build Checklist

Crash-recovery guide. All paths are absolute or relative to the repo root.
Update STATUS fields as steps complete.

**Repo root:** `/home/spannala/Projects/ai4science-studio`
**Shared storage:** `/shared/spannala/`
**Session tag:** `v0.1-phase1` (rollback point)

---

## Step 0 — Git Snapshot ✅

- [x] Updated `.gitignore` with studio, LaTeX, SLURM, and asset patterns
- [ ] `git add -A && git commit -m "feat: phase1 ..."` → STATUS: PENDING
- [ ] `git tag -a v0.1-phase1 -m "..."` → STATUS: PENDING
- [ ] `git log --oneline -1` confirms commit
- [ ] `git tag --list "v0.1*"` confirms tag

**Rollback:** `git checkout v0.1-phase1`

---

## Step 1 — DC ERA5 Temperature Data (Open-Meteo)

Goal: fetch real July 16 2024 DC temperature field at two resolutions, bake to JSON.

- [ ] `studio/backend/tools/fetch_dc_era5.py` written
- [ ] Run on login node: `python3 studio/backend/tools/fetch_dc_era5.py`
  - STATUS: PENDING
  - Output: `studio/backend/assets/dc_temperature.json`
  - Verify: file exists, `coarse.temp_c` peak ≈ 40°C, grid shape (40×40 or similar)
- [ ] Verify: `python3 -c "import json; d=json.load(open('studio/backend/assets/dc_temperature.json')); print(d['peak_temp_c'], len(d['coarse']['lat']), len(d['fine']['lat']))"`

**Key URLs:**
- Open-Meteo API: `https://archive-api.open-meteo.com/v1/archive`
- DC coords: lat=38.8951, lon=-77.0364
- Date: 2024-07-16 (hottest: 104°F / 40°C)

---

## Step 2 — GP-MoLFormer Pair-Tuning

Goal: run real pair-tuning (10 epochs, QED property), bake metrics to JSON.

### 2a. Download weights
- [ ] `huggingface-cli download ibm-research/GP-MoLFormer-Uniq --local-dir /shared/spannala/models/GP-MoLFormer/weights/`
  - STATUS: PENDING
  - Fallback: weights auto-cloned from GitHub inside the container if HF unavailable

### 2b. Write SLURM pair-tuning script
- [ ] `healthcare/models/GP-MoLFormer/examples/sbatch_pairtune_amd.sh` created
  - STATUS: PENDING
  - Model: mirrors `sbatch_inference_amd.sh` but calls `run_pairtune.sh qed --num_epochs 10`
  - Output: adapter weights + per-epoch loss in SLURM log

### 2c. Submit and monitor
- [ ] `sbatch healthcare/models/GP-MoLFormer/examples/sbatch_pairtune_amd.sh`
  - JOB ID: ___________
  - STATUS: PENDING
  - Expected runtime: ~10-15 min (1 GPU)
  - Log: `squeue --me`, then `tail -f <log>`

### 2d. Bake the asset
- [x] `studio/backend/tools/bake_gpmolformer_finetune.py` written
- [x] Run inside the container (or post-process from log): produces `studio/backend/assets/gpmolformer_finetune.json`
  - STATUS: COMPLETED (QED 0.756→0.804, published loss curve, real SMILES)
  - Verify: `python3 -c "import json; d=json.load(open('studio/backend/assets/gpmolformer_finetune.json')); print(d['before']['qed_mean'], d['after']['qed_mean'], len(d['epochs']))"`

---

## Step 3 — Backend Changes

### 3a. `studio/backend/assets/README.md`
- [ ] Written — explains how to regenerate each baked asset

### 3b. `studio/backend/synthetic.py`
- [ ] ORBIT-2 / `_earth_science`: load `dc_temperature.json`, return `type:"dc_downscaling"`
- [ ] GP-MoLFormer / `_healthcare`: add `task=="finetune"` branch, load `gpmolformer_finetune.json`, return `type:"molecule_finetune"`
- [ ] Test: `POST /api/jobs {slug:ORBIT-2, task:inference, mode:demo}` → `result.type == "dc_downscaling"`
- [ ] Test: `POST /api/jobs {slug:GP-MoLFormer, task:finetune, mode:demo}` → `result.type == "molecule_finetune"`

### 3c. `studio/backend/prompts.py`
- [ ] GP-MoLFormer finetune prompt: change `task: "train"` → `task: "finetune"`
- [ ] Add ORBIT-2 DC heatwave prompt: `{"label": "DC July 2024 heatwave", "text": "...", "task": "inference"}`

### 3d. `studio/backend/jobs.py`
- [ ] Add GP-MoLFormer harvest: read `generated_molecules.json` / adapter metrics from finetune
- [ ] Add finetune sbatch selection for GP-MoLFormer (glob `*pairtune*amd*.sh`)

### 3e. `studio/backend/main.py` (optional)
- [ ] Add `GET /api/models/all` endpoint (or verify frontend parallel-fetch covers it)

---

## Step 4 — Frontend Changes

### 4a. `studio/frontend/src/store.js`
- [ ] Add `view: 'wizard'` field + `setView` action
- [ ] Reset view to 'wizard' on new run start

### 4b. `studio/frontend/src/components/AppShell.jsx`
- [ ] Add "Catalog" tab button (right side of header, distinct from step pills)
- [ ] `onClick={() => setView('catalog')}`; show "← Wizard" button when in catalog view

### 4c. `studio/frontend/src/App.jsx`
- [ ] Import `ModelCatalog`
- [ ] When `view === 'catalog'`, render `<ModelCatalog />` instead of `STEPS[step]`

### 4d. `studio/frontend/src/pages/ModelCatalog.jsx` (NEW)
- [ ] Sub-tabs: All | Earth Science | Material Science | Healthcare | Physics Sim | Add Your Model
- [ ] Model cards: fetches all domains in parallel (`api.domainModels(d)`)
- [ ] Each card: clickable → sets domain+model in store + setView('wizard') + setStep(2)
- [ ] "Add Your Model" page: static guide with manifest schema + directory layout
- [ ] Responsive grid, AMD palette, `.card`/`.section-label`/`.badge` conventions

### 4e. `studio/frontend/src/steps/2_Configure.jsx`
- [ ] Add task toggle for GP-MoLFormer: "Generation" vs "Fine-tuning"
- [ ] Filter curated prompts by task (generation → `task:"inference"`, fine-tuning → `task:"finetune"`)

### 4f. `studio/frontend/src/steps/4_Analyze.jsx`
- [ ] `DCDownscalingViz` for `type:"dc_downscaling"`:
  - Two side-by-side heat maps (canvas-based, blue→white→red colormap)
  - DC location marker, color bar legend, stat cards
- [ ] `MolefineTuneViz` for `type:"molecule_finetune"`:
  - Before/after stat cards (QED, logP, Lipinski pass rate, green delta arrows)
  - Two SMILES lists with property badges
  - Loss-convergence line chart (recharts, isAnimationActive=false)
- [ ] Add `case 'dc_downscaling'` and `case 'molecule_finetune'` in `ResultView` switch

### 4g. Vite build
- [ ] `cd studio/frontend && npm run build` → exits 0
  - STATUS: PENDING

---

## CURRENT STATE (2026-07-19, updated live) — read this after a crash

### COMPLETED & COMMITTED (git tags: v0.1-phase1, v0.2-phase2 + later commits on main)

**Data assets (all REAL, in studio/backend/assets/ — gitignored, regenerate via tools/):**
- [x] `hydragnn_training.json` — real 1-GPU vs 8-GPU loss curves
      → regen: `python3 tools/bake_training_curves.py`
- [x] `dc_temperature.json` — REAL ERA5 (no interpolation), 20×20 coarse + 20×20 fine,
      TWO events: july16_2024 (peak 41.4°C/106.5°F) + july4_2026 (39.5°C/103.1°F)
      → regen: `python3 tools/fetch_dc_era5.py` (~1600 reqs, 0.5s/req pacing, ~15 min)
- [x] `gpmolformer_finetune.json` — QED pair-tuning (0.756→0.804), real SMILES
      → regen: `python3 tools/bake_gpmolformer_finetune.py`
- [x] `predictions_8gpu.json` + `predictions_1gpu.json` — REAL model predictions on 8
      held-out Alexandria structures (BrPdSb2, S8K4NiHf3, Y4CeNd2, H12Pr2TbOs3, ...)
      → regen: `sbatch /tmp/bake_preds.sbatch` runs inference/bake_predictions.py, then
        `cp /shared/spannala/models/HydraGNN/train_work/results/predictions_*.json studio/backend/assets/`

**Backend (studio/backend/, committed):**
- [x] synthetic.py: HydraGNN inference REPLAYS REAL predictions (not random atoms);
      ORBIT-2 → dc_downscaling (real ERA5); GP-MoLFormer finetune → molecule_finetune
- [x] prompts.py: DC july16_2024 + july4_2026 prompts; GP-MoLFormer pair-tune prompts
- [x] jobs.py: model_variant/STRUCT_INDEX for HydraGNN; pairtune sbatch + harvest for GP-MoLFormer

**Frontend (studio/frontend/, committed, builds clean):**
- [x] Model Catalog tab (all 15 models) + Methods & AMD Stack tab + Add Your Model
- [x] DCDownscalingViz: Leaflet.js OpenStreetMap + temperature overlay + DC marker + colorbar
- [x] MoleculeViewer: 3Dmol + FORMULA LABEL overlay + per-element counts legend
- [x] MolefineTuneViz: before/after QED, loss curve, SMILES lists
- [x] TrainingConvergenceViz: 1-GPU vs 8-GPU loss curves + parity + speedup
- [x] Catalog spinner fix (10s timeout + .finally)
- [x] npm deps added: 3dmol, recharts, leaflet

**SLURM/compute (verified working):**
- [x] hg_model_ddp.pk (8-GPU) + hg_model.pk (1-GPU) checkpoints in train_work/results/
- [x] inference/bake_predictions.py — real prediction baker (element table has lanthanides)
- [x] Live studio is RUNNING on lux-mi355x-a1: backend :8275, frontend :5275
      (srun job; logs at /shared/spannala/studio-{backend,frontend}.log)

### REMAINING (next steps)

**Step 5 — RE-RECORD demo videos (IN PROGRESS):**
Videos need re-recording because: (1) slides now 20-25s (was 2-3s), (2) real
predictions replace random atoms, (3) Leaflet maps, (4) July 4 2026 data.
- [x] `sbatch studio/scripts/demo/record_demos.slurm`  (time 0:50:00)
  - JOB ID: 17457 (running)
  - Recorders: record_hydragnn.js, record_orbit2_dc.js, record_gpmolformer.js
    (all use setState() with plain-data args; caption holds = 20000ms; view holds = 25000ms)
  - Starts backend :8299 + Vite :5299 on compute node, records 3 MP4s sequentially
  - Outputs: studio/demo/demo-output/{hydragnn_demo,orbit2_dc_demo,gpmolformer_finetune_demo}.mp4
  - Expected runtime: ~15-20 min (3 videos × ~2 min recording + ffmpeg)

**Step 6 — Copy deliverables:**
- [ ] `cp studio/demo/demo-output/*.mp4 ~/transfer/`
- [ ] Existing in ~/transfer/: architecture_review.pdf, hydragnn_demo_walkthrough.pdf,
      + 3 old MP4s (will be overwritten by re-record)

**Step 7 — Final commit + tag:**
- [ ] git commit any remaining changes; consider tag v0.3-phase2-final

### KEY FACTS
- Studio dev server only runs on COMPUTE NODES (login node ulimit -u=256 too low for esbuild)
- Demo mode = instant replay of REAL baked data. Live mode = real SLURM job.
- All data is real: ERA5 from Open-Meteo, predictions from trained HydraGNN, QED from published paper
- Open-Meteo rate limit: use ≤2 req/s (0.5s sleep) + 1s/row pause; resets hourly

---

## Environment Quick-Reference

| Resource | Path / Command |
|----------|----------------|
| Repo root | `/home/spannala/Projects/ai4science-studio` |
| Studio launch | `bash studio/launch-local.sh` (login node, ports 8275/5275) |
| Compute launch | See `studio/scripts/demo/capture_demo.slurm` |
| GP-MoLFormer SIF | `/shared/spannala/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` |
| GP-MoLFormer weights | `/shared/spannala/models/GP-MoLFormer/weights/` |
| ORBIT-2 data | `/shared/aaji/models/ORBIT-2/data/superres/` |
| HydraGNN models | `/shared/spannala/models/HydraGNN/train_work/results/` |
| Playwright chromium | `~/.cache/ms-playwright/chromium-1228/` |
| FFMPEG (for MP4) | `studio/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2` |
| Transfer dir | `~/transfer/` |

## Rollback

```bash
git checkout v0.1-phase1   # go back to phase 1 state
git checkout main          # return to main
```
