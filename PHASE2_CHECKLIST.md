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
- [ ] `studio/backend/tools/bake_gpmolformer_finetune.py` written
- [ ] Run inside the container (or post-process from log): produces `studio/backend/assets/gpmolformer_finetune.json`
  - STATUS: PENDING
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

## Step 5 — Demo Video Recordings

**Prerequisites:** Steps 1-4g complete; GP-MoLFormer finetune asset baked.

### 5a. Video recorder scripts (3 scripts)
- [ ] `studio/scripts/demo/record_hydragnn.js` written (Playwright screencast, ~90s)
- [ ] `studio/scripts/demo/record_orbit2_dc.js` written (~60s)
- [ ] `studio/scripts/demo/record_gpmolformer.js` written (~60s)

### 5b. Combined SLURM job
- [ ] `studio/scripts/demo/record_demos.slurm` written
  - Starts backend (port 8299) + Vite dev (port 5299) on compute node
  - Records all 3 videos sequentially
  - Converts WebM → MP4 via imageio-ffmpeg
- [ ] `sbatch studio/scripts/demo/record_demos.slurm`
  - JOB ID: ___________
  - STATUS: PENDING
  - Expected runtime: ~20 min (3 videos sequential)

### 5c. Expected outputs
- [ ] `studio/demo/demo-output/hydragnn_demo.mp4` (>1 MB)
- [ ] `studio/demo/demo-output/orbit2_dc_demo.mp4` (>1 MB)
- [ ] `studio/demo/demo-output/gpmolformer_finetune_demo.mp4` (>1 MB)

---

## Step 6 — GP-MoLFormer Fine-Tuning PDF (screenshot walkthrough)

- [ ] `studio/scripts/demo/capture_gpmolformer_demo.js` written
- [ ] Run via capture_demo.slurm (or new job)
- [ ] Output: `studio/demo/demo-output/gpmolformer_demo_walkthrough.pdf`

---

## Step 7 — Transfer & Final Verification

- [ ] `cp studio/demo/demo-output/*.mp4 ~/transfer/`
- [ ] `cp studio/demo/demo-output/*_walkthrough.pdf ~/transfer/`
- [ ] `ls -lh ~/transfer/` — all files present
- [ ] `scp rad-vultr-login:~/transfer/* .` (from laptop)

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
