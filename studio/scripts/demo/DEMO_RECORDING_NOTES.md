# Demo video recording — changes & rationale

Notes on the demo-video recorders (`record_hydragnn.js`, `record_orbit2_dc.js`,
`record_gpmolformer.js`) and the studio changes needed to make them produce
correct videos. Recorded here so a future session does not re-derive the same
bugs.

## How the recorders work

Each script launches headless Chromium (Playwright), opens the running studio
frontend, and drives the 5-step wizard by:
1. Mutating the Zustand store via `window.__studioStore.getState().setX(...)`
   (`setState()` helper) to jump between Domain → Model → Configure → Run →
   Analyze without clicking.
2. POSTing to the backend `/api/jobs` directly (`launchDemo()`), polling until
   the job completes, then injecting the result into the store.
3. Overlaying a caption per slide and holding, while a CDP screencast captures
   frames. Frames are piped to ffmpeg → WebM → MP4.

The recorders therefore depend on `window.__studioStore` being the **exact same
store instance the rendered React components subscribe to**.

## Fixes applied (2026-07-21)

### 1. Two Zustand store instances (root cause of "frozen on step 0")
**Symptom:** every recorded video showed the app stuck on the Domain page even
though captions advanced through all slides. The wizard never navigated.

**Cause:** `index.html` loaded the entry point with a cache-buster query,
`<script src="/src/main.jsx?v=20260721a">`. The browser/Vite dev server key
modules by full URL *including the query string*, so the query-tagged entry
evaluated `store.js` on a separate module path from the one the component tree
imported (`./store` with no query). Result: two live `create()` stores. The
recorder mutated one; the UI rendered from the other. Proven on a live server:
after a real click advanced the UI to "Material Science Models",
`window.__studioStore.getState()` still read `step:0, domain:null`.

**Fix:**
- Expose the store from **inside `store.js`** itself
  (`if (typeof window !== 'undefined') window.__studioStore = useStore`) instead
  of via a separate `import` in `main.jsx`. This binds the global to the one
  instance the whole app graph shares, regardless of how the entry is loaded.
- Removed the `?v=...` cache-buster from `index.html`.
- Removed the now-redundant store exposure from `main.jsx`.

### 2. Backend port mismatch (500 on every rendered page)
**Symptom:** rendered pages showed a red `500:` error; domain cards never loaded.

**Cause:** the Vite dev proxy forwards `/api` → `127.0.0.1:8376`
(`frontend/vite.config.js`), but `record_demos.slurm` started the backend on
`8299`. The recorder's own direct API calls (to `BACK_URL`) worked, so the job
exited 0, but every page's proxied `/api` fetch hit a dead port.

**Fix:** set the launcher `BACK_PORT=8376` to match the proxy, and updated the
three recorders' `BACK` default to `http://127.0.0.1:8376`. **Keep the launcher
backend port and the vite proxy target in sync.**

### 3. Near-empty videos (~100 KB, ~8 frames)
**Symptom:** MP4s encoded successfully but were a fraction of a second long.

**Cause:** the CDP screencast only emits a frame when the page **visually
changes**. With long static caption holds (2D molecule fallback, no animation),
almost no frames were emitted across an entire video.

**Fix:** retain the latest screencast frame and sample it on a fixed 2 fps timer
(`setInterval(..., 500)`); encode at matching `-r 2` so playback tracks
wall-clock. Each slide now contributes ~40 frames regardless of on-screen motion.

## Recording environment (GTK / Chromium)

Compute nodes are missing the GTK libs Chromium needs
(`libatk-1.0.so.0: cannot open shared object file`). The working recipe (used by
`record_demos.slurm` and for ad-hoc runs on a node with a live studio, e.g. a4):
`apt-get download` the `*t64` GTK/X libs, `dpkg-deb -x` into a scratch dir, add
it to `LD_LIBRARY_PATH`, copy `chromium-1228` to scratch, and point
`PLAYWRIGHT_BROWSERS_PATH` at it. Note the Ubuntu 24.04 `t64` package suffixes.

## Content / narrative changes

- Slide hold reduced 30s → 20s; app content adaptively scaled (`#root`
  transform) so tall Analyze pages fit the frame without cropping.
- HydraGNN captions updated to the current baked numbers (15× data, 600k vs 40k
  structures, corr 0.82→0.89, R² 0.67→0.79, MAE 0.288→0.240) and inference uses
  curated `struct_index` materials (FeS2, NaFeO2) with real pred-vs-DFT values.
- ORBIT-2 rewritten around the paper (`docs/orbit2_dc_downscaling.tex`): the
  headline is the 4-act OOD story (`task:"story"` → `type:"orbit2_story"`) —
  pretrained fails OOD (2.40°C) < bilinear (0.93°C); finetuning diverges;
  a ~10k-param physics-residual head beats bilinear everywhere (core 0.93→0.80°C,
  −13%). Two-map view peaks: 41.4°C/106.5°F (Jul 16 2024), 39.5°C/103.1°F
  (Jul 4 2026).
- GP-MoLFormer captions match the baked pair-tuning run (QED 0.756→0.804,
  Lipinski 100%→90%).

## Verifying output before shipping

Do **not** trust exit code or file size alone. Extract mid-slide frames and
confirm the rendered results page (not step 0, not a 500) is on screen:
```
ffmpeg -i <video>.mp4 -vf "select='eq(n,340)'" -vsync 0 -c:v mjpeg out.jpg
```
(This ffmpeg build has no reliable seek for these files and lacks a working PNG
muxer in some paths — use `-c:v mjpeg` and frame-select, not `-ss`.)

## 4K / 16:9 `_v1` variants (2026-07-21)

Added `record_{hydragnn,orbit2_dc,gpmolformer}_v1.js` + runner
`record_demos_v1.sh`. They output `*_demo_v1.mp4` at **3840×2160 (16:9)** and copy
the results to `~/transfer/`. Originals are untouched.

### What changed vs the originals
- **Viewport + capture:** `newContext` viewport `1920×1080` with
  `deviceScaleFactor:2` (paints at native 4K, sharp text); CDP screencast
  `maxWidth:3840, maxHeight:2160, quality:92`; the mp4 step forces exact dims with
  `-vf scale=3840:2160:flags=lanczos`; VP9 webm bitrate `1500k→12000k`, `crf 30→24`.
- **Fill the frame, don't letterbox.** The originals scaled `#root` by
  `min(1, innerH/scrollH, innerW/scrollW)` — only ever shrinking, and shrinking to
  fit the *tallest* Analyze page's full height, which squeezed the app into a small
  centered box with big margins. `_v1` does two things instead:
  1. `applyWideLayout(page)` injects CSS after load to widen the app —
     `#root main { max-width:none }`, `#root aside { flex:0 0 22% }` (from 34%),
     and relaxes inner `max-width` caps. Recorder-side only; no app file edits.
  2. Width-first fit in `showCaption`: `s = innerWidth/scrollWidth`, then clamp
     down to `innerHeight/scrollHeight` only if the scaled height would overflow.
     Normal steps fill the 16:9 frame ~1:1; only genuinely tall pages scale down.
- **Captions enlarged ~1.5×** for big-display legibility (font 20→30px, padding
  12/28→18/42px, etc.).
- **GP-MoLFormer `_v1` drops the baseline-generation act** — starts directly at
  fine-tuning (Pair-tuning), per demo direction (the baseline task-select was a
  dead-end revisit).

### mp4 encode: use `execFileSync`, not `execSync` (bug fix)
The originals encoded the mp4 with
`execSync(\`${FFMPEG} ... ${webmPath} ... ${mp4Path}\`)` — this runs through
**bash**, so any `[`, `]`, `(`, `)`, or space in an interpolated value (paths, the
new `-vf scale=` filter, SMILES-like strings) breaks the command. `_v1` uses
`execFileSync(FFMPEG, [args...])` (no shell) so everything is passed literally.
The VP9 `spawn` step already used an arg array and was fine.

### Where to run (important)
Run on the **compute node** (a4 = `lux-mi355x-a4`), not the login node
("login node has limits"). The studio node is held by the `studio-hold` SLURM job
(`squeue -u $USER`). Launch:
```
srun --jobid=<hold-jobid> --overlap bash \
  ~/Projects/ai4science-studio/studio/scripts/demo/record_demos_v1.sh
```
- a4 has node (`~/.local/bin/node`), the chromium-1228 cache, writable `/scratch`,
  and (unlike some compute nodes) apt-get reachability — so the `*t64` GTK-lib
  staging works there. Chrome fails with `libatk-1.0.so.0: not found` without it.
- **Fixed live ports (SERVING.md): frontend 5376, backend 8376** — NOT 5299 (a
  stale default in the older `record_demos.slurm`). `record_demos_v1.sh` defaults
  to 5376/8376 and only health-checks the already-running services (it does not
  start/stop them, since a4's services are persistent).
- Verify each output is `3840×2160` (`ffprobe ... stream=width,height`) and
  spot-check a mid-slide frame per the section above before shipping.
