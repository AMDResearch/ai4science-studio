'use strict'
const path = require('path')
const { chromium } = require('/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright')
const { execFileSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8376'
const OUT   = process.env.OUT_DIR   || '/home/spannala/Projects/ai4science-studio/studio/demo/demo-output'
const FFMPEG = process.env.FFMPEG || '/home/spannala/Projects/ai4science-studio/studio/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
const HOLD = 20000   // every content slide holds for 20 seconds

// Inject CSS once to widen the app so content fills a 16:9 big-display frame
// instead of sitting in a narrow centered column. Recorder-side only — does not
// touch any app source file.
async function applyWideLayout(page) {
  await page.evaluate(() => {
    document.getElementById('__wide__')?.remove()
    const st = document.createElement('style'); st.id = '__wide__'
    st.textContent = `
      #root main { max-width: none !important; }
      #root aside { flex: 0 0 22% !important; }
      #root aside > * { max-width: none !important; }
      #root main > * { max-width: none !important; }
    `
    document.head.appendChild(st)
  })
}

// Show a caption overlay and adaptively fill the frame width. The fit is
// width-first: scale up/down so #root spans the full frame width, then clamp
// down only if the scaled height would overflow (nothing cropped). Caption
// lives on <body> (outside #root) so it stays full-size and readable.
async function showCaption(page, text) {
  await page.evaluate(t => {
    const root = document.getElementById('root')
    if (root) {
      root.style.transform = ''
      root.style.transformOrigin = 'top center'
      root.style.width = '100%'
      const h = root.scrollHeight, w = root.scrollWidth
      let s = window.innerWidth / (w || 1)
      if (h * s > window.innerHeight - 8) s = (window.innerHeight - 8) / (h || 1)
      root.style.transform = `scale(${s})`
    }
    document.getElementById('__cap__')?.remove()
    const el = document.createElement('div'); el.id = '__cap__'
    el.style.cssText = 'position:fixed;bottom:56px;left:50%;transform:translateX(-50%);' +
      'background:rgba(10,2,3,0.88);color:#fff;font-weight:700;font-size:30px;' +
      'padding:18px 42px;border-radius:14px;z-index:99999;max-width:88vw;text-align:center;' +
      'font-family:Inter,sans-serif;letter-spacing:.015em;' +
      'border:3px solid #ED1C24;box-shadow:0 4px 24px rgba(0,0,0,.6)'
    el.textContent = t; document.body.appendChild(el)
  }, text)
}
async function hideCaption(page) {
  await page.evaluate(() => document.getElementById('__cap__')?.remove())
}

async function setState(page, obj) {
  await page.evaluate(o => {
    const s = window.__studioStore.getState()
    if (o.domain !== undefined) s.setDomain(o.domain)
    if (o.model !== undefined) s.setModel(o.model)
    if (o.mode !== undefined) s.setMode(o.mode)
    if (o.task !== undefined) s.setTask(o.task)
    if (o.prompt !== undefined) s.setPrompt(o.prompt)
    if (o.runId !== undefined) s.setRunId(o.runId)
    if (o.result !== undefined) s.setResult(o.result)
    if (o.step !== undefined) s.setStep(o.step)
  }, obj)
}

async function launchDemo(page, body) {
  return page.evaluate(async ([back, b]) => {
    const r = await (await fetch(`${back}/api/jobs`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b)
    })).json()
    const rid = r.run_id
    for (let i = 0; i < 60; i++) {
      const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
      if (j.state === 'completed' || j.state === 'failed') return { rid, job: j }
      await new Promise(res => setTimeout(res, 1000))
    }
    return { rid, job: null }
  }, [BACK, body])
}

;(async () => {
  const browser = await chromium.launch({
    channel: 'chromium', headless: true,
    args: ['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({ viewport: { width:1920, height:1080 }, deviceScaleFactor: 2 })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'hydragnn_demo_v1.webm')
  const mp4Path  = path.join(OUT, 'hydragnn_demo_v1.mp4')

  await page.goto(FRONT, { waitUntil:'networkidle', timeout:45000 })
  await page.waitForTimeout(1500)
  await applyWideLayout(page)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format:'jpeg', quality:92, maxWidth:3840, maxHeight:2160, everyNthFrame:1 })
  // CDP only emits a frame when the page visually changes, so long static holds
  // would otherwise capture almost nothing. Keep the latest frame and sample it
  // on a fixed 2 fps timer so every slide contributes ~40 frames regardless of
  // on-screen motion. Encode at the same 2 fps so playback matches wall-clock.
  const frames = []
  let lastFrame = null
  rec.on('Page.screencastFrame', async ({ data, sessionId }) => {
    lastFrame = Buffer.from(data, 'base64')
    await rec.send('Page.screencastFrameAck', { sessionId }).catch(() => {})
  })
  const sampler = setInterval(() => { if (lastFrame) frames.push(lastFrame) }, 500)

  // ── Slide 1: Domain picker (visible on screen) ──────────────────────────────
  await showCaption(page, 'AMD AI4Science Studio — AI for Science on AMD Instinct GPUs')
  await page.waitForTimeout(HOLD)

  // Navigate to Material Science → model list
  await setState(page, { domain:'material_science', step:1 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Material Science domain — HydraGNN graph foundation model')
  await page.waitForTimeout(HOLD)

  // Navigate to Configure (train task)
  await setState(page, {
    model:{slug:'HydraGNN',name:'HydraGNN (Predictive GFM 2024)',domain:'material_science'},
    mode:'demo', task:'train', step:2
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Task: Training Scaling — compare 1-GPU vs 8-GPU on real DFT data')
  await page.waitForTimeout(HOLD)

  // Run step
  await setState(page, {
    prompt:'Compare HydraGNN energy-model training on 1 GPU versus 8 GPUs on Alexandria DFT data: show loss convergence, throughput speedup, and final accuracy.',
    step:3
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Demo mode — replays real training results instantly, no GPU wait')
  await page.waitForTimeout(HOLD)

  // Launch and navigate to Analyze
  const {rid:rid1, job:job1} = await launchDemo(page, {
    slug:'HydraGNN', domain:'material_science', task:'train', mode:'demo',
    prompt:'Compare HydraGNN energy-model training on 1 GPU versus 8 GPUs on Alexandria DFT data: show loss convergence, throughput speedup, and final accuracy.',
    params:{model_variant:'8gpu'},
  })
  await setState(page, { runId:rid1, result:job1?.result, step:4 })
  await page.waitForTimeout(1500)

  // ── Slide: Convergence chart ────────────────────────────────────────────────
  await showCaption(page, 'Validation loss convergence — 200 epochs on real Alexandria DFT data')
  await page.waitForTimeout(HOLD)

  await showCaption(page, '8-GPU trains on 15× more data (600k vs 40k structures) — same wall-clock')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Final accuracy: corr 0.82 → 0.89 · R² 0.67 → 0.79 · MAE 0.288 → 0.240 eV/atom')
  await page.waitForTimeout(HOLD)

  // ── Act 2: Inference + 3D viewer (real curated held-out materials) ───────────
  await setState(page, { task:'inference', step:2 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Switch to Inference — predict energy on real held-out Alexandria materials')
  await page.waitForTimeout(HOLD)

  // Curated demo materials (struct_index maps to baked real predictions):
  //   0 = pyrite FeS2, 1 = permalloy FeNi3, 2 = sodium ferrite NaFeO2.
  // Each launch uses the 8-GPU model (val corr 0.889, MAE 0.240 eV/atom).
  await setState(page, {
    prompt:'Predict the per-atom DFT formation energy of pyrite FeS2 ("fool\'s gold") — an earth-abundant thin-film solar absorber and battery cathode.',
    step:3
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Material 1: pyrite FeS2 — earth-abundant solar absorber & battery cathode')
  await page.waitForTimeout(HOLD)

  const {rid:rid2, job:job2} = await launchDemo(page, {
    slug:'HydraGNN', domain:'material_science', task:'inference', mode:'demo',
    prompt:'Predict the per-atom DFT formation energy of pyrite FeS2 ("fool\'s gold") — an earth-abundant thin-film solar absorber and battery cathode.',
    params:{model_variant:'8gpu', struct_index:0},
  })
  await setState(page, { runId:rid2, result:job2?.result, step:4 })
  await page.waitForTimeout(1500)

  // ── Slide: 3D molecular viewer ──────────────────────────────────────────────
  await showCaption(page, '3D atomistic structure — real coordinates from a held-out DFT unit cell')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Viewer (3Dmol.js / SVG fallback) — bonds resolved across periodic boundaries')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'FeS2 predicted energy vs DFT reference — real pred, not random (pred −0.062 vs −0.056 eV/atom)')
  await page.waitForTimeout(HOLD)

  // Second material: sodium ferrite NaFeO2 — Na-ion battery cathode.
  await setState(page, {
    prompt:'Predict the per-atom DFT formation energy of sodium ferrite NaFeO2 — a low-cost sodium-ion battery cathode for grid-scale energy storage.',
    step:3
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Material 2: sodium ferrite NaFeO2 — low-cost Na-ion cathode for grid storage')
  await page.waitForTimeout(HOLD)

  const {rid:rid3, job:job3} = await launchDemo(page, {
    slug:'HydraGNN', domain:'material_science', task:'inference', mode:'demo',
    prompt:'Predict the per-atom DFT formation energy of sodium ferrite NaFeO2 — a low-cost sodium-ion battery cathode for grid-scale energy storage.',
    params:{model_variant:'8gpu', struct_index:2},
  })
  await setState(page, { runId:rid3, result:job3?.result, step:4 })
  await page.waitForTimeout(1500)

  await showCaption(page, 'NaFeO2 predicted vs DFT reference — trained 8-GPU HydraGNN model on AMD MI355X')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Model val corr 0.889 · MAE 0.240 eV/atom — production-grade materials ML')
  await page.waitForTimeout(HOLD)

  await hideCaption(page)
  await page.waitForTimeout(2000)
  clearInterval(sampler)
  await rec.send('Page.stopScreencast')

  const ffp = require('child_process').spawn(FFMPEG, [
    '-y','-f','image2pipe','-r','2','-i','pipe:0',
    '-c:v','libvpx-vp9','-b:v','12000k','-crf','24','-pix_fmt','yuv420p', webmPath,
  ])
  for (const f of frames) ffp.stdin.write(f)
  ffp.stdin.end()
  await new Promise((res, rej) => { ffp.on('close', c => c === 0 ? res() : rej(new Error(`ffmpeg ${c}`))); ffp.on('error', rej) })
  // execFileSync (no shell) — paths/filters with brackets or spaces stay literal.
  execFileSync(FFMPEG, [
    '-y','-i',webmPath,'-vf','scale=3840:2160:flags=lanczos',
    '-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p', mp4Path,
  ], { stdio:'inherit' })
  await browser.close()
  const sz = (require('fs').statSync(mp4Path).size / 1024 / 1024).toFixed(1)
  console.log(`[done] ${mp4Path} (${sz} MB)`)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
