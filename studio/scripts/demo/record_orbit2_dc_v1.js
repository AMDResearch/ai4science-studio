'use strict'
const path = require('path')
// Studio dir (repo-relative by default; override with STUDIO_DIR).
const STUDIO_DIR = process.env.STUDIO_DIR || path.resolve(__dirname, '..', '..')
const { chromium } = require(path.join(STUDIO_DIR, 'demo', 'node_modules', 'playwright'))
const { execFileSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8376'
const OUT   = process.env.OUT_DIR   || path.join(STUDIO_DIR, 'demo', 'demo-output')
const FFMPEG = process.env.FFMPEG || path.join(STUDIO_DIR, 'backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
const HOLD = 20000

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
    if (o.param) s.setParam(o.param.k, o.param.v)
    if (o.runId !== undefined) s.setRunId(o.runId)
    if (o.result !== undefined) s.setResult(o.result)
    if (o.step !== undefined) s.setStep(o.step)
  }, obj)
}

async function launchDemo(page, body) {
  return page.evaluate(async ([back, b]) => {
    const r = await (await fetch(`${back}/api/jobs`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b),
    })).json()
    const rid = r.run_id
    for (let i=0; i<60; i++) {
      const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
      if (j.state === 'completed' || j.state === 'failed') return { rid, job:j }
      await new Promise(r => setTimeout(r, 1000))
    }
    return { rid, job:null }
  }, [BACK, body])
}

// Click a story-act stepper button by (partial) title text. The ORBIT2StoryViz
// stepper renders one pill per act; clicking pauses auto-play and focuses that
// act so its maps are shown. Best-effort — wrapped by the caller in try/catch.
async function clickAct(page, titleFragment) {
  await page.evaluate(frag => {
    const btns = Array.from(document.querySelectorAll('button'))
    const b = btns.find(x => x.textContent && x.textContent.includes(frag))
    if (b) b.click()
  }, titleFragment)
}

;(async () => {
  const browser = await chromium.launch({
    channel:'chromium', headless:true,
    args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({ viewport:{width:1920,height:1080}, deviceScaleFactor:2 })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'orbit2_dc_demo_v1.webm')
  const mp4Path  = path.join(OUT, 'orbit2_dc_demo_v1.mp4')

  await page.goto(FRONT, { waitUntil:'networkidle', timeout:45000 })
  await page.waitForTimeout(1500)
  await applyWideLayout(page)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format:'jpeg',quality:92,maxWidth:3840,maxHeight:2160,everyNthFrame:1 })
  // CDP only emits a frame when the page visually changes, so long static holds
  // would otherwise capture almost nothing. Keep the latest frame and sample it
  // on a fixed 2 fps timer so every slide contributes ~40 frames regardless of
  // on-screen motion. Encode at the same 2 fps so playback matches wall-clock.
  const frames = []
  let lastFrame = null
  rec.on('Page.screencastFrame', async ({data,sessionId}) => {
    lastFrame = Buffer.from(data,'base64')
    await rec.send('Page.screencastFrameAck',{sessionId}).catch(()=>{})
  })
  const sampler = setInterval(() => { if (lastFrame) frames.push(lastFrame) }, 500)

  // ── Slide 1: Landing ────────────────────────────────────────────────────────
  await showCaption(page, 'ORBIT-2 — AI Climate Downscaling on AMD Instinct MI355X')
  await page.waitForTimeout(HOLD)

  // Domain
  await setState(page, { domain:'earth_science', step:1 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Earth Science domain — weather, climate, and atmospheric models')
  await page.waitForTimeout(HOLD)

  // Model
  await setState(page, {
    model:{slug:'ORBIT-2',name:'ORBIT-2 (Climate Downscaling)',domain:'earth_science'},
    mode:'demo', task:'story', step:2
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'ORBIT-2: 8M-param PRISM-trained climate ViT, 4× temperature super-resolution')
  await page.waitForTimeout(HOLD)

  // Configure — headline robustness + physics story
  await setState(page, {
    prompt:"Show the ORBIT-2 downscaling story over Washington DC: the pretrained model, its out-of-distribution accuracy gap, and how a physics-residual head on real GHSL urban density improves the fine-grid temperature prediction.",
    step:3,
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Real GPU runs on MI355X — replayed, not synthesized (measured numbers)')
  await page.waitForTimeout(HOLD)

  // Launch the 4-act OOD story (real replay: type=orbit2_story -> ORBIT2StoryViz)
  const {rid:r0, job:j0} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'story', mode:'demo',
    prompt:"Show the ORBIT-2 downscaling story over Washington DC: the pretrained model, its out-of-distribution accuracy gap, and how a physics-residual head on real GHSL urban density improves the fine-grid temperature prediction.",
    params:{ dc_event:'july16_2024' },
  })
  await setState(page, { runId:r0, result:j0?.result, step:4 })
  await page.waitForTimeout(2000)

  // ── Act-by-act narration follows the paper (docs/orbit2_dc_downscaling.tex).
  //    The viz auto-plays; we also click each stepper pill so the focused act's
  //    maps stay on screen for the full slide. Numbers match the re-baked asset
  //    = paper Table 1 (16 Jul 2024, true-OOD, land-only MAE, °C). ─────────────
  try { await clickAct(page, 'Pretrained ORBIT-2') } catch {}
  await showCaption(page, 'True OOD test: PRISM-trained ORBIT-2 applied to independent Open-Meteo ERA5 it never saw')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Act 1 — Pretrained ORBIT-2 fails OOD: DC tmax MAE 2.40°C, worse than trivial interpolation')
  await page.waitForTimeout(HOLD)

  try { await clickAct(page, 'Out-of-distribution gap') } catch {}
  await showCaption(page, 'Act 2 — OOD gap: plain bilinear (0.93°C) beats the 8M-param foundation model')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Full finetuning cannot cross it — OOD error is best at epoch 1, then diverges (2.04°C)')
  await page.waitForTimeout(HOLD)

  try { await clickAct(page, 'Physics-residual head') } catch {}
  await showCaption(page, 'Act 3 — Move physics out of the backbone: a ~10k-param residual head on the bilinear field')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Inputs: real GHSL built-up density + signed distance-to-water (coastal cooling)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Act 3 — Head beats bilinear everywhere: core 0.93→0.80°C (−13%), whole-land 0.93→0.76°C')
  await page.waitForTimeout(HOLD)

  try { await clickAct(page, 'Physics where the physics is') } catch {}
  await showCaption(page, 'Act 4 — An explicit UHI equation on top adds nothing: the head already internalized it')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Decisive detail: clean the PRISM ocean sentinel before coarsening, or coastal errors invert')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Known physics, learned as a residual — measured on real MI355X GPU runs, not synthesized')
  await page.waitForTimeout(HOLD)

  // ── Second half: two-resolution DC downscaling (type=dc_downscaling) ─────────
  await setState(page, {
    task:'inference',
    prompt:"Downscale the July 14-17 2024 Washington DC record heatwave (104F / 40C peak) from ERA5 0.25-degree to 0.1-degree resolution — show temperature field at two resolutions.",
    param:{ k:'dc_event', v:'july16_2024' },
    step:3,
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Two-resolution view: July 16 2024 — DC record heatwave (peak 41.4°C / 106.5°F)')
  await page.waitForTimeout(HOLD)

  const {rid:r1, job:j1} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'inference', mode:'demo',
    prompt:"Downscale the July 14-17 2024 Washington DC record heatwave (104F / 40C peak) from ERA5 0.25-degree to 0.1-degree resolution — show temperature field at two resolutions.",
    params:{ dc_event:'july16_2024' },
  })
  await setState(page, { runId:r1, result:j1?.result, step:4 })
  await page.waitForTimeout(1500)

  await showCaption(page, 'Coarse 0.25° ERA5 input (left) vs fine 0.1° ORBIT-2 output (right)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Red dot = Washington DC · real ERA5 temperature from Open-Meteo, no interpolation')
  await page.waitForTimeout(HOLD)

  // ── Switch to July 4 2026 dataset ───────────────────────────────────────────
  const {rid:r2, job:j2} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'inference', mode:'demo',
    prompt:"Downscale the July 4 2026 Independence Day heat in Washington DC from ERA5 0.25-degree to 0.1-degree resolution.",
    params:{ dc_event:'july4_2026' },
  })
  await setState(page, { runId:r2, result:j2?.result, step:4 })
  await page.waitForTimeout(1500)

  await showCaption(page, 'Dataset switch: July 4 2026 — Independence Day heatwave (peak 39.5°C / 103.1°F)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'ORBIT-2 4× spatial super-resolution: 0.25° → 0.1° (~28 km → ~11 km)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Real ERA5 data via Open-Meteo · AMD Instinct MI355X cluster · AI4Science Studio')
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
  await new Promise((res,rej)=>{ ffp.on('close',c=>c===0?res():rej(new Error(`ffmpeg ${c}`))); ffp.on('error',rej) })
  // execFileSync (no shell) — paths/filters with brackets or spaces stay literal.
  execFileSync(FFMPEG, [
    '-y','-i',webmPath,'-vf','scale=3840:2160:flags=lanczos',
    '-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p', mp4Path,
  ], { stdio:'inherit' })
  await browser.close()
  const sz = (require('fs').statSync(mp4Path).size / 1024 / 1024).toFixed(1)
  console.log(`[done] ${mp4Path} (${sz} MB)`)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
