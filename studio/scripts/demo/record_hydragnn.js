'use strict'
const path = require('path')
const { chromium } = require('/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright')
const { execSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8299'
const OUT   = process.env.OUT_DIR   || '/home/spannala/Projects/ai4science-studio/studio/demo/demo-output'
const FFMPEG = process.env.FFMPEG || '/home/spannala/Projects/ai4science-studio/studio/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
const HOLD = 30000   // every content slide holds for 30 seconds

// Show a caption overlay and hold for ms, then remove it
async function caption(page, text, ms = HOLD) {
  await page.evaluate(t => {
    document.getElementById('__cap__')?.remove()
    const el = document.createElement('div'); el.id = '__cap__'
    el.style.cssText = 'position:fixed;bottom:40px;left:50%;transform:translateX(-50%);' +
      'background:rgba(10,2,3,0.88);color:#fff;font-weight:700;font-size:20px;' +
      'padding:12px 28px;border-radius:10px;z-index:99999;max-width:88vw;text-align:center;' +
      'font-family:Inter,sans-serif;letter-spacing:.015em;' +
      'border:2px solid #ED1C24;box-shadow:0 4px 24px rgba(0,0,0,.6)'
    el.textContent = t; document.body.appendChild(el)
  }, text)
  await page.waitForTimeout(ms)
  await page.evaluate(() => document.getElementById('__cap__')?.remove())
}

// Show caption immediately (no wait) — use when content is already on screen
async function showCaption(page, text) {
  await page.evaluate(t => {
    document.getElementById('__cap__')?.remove()
    const el = document.createElement('div'); el.id = '__cap__'
    el.style.cssText = 'position:fixed;bottom:40px;left:50%;transform:translateX(-50%);' +
      'background:rgba(10,2,3,0.88);color:#fff;font-weight:700;font-size:20px;' +
      'padding:12px 28px;border-radius:10px;z-index:99999;max-width:88vw;text-align:center;' +
      'font-family:Inter,sans-serif;letter-spacing:.015em;' +
      'border:2px solid #ED1C24;box-shadow:0 4px 24px rgba(0,0,0,.6)'
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
  const ctx = await browser.newContext({ viewport: { width:1280, height:900 } })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'hydragnn_demo.webm')
  const mp4Path  = path.join(OUT, 'hydragnn_demo.mp4')

  await page.goto(FRONT, { waitUntil:'networkidle', timeout:45000 })
  await page.waitForTimeout(1500)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format:'jpeg', quality:85, maxWidth:1280, maxHeight:900, everyNthFrame:2 })
  const frames = []
  rec.on('Page.screencastFrame', async ({ data, sessionId }) => {
    frames.push(Buffer.from(data, 'base64'))
    await rec.send('Page.screencastFrameAck', { sessionId }).catch(() => {})
  })

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
  await showCaption(page, 'Validation loss convergence — 100 epochs on real Alexandria DFT data')
  await page.waitForTimeout(HOLD)

  await showCaption(page, '8-GPU trains on 6.7× more data (268k vs 40k structures) — same wall-clock')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Final accuracy: corr 0.75 → 0.79 · R² 0.56 → 0.63 · MAE 0.326 → 0.298 eV/atom')
  await page.waitForTimeout(HOLD)

  // ── Act 2: Inference + 3D viewer ─────────────────────────────────────────────
  await setState(page, { task:'inference', step:2 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Switch to Inference — predict energy on a real held-out Alexandria structure')
  await page.waitForTimeout(HOLD)

  await setState(page, {
    prompt:'Predict formation energy, atomic forces, and bulk modulus for an iron-carbon alloy with 5 atomic percent carbon using HydraGNN.',
    step:3
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Prompt: predict energy for a real DFT structure from the Alexandria dataset')
  await page.waitForTimeout(HOLD)

  const {rid:rid2, job:job2} = await launchDemo(page, {
    slug:'HydraGNN', domain:'material_science', task:'inference', mode:'demo',
    prompt:'Predict formation energy, atomic forces, and bulk modulus for an iron-carbon alloy with 5 atomic percent carbon using HydraGNN.',
    params:{model_variant:'8gpu'},
  })
  await setState(page, { runId:rid2, result:job2?.result, step:4 })
  await page.waitForTimeout(1500)

  // ── Slide: 3D molecular viewer ──────────────────────────────────────────────
  await showCaption(page, '3D atomistic structure — real coordinates from held-out DFT valset')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Ball-and-stick viewer (3Dmol.js) — drag to rotate, scroll to zoom')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Predicted energy vs DFT reference — trained 8-GPU HydraGNN model on AMD MI355X')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Model val corr 0.794 · MAE 0.298 eV/atom — production-grade materials ML')
  await page.waitForTimeout(HOLD)

  await hideCaption(page)
  await page.waitForTimeout(2000)
  await rec.send('Page.stopScreencast')

  const ffp = require('child_process').spawn(FFMPEG, [
    '-y','-f','image2pipe','-r','12','-i','pipe:0',
    '-c:v','libvpx-vp9','-b:v','1500k','-crf','30','-pix_fmt','yuv420p', webmPath,
  ])
  for (const f of frames) ffp.stdin.write(f)
  ffp.stdin.end()
  await new Promise((res, rej) => { ffp.on('close', c => c === 0 ? res() : rej(new Error(`ffmpeg ${c}`))); ffp.on('error', rej) })
  execSync(`${FFMPEG} -y -i ${webmPath} -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p ${mp4Path}`, { stdio:'inherit' })
  await browser.close()
  const sz = (require('fs').statSync(mp4Path).size / 1024 / 1024).toFixed(1)
  console.log(`[done] ${mp4Path} (${sz} MB)`)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
