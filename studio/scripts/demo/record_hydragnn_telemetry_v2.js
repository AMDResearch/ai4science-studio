'use strict'
// 4K (3840×2160, 16:9) demo of the HydraGNN 8-GPU TRAINING + live Omnistat GPU
// telemetry story. End-to-end workflow: Domain → Model → Configure (epochs +
// precision) → Run a LIVE 8-GPU job → Analyze/Performance (live telemetry) →
// full static demo replay → Analyze/Results (loss convergence).
//
// Built on the proven record_hydragnn_v1.js scaffolding (CDP screencast + 2 fps
// sampling + width-first fit + execFileSync encode). See DEMO_RECORDING_NOTES.md.
// Run via record_telemetry_v2.sh on the compute node (a4) — NOT the login node.
const path = require('path')
const fs = require('fs')
// Studio dir (repo-relative by default; override with STUDIO_DIR).
const STUDIO_DIR = process.env.STUDIO_DIR || path.resolve(__dirname, '..', '..')
const { chromium } = require(path.join(STUDIO_DIR, 'demo', 'node_modules', 'playwright'))
const { execFileSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5376'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8376'
const OUT   = process.env.OUT_DIR   || path.join(STUDIO_DIR, 'demo', 'demo-output')
const FFMPEG = process.env.FFMPEG || path.join(STUDIO_DIR, 'backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
const HOLD = 14000
const PROMPT = 'Train the HydraGNN energy model on 8 AMD MI355X GPUs over Alexandria DFT data and capture live GPU telemetry (utilization, power, temperature, FP64 throughput, HBM bandwidth, energy) with AMD Omnistat.'

async function applyWideLayout(page) {
  await page.evaluate(() => {
    document.getElementById('__wide__')?.remove()
    const st = document.createElement('style'); st.id = '__wide__'
    st.textContent = `
      #root main { max-width: none !important; }
      #root aside { flex: 0 0 22% !important; }
      #root aside > * { max-width: none !important; }
      #root main > * { max-width: none !important; }`
    document.head.appendChild(st)
  })
}
async function showCaption(page, text) {
  await page.evaluate(t => {
    const root = document.getElementById('root')
    if (root) {
      root.style.transform = ''; root.style.transformOrigin = 'top center'; root.style.width = '100%'
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
async function hideCaption(page) { await page.evaluate(() => document.getElementById('__cap__')?.remove()) }

async function setState(page, obj) {
  await page.evaluate(o => {
    const s = window.__studioStore.getState()
    if (o.domain !== undefined) s.setDomain(o.domain)
    if (o.model !== undefined) s.setModel(o.model)
    if (o.mode !== undefined) s.setMode(o.mode)
    if (o.task !== undefined) s.setTask(o.task)
    if (o.prompt !== undefined) s.setPrompt(o.prompt)
    if (o.runId !== undefined) s.setRunId(o.runId)
    if (o.runState !== undefined) s.setRunState(o.runState)
    if (o.result !== undefined) s.setResult(o.result)
    if (o.step !== undefined) s.setStep(o.step)
    if (o.params) for (const [k, v] of Object.entries(o.params)) s.setParam(k, v)
  }, obj)
}
async function clickTab(page, label) {
  await page.evaluate(l => {
    const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === l)
    if (b) b.click()
  }, label)
  await page.waitForTimeout(500)
}
async function launchJob(page, body, timeoutS = 60) {
  return page.evaluate(async ([back, b, ts]) => {
    const r = await (await fetch(`${back}/api/jobs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b),
    })).json()
    const rid = r.run_id
    for (let i = 0; i < ts; i++) {
      const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
      if (j.state === 'completed' || j.state === 'failed') return { rid, job: j }
      await new Promise(res => setTimeout(res, 1000))
    }
    const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
    return { rid, job: j }
  }, [BACK, body, timeoutS])
}

;(async () => {
  const browser = await chromium.launch({
    channel: 'chromium', headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 2 })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'hydragnn_telemetry_v2.webm')
  const mp4Path = path.join(OUT, 'hydragnn_telemetry_v2.mp4')

  await page.goto(FRONT, { waitUntil: 'networkidle', timeout: 45000 })
  await page.waitForTimeout(1500)
  await applyWideLayout(page)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 3840, maxHeight: 2160, everyNthFrame: 1 })
  const frames = []; let lastFrame = null
  rec.on('Page.screencastFrame', async ({ data, sessionId }) => {
    lastFrame = Buffer.from(data, 'base64')
    await rec.send('Page.screencastFrameAck', { sessionId }).catch(() => {})
  })
  const sampler = setInterval(() => { if (lastFrame) frames.push(lastFrame) }, 500)

  // ── Intro / Domain ──────────────────────────────────────────────────────────
  await showCaption(page, 'AMD AI4Science Studio — HydraGNN 8-GPU training with live Omnistat telemetry')
  await page.waitForTimeout(HOLD)
  await setState(page, { domain: 'material_science', step: 1 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Material Science — HydraGNN graph foundation model (Predictive GFM 2024)')
  await page.waitForTimeout(HOLD)

  // ── Configure: train task, epochs slider + precision toggle ─────────────────
  await setState(page, {
    model: { slug: 'HydraGNN', name: 'HydraGNN (Predictive GFM 2024)', domain: 'material_science' },
    mode: 'live', task: 'train', prompt: PROMPT, params: { epochs: 200, precision: 'fp64' }, step: 2,
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Configure: 8-GPU training · epochs slider · FP64 to exercise the MI355X matrix units')
  await page.waitForTimeout(HOLD)
  await showCaption(page, 'Why FP64: HydraGNN trains in FP64 for convergence — and inference needs it too, computing forces as dE/dx (energy gradients). FP64/FP32 beat BF16 on force accuracy.')
  await page.waitForTimeout(HOLD)

  // ── RUN a LIVE 8-GPU job on the cluster ─────────────────────────────────────
  await setState(page, { step: 3 })
  await page.waitForTimeout(600)
  await showCaption(page, 'Submit a LIVE job — 8× AMD Instinct MI355X, real DDP training on Alexandria DFT')
  await page.waitForTimeout(HOLD)

  const live = await launchJob(page, {
    slug: 'HydraGNN', domain: 'material_science', task: 'train', mode: 'live',
    prompt: PROMPT, params: { epochs: 8, precision: 'fp64' }, partition: 'lux',
  }, 5)  // don't block; show the live-collecting state next
  await setState(page, { runId: live.rid, runState: 'running', result: null, step: 4 })
  await clickTab(page, 'Performance')
  await showCaption(page, 'Live GPU telemetry — Omnistat is collecting on the MI355X node right now')
  await page.waitForTimeout(HOLD)

  // Poll the live job to completion so the panel fills with real streamed data.
  let liveJob = null
  for (let i = 0; i < 120; i++) {
    liveJob = await page.evaluate(async ([back, id]) =>
      (await (await fetch(`${back}/api/jobs/${id}`)).json()), [BACK, live.rid])
    if (liveJob.state === 'completed' || liveJob.state === 'failed') break
    await page.waitForTimeout(2000)
  }
  if (liveJob && liveJob.result && liveJob.result.telemetry) {
    await setState(page, { runId: live.rid, runState: 'completed', result: liveJob.result, step: 4 })
    await clickTab(page, 'Performance')
    await showCaption(page, 'LIVE telemetry captured — real peaks & means from the run\'s Omnistat DB')
    await page.waitForTimeout(HOLD)
    await page.evaluate(() => window.scrollTo({ top: 600, behavior: 'smooth' }))
    await showCaption(page, 'Pick any metric: FP64 throughput, HBM bandwidth, xGMI scale-up — choose what to plot from the full Omnistat catalog')
    await page.waitForTimeout(HOLD)
    await showCaption(page, 'Omnistat also captures raw hardware perf counters, scale-up (xGMI) & scale-out (network) bandwidth, host I/O traffic, and more')
    await page.waitForTimeout(HOLD)
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }))
  } else {
    await showCaption(page, 'Live job dispatched to SLURM — switching to the full reference run')
    await page.waitForTimeout(6000)
  }

  // ── STATIC DEMO replay: full 200-epoch reference telemetry ──────────────────
  const demo = await launchJob(page, {
    slug: 'HydraGNN', domain: 'material_science', task: 'train', mode: 'demo',
    prompt: PROMPT, params: { epochs: 200, precision: 'fp64' }, partition: 'lux',
  }, 30)
  await setState(page, { mode: 'demo', runId: demo.rid, runState: 'completed', result: demo.job?.result, step: 4 })
  await clickTab(page, 'Performance')
  await showCaption(page, 'Full 200-epoch reference run — the complete 8-GPU telemetry story')
  await page.waitForTimeout(HOLD)
  await page.evaluate(() => window.scrollTo({ top: 700, behavior: 'smooth' }))
  await showCaption(page, 'Peak vs mean — utilization peaks at 100% in bursts; the mean is the honest load')
  await page.waitForTimeout(HOLD)
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }))
  await page.waitForTimeout(600)

  // ── RESULTS tab at the end: loss convergence + accuracy ─────────────────────
  await clickTab(page, 'Results')
  await showCaption(page, 'Results — training loss convergence on held-out Alexandria DFT')
  await page.waitForTimeout(HOLD)
  await page.evaluate(() => window.scrollTo({ top: 500, behavior: 'smooth' }))
  await showCaption(page, 'Train · Validation · Test loss vs epoch — the model converges across 8 GPUs')
  await page.waitForTimeout(HOLD)
  await showCaption(page, 'End-to-end on AMD: open model, 8× MI355X, Omnistat telemetry, reproducible')
  await page.waitForTimeout(HOLD)

  await hideCaption(page)
  await page.waitForTimeout(1500)
  clearInterval(sampler)
  await rec.send('Page.stopScreencast')

  const ffp = require('child_process').spawn(FFMPEG, [
    '-y', '-f', 'image2pipe', '-r', '2', '-i', 'pipe:0',
    '-c:v', 'libvpx-vp9', '-b:v', '12000k', '-crf', '24', '-pix_fmt', 'yuv420p', webmPath,
  ])
  for (const f of frames) ffp.stdin.write(f)
  ffp.stdin.end()
  await new Promise((res, rej) => { ffp.on('close', c => c === 0 ? res() : rej(new Error(`ffmpeg ${c}`))); ffp.on('error', rej) })
  execFileSync(FFMPEG, [
    '-y', '-i', webmPath, '-vf', 'scale=3840:2160:flags=lanczos',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', mp4Path,
  ], { stdio: 'inherit' })
  await browser.close()
  const sz = (fs.statSync(mp4Path).size / 1024 / 1024).toFixed(1)
  console.log(`[done] ${mp4Path} (${sz} MB)`)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
