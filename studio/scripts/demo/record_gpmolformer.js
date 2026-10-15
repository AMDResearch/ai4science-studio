'use strict'
const path = require('path')
// Studio dir (repo-relative by default; override with STUDIO_DIR).
const STUDIO_DIR = process.env.STUDIO_DIR || path.resolve(__dirname, '..', '..')
const { chromium } = require(path.join(STUDIO_DIR, 'demo', 'node_modules', 'playwright'))
const { execSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8376'
const OUT   = process.env.OUT_DIR   || path.join(STUDIO_DIR, 'demo', 'demo-output')
const FFMPEG = process.env.FFMPEG || path.join(STUDIO_DIR, 'backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
const HOLD = 20000

// Adaptively scale the app so the entire page fits inside the recording frame
// (tall before/after Analyze views are otherwise cropped). Caption lives on
// <body> outside #root so it stays full-size.
async function showCaption(page, text) {
  await page.evaluate(t => {
    const root = document.getElementById('root')
    if (root) {
      root.style.transform = ''
      root.style.transformOrigin = 'top center'
      root.style.width = '100%'
      const h = root.scrollHeight, w = root.scrollWidth
      const s = Math.min(1, (window.innerHeight - 8) / (h || 1), window.innerWidth / (w || 1))
      root.style.transform = `scale(${s})`
    }
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
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b),
    })).json()
    const rid = r.run_id
    for (let i=0; i<60; i++) {
      const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
      if (j.state==='completed'||j.state==='failed') return {rid, job:j}
      await new Promise(r=>setTimeout(r,1000))
    }
    return {rid, job:null}
  }, [BACK, body])
}

;(async () => {
  const browser = await chromium.launch({
    channel:'chromium', headless:true,
    args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({ viewport:{width:1280,height:900} })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'gpmolformer_finetune_demo.webm')
  const mp4Path  = path.join(OUT, 'gpmolformer_finetune_demo.mp4')

  await page.goto(FRONT, { waitUntil:'networkidle', timeout:45000 })
  await page.waitForTimeout(1500)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format:'jpeg',quality:85,maxWidth:1280,maxHeight:900,everyNthFrame:1 })
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
  await showCaption(page, 'GP-MoLFormer — AI Drug Discovery · IBM Research × AMD')
  await page.waitForTimeout(HOLD)

  // Domain
  await setState(page, { domain:'healthcare', step:1 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Healthcare domain — drug discovery, molecular design, medical imaging')
  await page.waitForTimeout(HOLD)

  // Model
  await setState(page, {
    model:{slug:'GP-MoLFormer',name:'GP-MoLFormer',domain:'healthcare'},
    mode:'demo', task:'inference', step:2,
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'GP-MoLFormer: pretrained generative model for SMILES molecule design')
  await page.waitForTimeout(HOLD)

  // Configure / run baseline generation
  await setState(page, {
    prompt:'Generate 20 drug-like molecules with a benzene scaffold (SMILES: c1ccccc1) optimized for oral bioavailability.',
    step:3
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Baseline generation: benzene-scaffold molecules, no property steering')
  await page.waitForTimeout(HOLD)

  const {rid:r1, job:j1} = await launchDemo(page, {
    slug:'GP-MoLFormer', domain:'healthcare', task:'inference', mode:'demo',
    prompt:'Generate 20 drug-like molecules with a benzene scaffold (SMILES: c1ccccc1) optimized for oral bioavailability.',
    params:{},
  })
  await setState(page, { runId:r1, result:j1?.result, step:4 })
  await page.waitForTimeout(1500)

  // ── Slides: baseline results ─────────────────────────────────────────────────
  await showCaption(page, 'Baseline generation: diverse SMILES molecules from pretrained GP-MoLFormer')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Lipinski rule-of-five: MW ≤500, LogP ≤5, HBD ≤5, HBA ≤10 — drug-likeness filter')
  await page.waitForTimeout(HOLD)

  // Switch to pair-tuning
  await setState(page, { task:'finetune', step:2 })
  await page.waitForTimeout(800)
  await showCaption(page, 'Pair-Tuning (PEFT) — steer generation toward higher QED drug-likeness score')
  await page.waitForTimeout(HOLD)

  await setState(page, {
    prompt:'Pair-tune GP-MoLFormer on 1000 QED-steered molecule pairs to shift generation toward higher drug-likeness. Compare before/after QED distribution and Lipinski compliance.',
    step:3
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Pair-tuning trains only N soft-prompt tokens — backbone weights stay frozen')
  await page.waitForTimeout(HOLD)

  const {rid:r2, job:j2} = await launchDemo(page, {
    slug:'GP-MoLFormer', domain:'healthcare', task:'finetune', mode:'demo',
    prompt:'Pair-tune GP-MoLFormer on 1000 QED-steered molecule pairs to shift generation toward higher drug-likeness. Compare before/after QED distribution and Lipinski compliance.',
    params:{},
  })
  await setState(page, { runId:r2, result:j2?.result, step:4 })
  await page.waitForTimeout(1500)

  // ── Slides: pair-tuning results (real baked values) ──────────────────────────
  await showCaption(page, 'After pair-tuning: QED mean 0.756 → 0.804 — generation steered toward drug-likeness')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Real baked run: 1000 QED-steered pairs, 10 epochs — measured before/after property shift')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Loss curve converges over 10 epochs — efficient PEFT, no backbone retraining needed')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Before/after SMILES comparison — QED badges shift higher after pair-tuning')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'GP-MoLFormer fine-tuning on AMD Instinct MI355X · IBM Research model · AI4Science Studio')
  await page.waitForTimeout(HOLD)

  await hideCaption(page)
  await page.waitForTimeout(2000)
  clearInterval(sampler)
  await rec.send('Page.stopScreencast')

  const ffp = require('child_process').spawn(FFMPEG, [
    '-y','-f','image2pipe','-r','2','-i','pipe:0',
    '-c:v','libvpx-vp9','-b:v','1500k','-crf','30','-pix_fmt','yuv420p', webmPath,
  ])
  for (const f of frames) ffp.stdin.write(f)
  ffp.stdin.end()
  await new Promise((res,rej)=>{ ffp.on('close',c=>c===0?res():rej(new Error(`ffmpeg ${c}`))); ffp.on('error',rej) })
  execSync(`${FFMPEG} -y -i ${webmPath} -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p ${mp4Path}`, { stdio:'inherit' })
  await browser.close()
  const sz = (require('fs').statSync(mp4Path).size / 1024 / 1024).toFixed(1)
  console.log(`[done] ${mp4Path} (${sz} MB)`)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
