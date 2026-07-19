'use strict'
/* HydraGNN demo video: Training scaling (1-GPU vs 8-GPU) then live inference + 3D viewer. */
const fs = require('fs')
const path = require('path')
const { chromium } = require('/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright')
const { execSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8299'
const OUT   = process.env.OUT_DIR   || '/home/spannala/Projects/ai4science-studio/studio/demo/demo-output'
const FFMPEG = process.env.FFMPEG   || '/home/spannala/Projects/ai4science-studio/studio/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'

async function caption(page, text, durationMs = 2200) {
  await page.evaluate(t => {
    const el = document.createElement('div')
    el.id = '__cap__'
    el.style.cssText = 'position:fixed;bottom:48px;left:50%;transform:translateX(-50%);' +
      'background:rgba(237,28,36,.92);color:#fff;font-weight:700;font-size:22px;' +
      'padding:14px 32px;border-radius:12px;z-index:99999;max-width:80vw;text-align:center;' +
      'font-family:Inter,sans-serif;letter-spacing:.02em;box-shadow:0 4px 24px rgba(0,0,0,.5)'
    el.textContent = t
    document.body.appendChild(el)
  }, text)
  await page.waitForTimeout(durationMs)
  await page.evaluate(() => { document.getElementById('__cap__')?.remove() })
}

async function launchDemo(page, body) {
  return page.evaluate(async ([back, b]) => {
    const r = await (await fetch(`${back}/api/jobs`, {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(b)
    })).json()
    const rid = r.run_id
    for (let i=0; i<60; i++) {
      const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
      if (j.state === 'completed' || j.state === 'failed') return {rid, job: j}
      await new Promise(res => setTimeout(res, 1000))
    }
    return {rid, job: null}
  }, [BACK, body])
}

;(async () => {
  const browser = await chromium.launch({
    channel: 'chromium', headless: true,
    args: ['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({ viewport: {width:1280, height:900} })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'hydragnn_demo.webm')
  const mp4Path  = path.join(OUT, 'hydragnn_demo.mp4')

  await page.goto(FRONT, { waitUntil:'networkidle', timeout:45000 })
  await page.waitForTimeout(1200)

  // Start recording
  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format:'jpeg', quality:85, maxWidth:1280, maxHeight:900, everyNthFrame:2 })
  const frames = []
  rec.on('Page.screencastFrame', async ({data, sessionId}) => {
    frames.push(Buffer.from(data, 'base64'))
    await rec.send('Page.screencastFrameAck', {sessionId}).catch(()=>{})
  })

  const st = s => page.evaluate(fn => { const st = window.__studioStore.getState(); fn(st) }, s)

  // Act 1: Training scaling
  await caption(page, 'AMD AI4Science Studio — HydraGNN Materials AI', 2500)
  await st(s => { s.setDomain('material_science'); s.setStep(1) })
  await page.waitForTimeout(1000)
  await caption(page, 'Select HydraGNN — Graph Foundation Model for Atomistic Materials', 2200)
  await st(s => { s.setModel({slug:'HydraGNN',name:'HydraGNN (Predictive GFM 2024)',domain:'material_science'}); s.setMode('demo'); s.setTask('train'); s.setStep(2) })
  await page.waitForTimeout(1000)
  await caption(page, 'Training-Scaling Demo: 1-GPU vs 8-GPU on Alexandria DFT Data', 2200)
  await st(s => { s.setPrompt('Compare HydraGNN energy-model training on 1 GPU versus 8 GPUs on Alexandria DFT data: show loss convergence, throughput speedup, and final accuracy.'); s.setStep(3) })
  await page.waitForTimeout(1200)

  const {rid: rid1, job: job1} = await launchDemo(page, {
    slug:'HydraGNN', domain:'material_science', task:'train', mode:'demo',
    prompt:'Compare HydraGNN energy-model training on 1 GPU versus 8 GPUs on Alexandria DFT data: show loss convergence, throughput speedup, and final accuracy.',
    params:{model_variant:'8gpu'},
  })
  await st(([rid, result]) => {
    const s = window.__studioStore.getState()
    s.setRunId(rid); s.setResult(result); s.setStep(4)
  }, [rid1, job1?.result])
  await page.waitForTimeout(3000)
  await caption(page, '1-GPU vs 8-GPU: 6.7× more data, corr 0.75 → 0.79', 2500)
  await page.waitForTimeout(1500)

  // Act 2: Live inference + 3D viewer
  await caption(page, 'Live Inference: Predict Energy on a Real DFT Structure', 2200)
  const {rid: rid2, job: job2} = await launchDemo(page, {
    slug:'HydraGNN', domain:'material_science', task:'inference', mode:'demo',
    prompt:'Predict formation energy, atomic forces, and bulk modulus for an iron-carbon alloy.',
    params:{model_variant:'8gpu'},
  })
  await st(([rid, result]) => {
    const s = window.__studioStore.getState()
    s.setRunId(rid); s.setResult(result); s.setStep(4)
  }, [rid2, job2?.result])
  await page.waitForTimeout(3500)
  await caption(page, '3D Atomistic Structure Viewer — ball-and-stick, rotatable', 2500)
  await page.waitForTimeout(2000)
  await caption(page, 'Predicted vs DFT Energy — AMD MI355X GPU Cluster', 2500)
  await page.waitForTimeout(1500)

  // Stop recording
  await rec.send('Page.stopScreencast')

  // Write WebM
  const { Writable } = require('stream')
  const ffmpegArgs = [
    '-y', '-f', 'image2pipe', '-r', '12', '-i', 'pipe:0',
    '-c:v', 'libvpx-vp9', '-b:v', '1200k', '-crf', '33',
    '-pix_fmt', 'yuv420p', webmPath
  ]
  const ffp = require('child_process').spawn(FFMPEG, ffmpegArgs)
  for (const f of frames) { ffp.stdin.write(f) }
  ffp.stdin.end()
  await new Promise((res, rej) => { ffp.on('close', c => c === 0 ? res() : rej(new Error(`ffmpeg ${c}`))); ffp.on('error', rej) })

  // WebM → MP4
  execSync(`${FFMPEG} -y -i ${webmPath} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p ${mp4Path}`, {stdio:'inherit'})

  await browser.close()
  console.log('[done] HydraGNN video:', mp4Path)
  console.log('[size]', require('fs').statSync(mp4Path).size, 'bytes')
})().catch(e => { console.error('FAIL:', e.message); process.exit(1) })
