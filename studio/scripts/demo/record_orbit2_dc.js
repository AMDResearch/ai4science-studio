'use strict'
const path = require('path')
const { chromium } = require('/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright')
const { execSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8299'
const OUT   = process.env.OUT_DIR   || '/home/spannala/Projects/ai4science-studio/studio/demo/demo-output'
const FFMPEG = process.env.FFMPEG || '/home/spannala/Projects/ai4science-studio/studio/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
const HOLD = 30000

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

;(async () => {
  const browser = await chromium.launch({
    channel:'chromium', headless:true,
    args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({ viewport:{width:1280,height:900} })
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'orbit2_dc_demo.webm')
  const mp4Path  = path.join(OUT, 'orbit2_dc_demo.mp4')

  await page.goto(FRONT, { waitUntil:'networkidle', timeout:45000 })
  await page.waitForTimeout(1500)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', { format:'jpeg',quality:85,maxWidth:1280,maxHeight:900,everyNthFrame:2 })
  const frames = []
  rec.on('Page.screencastFrame', async ({data,sessionId}) => {
    frames.push(Buffer.from(data,'base64'))
    await rec.send('Page.screencastFrameAck',{sessionId}).catch(()=>{})
  })

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
    mode:'demo', task:'inference', step:2
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'ORBIT-2: Vision Foundation Model for spatial climate downscaling')
  await page.waitForTimeout(HOLD)

  // Configure prompt
  await setState(page, {
    prompt:"Downscale the July 14-17 2024 Washington DC record heatwave (104F / 40C peak) from ERA5 0.25-degree to 0.1-degree resolution — show temperature field at two resolutions.",
    param: { k:'dc_event', v:'july16_2024' },
    step:3,
  })
  await page.waitForTimeout(800)
  await showCaption(page, 'Dataset: July 16 2024 — hottest day in DC since 1930 (104°F / 40°C)')
  await page.waitForTimeout(HOLD)

  // Launch July 16 2024
  const {rid:r1, job:j1} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'inference', mode:'demo',
    prompt:"Downscale the July 14-17 2024 Washington DC record heatwave (104F / 40C peak) from ERA5 0.25-degree to 0.1-degree resolution — show temperature field at two resolutions.",
    params:{ dc_event:'july16_2024' },
  })
  await setState(page, { runId:r1, result:j1?.result, step:4 })
  await page.waitForTimeout(1500)

  // ── Slides on Analyze: July 16 2024 ─────────────────────────────────────────
  await showCaption(page, 'Temperature field: coarse 0.25° ERA5 input (left) vs fine 0.1° output (right)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'All values are real — fetched from Open-Meteo ERA5 reanalysis, no interpolation')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Red dot = Washington DC · Blue→White→Red scale = cold→hot temperature')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Peak 41.4°C / 106.5°F — Appalachian cooling visible to the west')
  await page.waitForTimeout(HOLD)

  // ── Switch to July 4 2026 ────────────────────────────────────────────────────
  const {rid:r2, job:j2} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'inference', mode:'demo',
    prompt:"Downscale the July 4 2026 Independence Day heat in Washington DC from ERA5 0.25-degree to 0.1-degree resolution.",
    params:{ dc_event:'july4_2026' },
  })
  await setState(page, { runId:r2, result:j2?.result, step:4 })
  await page.waitForTimeout(1500)

  await showCaption(page, 'Dataset switch: July 4 2026 — Independence Day heatwave (103°F / 39.5°C)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'ORBIT-2 4× spatial super-resolution: 0.25° → 0.1° (~28 km → ~11 km)')
  await page.waitForTimeout(HOLD)

  await showCaption(page, 'Real ERA5 data via Open-Meteo · AMD Instinct MI355X cluster · AI4Science Studio')
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
  await new Promise((res,rej)=>{ ffp.on('close',c=>c===0?res():rej(new Error(`ffmpeg ${c}`))); ffp.on('error',rej) })
  execSync(`${FFMPEG} -y -i ${webmPath} -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p ${mp4Path}`, { stdio:'inherit' })
  await browser.close()
  const sz = (require('fs').statSync(mp4Path).size / 1024 / 1024).toFixed(1)
  console.log(`[done] ${mp4Path} (${sz} MB)`)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
