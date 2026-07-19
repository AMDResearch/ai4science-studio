'use strict'
/* ORBIT-2 DC temperature downscaling demo video. */
const path = require('path')
const { chromium } = require('/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright')
const { execSync } = require('child_process')

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK  = process.env.BACK_URL  || 'http://127.0.0.1:8299'
const OUT   = process.env.OUT_DIR   || '/home/spannala/Projects/ai4science-studio/studio/demo/demo-output'
const FFMPEG = process.env.FFMPEG   || '/home/spannala/Projects/ai4science-studio/studio/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'

async function caption(page, text, ms = 2200) {
  await page.evaluate(t => {
    const el = document.createElement('div')
    el.id = '__cap__'
    el.style.cssText = 'position:fixed;bottom:48px;left:50%;transform:translateX(-50%);' +
      'background:rgba(237,28,36,.92);color:#fff;font-weight:700;font-size:22px;' +
      'padding:14px 32px;border-radius:12px;z-index:99999;max-width:80vw;text-align:center;' +
      'font-family:Inter,sans-serif;letter-spacing:.02em;box-shadow:0 4px 24px rgba(0,0,0,.5)'
    el.textContent = t; document.body.appendChild(el)
  }, text)
  await page.waitForTimeout(ms)
  await page.evaluate(() => document.getElementById('__cap__')?.remove())
}

async function launchDemo(page, body) {
  return page.evaluate(async ([back, b]) => {
    const r = await (await fetch(`${back}/api/jobs`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b),
    })).json()
    const rid = r.run_id
    for (let i=0; i<60; i++) {
      const j = await (await fetch(`${back}/api/jobs/${rid}`)).json()
      if (j.state === 'completed' || j.state === 'failed') return {rid, job:j}
      await new Promise(r => setTimeout(r, 1000))
    }
    return {rid, job:null}
  }, [BACK, body])
}

;(async () => {
  const browser = await chromium.launch({
    channel:'chromium', headless:true,
    args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--use-gl=swiftshader'],
  })
  const ctx = await browser.newContext({viewport:{width:1280,height:900}})
  const page = await ctx.newPage()
  const webmPath = path.join(OUT, 'orbit2_dc_demo.webm')
  const mp4Path  = path.join(OUT, 'orbit2_dc_demo.mp4')

  await page.goto(FRONT, {waitUntil:'networkidle', timeout:45000})
  await page.waitForTimeout(1200)

  const rec = await ctx.newCDPSession(page)
  await rec.send('Page.startScreencast', {format:'jpeg',quality:85,maxWidth:1280,maxHeight:900,everyNthFrame:2})
  const frames = []
  rec.on('Page.screencastFrame', async ({data,sessionId}) => {
    frames.push(Buffer.from(data,'base64'))
    await rec.send('Page.screencastFrameAck',{sessionId}).catch(()=>{})
  })

  const st = s => page.evaluate(fn => { const st=window.__studioStore.getState(); fn(st) }, s)

  // Act 1: July 16 2024 heatwave
  await caption(page, 'ORBIT-2 Downscaling — Washington DC Heatwave', 2500)
  await st(s => { s.setDomain('earth_science'); s.setStep(1) })
  await page.waitForTimeout(900)
  await caption(page, 'Select ORBIT-2 — Vision Foundation Model for Climate Downscaling', 2200)
  await st(s => { s.setModel({slug:'ORBIT-2',name:'ORBIT-2 (Climate Downscaling)',domain:'earth_science'}); s.setMode('demo'); s.setTask('inference'); s.setStep(2) })
  await page.waitForTimeout(900)
  await caption(page, 'July 16 2024: DC Record High 104°F / 40°C — Hottest Since 1930', 2500)
  await st(s => { s.setPrompt("Downscale the July 14-17 2024 Washington DC record heatwave (104F / 40C peak) from ERA5 0.25-degree to 0.1-degree resolution — show temperature field at two resolutions."); s.setParam('dc_event','july16_2024'); s.setStep(3) })
  await page.waitForTimeout(1200)

  const {rid:r1, job:j1} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'inference', mode:'demo',
    prompt:"Downscale the July 14-17 2024 Washington DC record heatwave (104F / 40C peak) from ERA5 0.25-degree to 0.1-degree resolution — show temperature field at two resolutions.",
    params:{dc_event:'july16_2024'},
  })
  await st(([rid,result]) => {
    const s=window.__studioStore.getState(); s.setRunId(rid); s.setResult(result); s.setStep(4)
  }, [r1, j1?.result])
  await page.waitForTimeout(3000)
  await caption(page, 'Real ERA5 Data: 0.25° Coarse → 0.1° Fine Grid — DC marked in red', 2500)
  await page.waitForTimeout(2000)

  // Act 2: July 4 comparison
  await caption(page, 'Compare with Independence Day Heat: July 4 2024', 2200)
  const {rid:r2, job:j2} = await launchDemo(page, {
    slug:'ORBIT-2', domain:'earth_science', task:'inference', mode:'demo',
    prompt:"Downscale the July 4 2024 Independence Day heat in Washington DC from ERA5 0.25-degree to 0.1-degree resolution.",
    params:{dc_event:'july4_2024'},
  })
  await st(([rid,result]) => {
    const s=window.__studioStore.getState(); s.setRunId(rid); s.setResult(result); s.setStep(4)
  }, [r2, j2?.result])
  await page.waitForTimeout(3000)
  await caption(page, 'Real ERA5 + ERA5-Land — Open-Meteo Data — AMD MI355X Cluster', 2500)
  await page.waitForTimeout(1500)

  await rec.send('Page.stopScreencast')

  const ffp = require('child_process').spawn(FFMPEG, [
    '-y','-f','image2pipe','-r','12','-i','pipe:0',
    '-c:v','libvpx-vp9','-b:v','1200k','-crf','33','-pix_fmt','yuv420p', webmPath,
  ])
  for (const f of frames) ffp.stdin.write(f)
  ffp.stdin.end()
  await new Promise((res,rej) => { ffp.on('close',c=>c===0?res():rej(new Error(`ffmpeg ${c}`))); ffp.on('error',rej) })
  execSync(`${FFMPEG} -y -i ${webmPath} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p ${mp4Path}`, {stdio:'inherit'})
  await browser.close()
  console.log('[done] ORBIT-2 DC video:', mp4Path)
})().catch(e => { console.error('FAIL:', e.message); process.exit(1) })
