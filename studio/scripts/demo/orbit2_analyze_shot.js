'use strict'
const { chromium } = require('/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright')
const RID = process.env.RID
;(async () => {
  const browser = await chromium.launch({ channel: 'chromium', headless: true,
    args: ['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'] })
  const page = await browser.newPage()
  await page.setViewportSize({ width: 1280, height: 1050 })
  await page.goto('http://localhost:5275', { waitUntil: 'networkidle', timeout: 30000 })
  await page.evaluate(async (rid) => {
    const job = await (await fetch(`/api/jobs/${rid}`)).json()
    const st = window.__studioStore.getState()
    st.setDomain('earth_science')
    st.setModel({ slug: 'ORBIT-2', name: 'ORBIT-2 (Climate Downscaling)' })
    st.setRunId(rid)
    st.setResult(job.result)
    st.setStep(4)
  }, RID)
  await page.waitForTimeout(2500)
  await page.screenshot({ path: '/home/spannala/orbit2_analyze.png', fullPage: true })
  console.log('shot taken')
  await browser.close()
})().catch(e => { console.error('FAIL:', e.message); process.exit(1) })
