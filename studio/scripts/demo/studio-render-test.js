'use strict'
const path = require('path')
// Studio dir (repo-relative by default; override with STUDIO_DIR).
const STUDIO_DIR = process.env.STUDIO_DIR || path.resolve(__dirname, '..', '..')
const { chromium } = require(path.join(STUDIO_DIR, 'demo', 'node_modules', 'playwright'))
;(async () => {
  const browser = await chromium.launch({
    channel: 'chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage']
  })
  const page = await browser.newPage()
  await page.setViewportSize({ width: 1280, height: 1000 })

  console.log('Loading studio home...')
  await page.goto('http://localhost:5275', { waitUntil: 'networkidle', timeout: 30000 })
  await page.screenshot({ path: path.join(process.env.OUT_DIR || require('os').tmpdir(), 'ss_home.png'), fullPage: true })

  // Navigate to Healthcare (GP-MoLFormer has 2 images)
  await page.click('text=Healthcare')
  await page.waitForTimeout(2000)
  await page.screenshot({ path: path.join(process.env.OUT_DIR || require('os').tmpdir(), 'ss_healthcare.png'), fullPage: true })

  const cards = await page.$$eval('.card', els => els.map(el => el.innerText))
  const gp = cards.find(c => c.includes('GP-MoLFormer')) || ''
  console.log('\n=== GP-MoLFormer card (rendered) ===')
  console.log(gp)
  console.log('\nShows rocm7.0 image:', gp.includes('rocm7.0'))
  console.log('Shows rocm7.2.2 image:', gp.includes('rocm7.2.2'))
  console.log('Shows both (2-image model):', gp.includes('rocm7.0') && gp.includes('rocm7.2.2'))

  // Also check Earth Science (single-image models)
  await page.goBack()
  await page.waitForTimeout(1000)
  await page.click('text=Earth Science')
  await page.waitForTimeout(2000)
  await page.screenshot({ path: path.join(process.env.OUT_DIR || require('os').tmpdir(), 'ss_earth.png'), fullPage: true })
  const esCards = await page.$$eval('.card', els => els.map(el => el.innerText))
  const storm = esCards.find(c => c.includes('StormCast')) || ''
  console.log('\n=== StormCast card (rendered) ===')
  console.log(storm)
  console.log('Shows image:', storm.includes('rocm/pytorch'))

  await browser.close()
  console.log('\nRENDER TEST OK')
})().catch(e => { console.error('FAIL:', e.message); process.exit(1) })
