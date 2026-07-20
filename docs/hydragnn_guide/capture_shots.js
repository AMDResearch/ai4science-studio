'use strict'
/* Capture current-state HydraGNN studio screenshots for the LaTeX guide.
 *
 * Navigates the LIVE UI the way a user does: real DOM clicks (so React's own
 * event handlers fire and re-render — external store.setState from outside React
 * does not reliably re-render and produced 10 identical shots previously).
 *
 * The four per-material inference pages are produced by launching the demo with
 * an explicit params.struct_index (0..3), which the backend maps to a specific
 * baked prediction (see synthetic.py). The store result is set from the returned
 * job, then setStep(4) is called via a click on the ANALYZE stepper.
 *
 * Requires the studio running: BACK_URL (127.0.0.1:8376), FRONT_URL (5299). */
const fs = require('fs')
const path = require('path')
const PW = '/home/spannala/Projects/ai4science-studio/studio/demo/node_modules/playwright'
const { chromium } = require(PW)

const FRONT = process.env.FRONT_URL || 'http://127.0.0.1:5299'
const BACK = process.env.BACK_URL || 'http://127.0.0.1:8376'
const SHOTS = process.env.SHOTS_DIR || '/home/spannala/Projects/ai4science-studio/docs/hydragnn_guide/shots'
fs.mkdirSync(SHOTS, { recursive: true })

async function shot(page, file) {
  await page.waitForTimeout(700)
  await page.screenshot({ path: path.join(SHOTS, file), fullPage: true })
  console.log(`[shot] ${file}`)
}
// Click the first visible element whose trimmed text contains `txt`.
async function clickText(page, txt, opts = {}) {
  const loc = page.locator(`text=${txt}`).first()
  await loc.waitFor({ state: 'visible', timeout: opts.timeout || 8000 })
  await loc.click()
}
// Click a stepper pill by its label word (CONFIGURE, RUN, ANALYZE). The pill text
// is either "N. LABEL" (future) or "✓ LABEL" (done); only done pills are clickable.
async function clickStep(page, label, opts = {}) {
  const loc = page.locator('.step-pill').filter({ hasText: new RegExp(label, 'i') }).first()
  await loc.waitFor({ state: 'visible', timeout: opts.timeout || 8000 })
  await loc.click()
}
async function stepState(page) {
  return await page.evaluate(() => {
    const s = window.__studioStore.getState()
    return { step: s.step, domain: s.domain, model: s.model?.slug, mode: s.mode,
             task: s.task, formula: s.result?.formula, type: s.result?.type }
  })
}

;(async () => {
  const browser = await chromium.launch({
    channel: 'chromium', headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--use-gl=swiftshader'],
  })
  const page = await browser.newPage()
  page.on('console', m => { if (m.type() === 'error') console.log('[browser-error]', m.text()) })
  page.on('pageerror', e => console.log('[page-error]', e.message))
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto(FRONT, { waitUntil: 'networkidle', timeout: 45000 })
  await page.waitForTimeout(1500)

  // ── Step 1: Domain ────────────────────────────────────────────────────────
  await shot(page, '00_domains.png')

  // ── Step 2: Model (click Material Science card) ──────────────────────────
  await clickText(page, 'Material Science')
  await page.waitForTimeout(900)
  await shot(page, '01_models.png')

  // ── Step 3: Configure (click HydraGNN model card) ────────────────────────
  await clickText(page, 'HydraGNN')
  await page.waitForTimeout(900)
  // Select the training-scaling task, then screenshot the configured page.
  await clickText(page, 'Training scaling')
  await page.waitForTimeout(600)
  // Pick the curated training prompt (a .prompt-card) so Continue is enabled.
  await page.locator('.prompt-card').filter({ hasText: /GPU scaling|1 vs 8/i }).first().click({ timeout: 5000 })
  await page.waitForTimeout(500)
  await shot(page, '02_configure_train.png')

  // ── Step 4: Run (training) ───────────────────────────────────────────────
  await clickText(page, 'Continue to Run')
  await page.waitForTimeout(900)
  await shot(page, '03_run_train.png')
  await clickText(page, 'Run Demo')
  // Wait for completion -> app auto-advances to the Analyze page. Wait on the
  // DOM (what actually renders), not on window.__studioStore (a disconnected
  // store instance in this build).
  await page.locator('text=Validation Loss Convergence').first()
    .waitFor({ state: 'visible', timeout: 45000 })
  await page.waitForTimeout(3000)
  await shot(page, '04_analyze_convergence.png')
  // Probe whether the global store reflects the rendered tree.
  const probe = await page.evaluate(() => {
    try { return window.__studioStore.getState().step } catch { return 'no-store' }
  })
  console.log('[probe] window.__studioStore.step after train render =', probe)

  // ── Inference configure ──────────────────────────────────────────────────
  // Go back to Configure via the stepper, switch task to Inference, and pick a
  // curated inference prompt so Continue is enabled.
  await clickStep(page, 'CONFIGURE')
  await page.waitForTimeout(700)
  await clickText(page, 'Inference')
  await page.waitForTimeout(600)
  await page.locator('.prompt-card').filter({ hasText: /Iron-carbon|formation energy/i }).first().click({ timeout: 5000 })
  await page.waitForTimeout(500)
  await shot(page, '05_configure_infer.png')
  await clickText(page, 'Continue to Run')
  await page.waitForTimeout(800)

  // ── Per-material inference ────────────────────────────────────────────────
  // The demo is launched through the app's own "Run Demo" button so React sets
  // the store and renders the Analyze page (external store.setState does not
  // re-render in this build). The material is selected by writing an index to
  // the control file the backend reads (HG_DEMO_STRUCT_INDEX_FILE).
  const ctl = process.env.HG_DEMO_STRUCT_INDEX_FILE
  const materials = [
    { idx: 0, file: '06_energy_fes2.png',  name: 'Pyrite' },
    { idx: 1, file: '07_energy_feni3.png', name: 'Awaruite' },
    { idx: 2, file: '08_energy_nafeo2.png', name: 'Sodium ferrite' },
    { idx: 3, file: '09_energy_fe2h6.png', name: 'Iron hydride' },
  ]
  for (let k = 0; k < materials.length; k++) {
    const m = materials[k]
    if (ctl) fs.writeFileSync(ctl, String(m.idx))
    if (k === 0) {
      // First run: we're already on the Run page.
      await clickText(page, 'Run Demo')
    } else {
      // Return to Run via the stepper, reset, and run again with the new index.
      await clickStep(page, 'RUN')
      await page.waitForTimeout(500)
      await clickText(page, 'Reset')
      await page.waitForTimeout(400)
      await clickText(page, 'Run Demo')
    }
    // Wait on the DOM for this material's Analyze page to render, then screenshot.
    await page.locator(`text=${m.name}`).first()
      .waitFor({ state: 'visible', timeout: 30000 })
      .catch(() => console.log(`[warn] ${m.name} not visible after run`))
    await page.waitForTimeout(3200)  // molecule 2D render
    await shot(page, m.file)
  }

  await browser.close()
  console.log('[done] shots in', SHOTS)
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1) })
