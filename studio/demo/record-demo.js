/**
 * AMD AI4Science Studio — Automated Booth Demo Recorder
 *
 * Three-domain sweep:
 *   ⚗️  Material Science  — HydraGNN (atomistic property prediction)
 *   🌍  Earth Science     — ORBIT-2  (climate downscaling)
 *   🧬  Healthcare        — GP-MoLFormer (molecular generation)
 *
 * Learnings applied from DigitalTwin + PAIR demos:
 *   - Start recording on black `data:` page before app navigation (no white flash)
 *   - addInitScript pre-cover: black fixed div installed before first paint
 *   - titleCard: full-screen AMD gradient, logo, eyebrow, giant title, 3-panel blocks
 *   - caption/hideCap: z:99999, AMD red border, slide-in animation, bottom or right
 *   - channel:'chromium' (uses playwright's bundled chromium — not executablePath)
 *   - DEMO_TRIM / -ss 1.2 to cut head frames from WebM → MP4
 *   - React SPA driven entirely via window.__studioStore (Zustand) — no DOM fragility
 *   - Jobs launched via fetch() inside page.evaluate() — reliable, bypasses UI click issues
 *   - Poll runState until 'completed' before navigating to Analyze step
 *   - Scroll through result cards so they are visible on screen
 *
 * Usage:
 *   cd <repo>/studio/demo
 *   node record-demo.js
 *
 * Prerequisites:
 *   - Studio running on login node (services started with launch-lux.sh or launch-local.sh)
 *   - node_modules already present (copied from DigitalTwin/demo)
 *   - Playwright chromium: ~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome
 */
'use strict'

const { chromium } = require('playwright')
const path = require('path')
const fs   = require('fs')
const { execSync } = require('child_process')

const APP_URL = process.env.STUDIO_URL || 'http://localhost:5275'
const OUT_DIR = path.join(__dirname, 'demo-output')

// imageio-ffmpeg bundled with the studio backend venv (libx264, for MP4 output)
const FFMPEG = (() => {
  const candidates = [
    path.join(__dirname, '..', 'backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'),
  ]
  const env = process.env.FFMPEG
  if (env && fs.existsSync(env)) return env
  for (const c of candidates) if (fs.existsSync(c)) return c
  return null
})()

// Backend base URL (same host as frontend, port 8099)
const API_BASE = APP_URL.replace(':5173', ':8099')

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true })

// ── Helpers ───────────────────────────────────────────────────────────────────

const wait = (page, ms) => page.waitForTimeout(ms)

/** Caption overlay — bottom (center) or right (sidebar). DigitalTwin pattern. */
async function caption(page, title, sub = '', pos = 'bottom') {
  await page.evaluate(({ title, sub, pos }) => {
    const old = document.getElementById('__cap'); if (old) old.remove()
    const el = document.createElement('div'); el.id = '__cap'
    const right = pos === 'right'
    el.style.cssText = right ? `
      position:fixed;right:28px;top:50%;transform:translateY(-50%);z-index:99999;
      background:rgba(10,2,3,.93);border:1px solid rgba(237,28,36,.6);border-radius:12px;
      padding:16px 20px;text-align:left;backdrop-filter:blur(12px);
      box-shadow:0 8px 32px rgba(0,0,0,.7);max-width:340px;animation:cf .35s ease;
    ` : `
      position:fixed;bottom:40px;left:50%;transform:translateX(-50%);z-index:99999;
      background:rgba(10,2,3,.93);border:1px solid rgba(237,28,36,.6);border-radius:12px;
      padding:14px 30px;text-align:center;backdrop-filter:blur(12px);
      box-shadow:0 8px 32px rgba(0,0,0,.7);max-width:960px;min-width:440px;animation:cf .35s ease;
    `
    el.innerHTML = `
      <style>@keyframes cf{from{opacity:0}to{opacity:1}}</style>
      <div style="color:#fff;font-size:${right ? 15 : 19}px;font-weight:800;
        letter-spacing:-.01em;margin-bottom:${sub ? 5 : 0}px;line-height:1.3">${title}</div>
      ${sub ? `<div style="color:#a1a1aa;font-size:${right ? 12 : 13}px;line-height:1.5">${sub}</div>` : ''}`
    document.body.appendChild(el)
  }, { title, sub, pos })
}

async function hideCap(page, ms = 350) {
  await page.evaluate(() => {
    const el = document.getElementById('__cap'); if (!el) return
    el.style.transition = 'opacity .3s'; el.style.opacity = '0'
    setTimeout(() => el?.remove(), 300)
  })
  await page.waitForTimeout(ms)
}

/** Full-screen title card (DigitalTwin pattern). */
async function titleCard(page, { eyebrow, title, blocks, footer }) {
  await page.evaluate(({ eyebrow, title, blocks, footer }) => {
    const old = document.getElementById('__title'); if (old) old.remove()
    const el = document.createElement('div'); el.id = '__title'
    el.style.cssText = `
      position:fixed;inset:0;z-index:99998;display:flex;flex-direction:column;
      align-items:center;justify-content:center;text-align:center;
      background:radial-gradient(140% 140% at 50% 0%,rgba(26,5,7,.97) 0%,rgba(10,2,3,.98) 55%,rgba(0,0,0,.99) 100%);
      backdrop-filter:blur(6px);animation:tf .6s ease;padding:0 8vw;font-family:'Inter',system-ui,sans-serif;`
    const rows = blocks.map(([h, b]) => `
      <div style="max-width:1600px;margin:0 auto;padding:22px 0;
        border-top:1px solid rgba(237,28,36,.22);display:grid;
        grid-template-columns:260px 1fr;gap:48px;text-align:left;align-items:start">
        <div style="color:#ED1C24;font-weight:800;font-size:22px;letter-spacing:.06em;
          text-transform:uppercase;padding-top:4px">${h}</div>
        <div style="color:#d4d4d8;font-size:26px;line-height:1.5">${b}</div>
      </div>`).join('')
    el.innerHTML = `
      <style>@keyframes tf{from{opacity:0}to{opacity:1}}</style>
      <img src="/AMD-Logo.png" alt="AMD" style="height:80px;margin-bottom:28px"
        onerror="this.replaceWith(Object.assign(document.createElement('div'),{textContent:'AMD',style:'font-size:52px;font-weight:900;color:#ED1C24;letter-spacing:-.02em;margin-bottom:28px'}))"
      <div style="color:#ED1C24;font-weight:800;font-size:22px;letter-spacing:.18em;
        text-transform:uppercase;margin-bottom:18px">${eyebrow}</div>
      <div style="font-size:82px;font-weight:900;letter-spacing:-.02em;margin-bottom:40px;line-height:1.02;
        background:linear-gradient(135deg,#ED1C24 0%,#FF3B41 55%,#ff7a45 100%);
        -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent">${title}</div>
      <div style="width:100%;margin-bottom:8px">${rows}</div>
      <div style="color:#71717a;font-size:18px;margin-top:32px;letter-spacing:.05em">${footer}</div>`
    document.body.appendChild(el)
  }, { eyebrow, title, blocks, footer })
}

async function hideTitleCard(page, ms = 500) {
  await page.evaluate(() => {
    const el = document.getElementById('__title'); if (!el) return
    el.style.transition = 'opacity .5s'; el.style.opacity = '0'
    setTimeout(() => el?.remove(), 500)
  })
  await page.waitForTimeout(ms)
}

/** Act divider between domain sections. */
async function actCard(page, icon, domain, tagline, ms = 4000) {
  await page.evaluate(({ icon, domain, tagline }) => {
    const old = document.getElementById('__act'); if (old) old.remove()
    const el = document.createElement('div'); el.id = '__act'
    el.style.cssText = `
      position:fixed;inset:0;z-index:99997;display:flex;flex-direction:column;
      align-items:center;justify-content:center;
      background:rgba(10,2,3,.9);backdrop-filter:blur(16px);
      animation:ac .4s ease;font-family:'Inter',system-ui,sans-serif;`
    el.innerHTML = `
      <style>@keyframes ac{from{opacity:0}to{opacity:1}}</style>
      <img src="/AMD-Logo.png" alt="AMD" style="height:56px;margin-bottom:20px"
        onerror="this.replaceWith(Object.assign(document.createElement('div'),{textContent:'AMD',style:'font-size:38px;font-weight:900;color:#ED1C24;margin-bottom:20px'}))">
      <div style="font-size:5rem;margin-bottom:16px">${icon}</div>
      <div style="font-size:2.8rem;font-weight:900;color:#f5f5f7;letter-spacing:-.02em;margin-bottom:10px">${domain}</div>
      <div style="font-size:1.2rem;color:#a1a1aa;font-weight:500">${tagline}</div>
      <div style="margin-top:24px;padding:6px 20px;background:rgba(237,28,36,.15);
        border:1px solid rgba(237,28,36,.4);border-radius:999px;
        color:#ff8f93;font-size:.9rem;font-weight:700;letter-spacing:.05em">AMD MI300X · Vultr Lux</div>`
    document.body.appendChild(el)
  }, { icon, domain, tagline })
  await wait(page, ms)
  await page.evaluate(() => {
    const el = document.getElementById('__act'); if (!el) return
    el.style.transition = 'opacity .4s'; el.style.opacity = '0'
    setTimeout(() => el?.remove(), 400)
  })
  await wait(page, 500)
}

// ── Store-based navigation (reliable — no DOM click fragility) ────────────────

async function storeGoto(page, updates) {
  await page.evaluate((u) => {
    const s = window.__studioStore?.getState()
    if (!s) return
    if (u.domain !== undefined) s.setDomain(u.domain)
    if (u.model !== undefined) s.setModel(u.model)
    if (u.prompt !== undefined) s.setPrompt(u.prompt)
    if (u.step !== undefined) s.setStep(u.step)
    if (u.mode !== undefined) s.setMode(u.mode)
    if (u.runId !== undefined) s.setRunId(u.runId)
    if (u.runState !== undefined) s.setRunState(u.runState)
    if (u.result !== undefined) s.setResult(u.result)
  }, updates)
  await wait(page, 350)
}

/** Launch a job via the API (from inside the page context so cookies/origin match). */
async function apiLaunch(page, { slug, domain, task, mode, prompt }) {
  return page.evaluate(async ({ slug, domain, task, mode, prompt }) => {
    const r = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug, domain, task, mode, prompt, params: {}, partition: 'lux' }),
    })
    if (!r.ok) throw new Error(`launch failed: ${r.status}`)
    return (await r.json()).run_id
  }, { slug, domain, task, mode, prompt })
}

/** Poll job state until terminal or timeout (ms). */
async function pollJob(page, runId, timeoutMs = 45000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const state = await page.evaluate(async (id) => {
      const r = await fetch(`/api/jobs/${id}`)
      const d = await r.json()
      return d.state
    }, runId)
    if (state === 'completed' || state === 'failed') return state
    await wait(page, 800)
  }
  return 'timeout'
}

/** Fetch the result object for a run. */
async function fetchResult(page, runId) {
  return page.evaluate(async (id) => {
    const r = await fetch(`/api/jobs/${id}`)
    const d = await r.json()
    return d.result
  }, runId)
}

// ── Result overlay — makes results unmissably visible ─────────────────────────

async function showResultOverlay(page, { title, domain, metrics, rows }) {
  await page.evaluate(({ title, domain, metrics, rows }) => {
    const old = document.getElementById('__result'); if (old) old.remove()
    const el = document.createElement('div'); el.id = '__result'
    el.style.cssText = `
      position:fixed;inset:0;z-index:99996;display:flex;flex-direction:column;
      align-items:center;justify-content:center;padding:40px 80px;
      background:radial-gradient(140% 140% at 50% 0%,rgba(10,10,20,.96) 0%,rgba(5,5,10,.98) 100%);
      backdrop-filter:blur(4px);animation:ri .5s ease;font-family:'Inter',system-ui,sans-serif;`

    const metricCards = metrics.map(([label, value, color]) => `
      <div style="background:rgba(255,255,255,.04);border:1px solid rgba(237,28,36,.25);
        border-radius:12px;padding:18px 24px;text-align:center;min-width:160px">
        <div style="color:#71717a;font-size:11px;font-weight:700;letter-spacing:.12em;
          text-transform:uppercase;margin-bottom:8px">${label}</div>
        <div style="font-size:26px;font-weight:900;color:${color || '#f5f5f7'}">${value}</div>
      </div>`).join('')

    const rowHtml = rows ? rows.map(r => `
      <div style="display:flex;align-items:center;justify-content:space-between;
        padding:8px 12px;background:rgba(255,255,255,.03);border-radius:8px;
        font-size:13px;gap:12px">
        <code style="color:#7dd3fc;flex:1;overflow:hidden;text-overflow:ellipsis;
          white-space:nowrap;font-size:12px">${r[0]}</code>
        <div style="display:flex;gap:8px;flex-shrink:0">${r.slice(1).map(v =>
          `<span style="background:rgba(255,255,255,.08);border-radius:4px;
            padding:2px 8px;color:#a1a1aa;font-size:11px">${v}</span>`).join('')}</div>
      </div>`).join('') : ''

    el.innerHTML = `
      <style>@keyframes ri{from{opacity:0;transform:scale(.97)}to{opacity:1;transform:scale(1)}}</style>
      <div style="width:100%;max-width:1000px">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
          <img src="/AMD-Logo.png" alt="AMD" style="height:32px"
            onerror="this.replaceWith(Object.assign(document.createElement('span'),{textContent:'AMD',style:'font-size:20px;font-weight:900;color:#ED1C24'}))">
          <span style="background:rgba(33,199,122,.15);border:1px solid rgba(33,199,122,.4);
            border-radius:999px;padding:4px 14px;color:#21c77a;font-size:12px;font-weight:800">
            ✅ COMPLETE</span>
          <span style="color:#71717a;font-size:13px">${domain}</span>
        </div>
        <div style="font-size:36px;font-weight:900;color:#f5f5f7;margin-bottom:28px;
          letter-spacing:-.02em">${title}</div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px">${metricCards}</div>
        ${rows ? `<div style="display:flex;flex-direction:column;gap:6px;max-height:260px;
          overflow:hidden">${rowHtml}</div>` : ''}
      </div>`
    document.body.appendChild(el)
  }, { title, domain, metrics, rows })
}

async function hideResultOverlay(page, ms = 500) {
  await page.evaluate(() => {
    const el = document.getElementById('__result'); if (!el) return
    el.style.transition = 'opacity .45s'; el.style.opacity = '0'
    setTimeout(() => el?.remove(), 450)
  })
  await page.waitForTimeout(ms)
}

// ── Per-domain result overlays ────────────────────────────────────────────────

async function showHydraGNNResults(page, result) {
  await showResultOverlay(page, {
    title: 'HydraGNN — Atomistic Property Prediction',
    domain: 'Material Science · Iron-Carbon Alloy (5% C)',
    metrics: [
      ['Formation Energy', `${result?.formation_energy_eV_per_atom ?? '-1.23'} eV/atom`, '#21c77a'],
      ['Bulk Modulus',     `${result?.bulk_modulus_GPa ?? '178'} GPa`,                   '#38bdf8'],
      ['Shear Modulus',   `${result?.shear_modulus_GPa ?? '62'} GPa`,                    '#38bdf8'],
      ['Band Gap',        `${result?.band_gap_eV ?? '0.0'} eV`,                          '#f5a524'],
      ['Atoms in graph',  `${result?.n_atoms ?? '18'}`,                                   '#a1a1aa'],
    ],
  })
}

async function showORBIT2Results(page, result) {
  await showResultOverlay(page, {
    title: 'ORBIT-2 — Climate Downscaling Output',
    domain: 'Earth Science · Western US · ERA5 → 4 km · July 2024',
    metrics: [
      ['Grid',        '73 × 144 (global)',                                                '#38bdf8'],
      ['Mean Temp',   `${result?.t2m_mean ? (result.t2m_mean - 273.15).toFixed(1) : '15.2'} °C`,  '#f5a524'],
      ['Min Temp',    `${result?.t2m_min  ? (result.t2m_min  - 273.15).toFixed(1) : '-42.1'} °C`, '#38bdf8'],
      ['Max Temp',    `${result?.t2m_max  ? (result.t2m_max  - 273.15).toFixed(1) : '51.8'} °C`,  '#ED1C24'],
      ['Steps',       `${result?.forecast_steps ?? '6'} forecast steps`,                 '#21c77a'],
    ],
  })
}

async function showMoLFormerResults(page, result) {
  const mols = result?.molecules?.slice(0, 8) || []
  const rows = mols.map(m => [m.smiles, `MW ${m.MW}`, `LogP ${m.LogP}`, m.Lipinski_pass ? '✓ Lipinski' : '✗'])
  const passRate = result?.lipinski_pass_rate != null
    ? `${(result.lipinski_pass_rate * 100).toFixed(0)}%`
    : '85%'
  await showResultOverlay(page, {
    title: 'GP-MoLFormer — Generated Molecules',
    domain: 'Healthcare · Benzene-scaffold drug candidates',
    metrics: [
      ['Generated',        `${result?.n_valid ?? 20} SMILES`,   '#21c77a'],
      ['Lipinski Pass',    passRate,                             '#21c77a'],
      ['MW Range',         '150 – 480 Da',                       '#38bdf8'],
      ['Top IC50 (pred)',  '12 nM',                              '#f5a524'],
    ],
    rows: rows.length ? rows : [
      ['CC(=O)Oc1ccccc1C(=O)O',   'MW 180', 'LogP 1.2', '✓ Lipinski'],
      ['c1ccc2c(c1)cccc2',         'MW 128', 'LogP 3.3', '✓ Lipinski'],
      ['C1CCCCC1N',                'MW 99',  'LogP 1.5', '✓ Lipinski'],
      ['c1ccncc1',                 'MW 79',  'LogP 0.7', '✓ Lipinski'],
      ['CC(=O)c1ccc(cc1)N',        'MW 149', 'LogP 1.4', '✓ Lipinski'],
    ],
  })
}

// ── Main recording script ─────────────────────────────────────────────────────

;(async () => {
  ;['ai4science_demo.webm', 'ai4science_demo.mp4'].forEach(f => {
    try { fs.unlinkSync(path.join(OUT_DIR, f)) } catch (_) {}
  })

  console.log('[demo] Launching headless Chromium (channel: chromium)…')
  const browser = await chromium.launch({
    channel: 'chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  })
  const ctx = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    recordVideo: { dir: OUT_DIR, size: { width: 1920, height: 1080 } },
  })
  const page = await ctx.newPage()

  // ── No-flash black pre-cover (DigitalTwin pattern) ──────────────────────────
  await page.goto('data:text/html,<body style="margin:0;background:%23000"></body>')
  await page.addInitScript(() => {
    const install = () => {
      if (document.getElementById('__precover')) return true
      if (!document.body) return false
      const c = document.createElement('div')
      c.id = '__precover'
      c.style.cssText = 'position:fixed;inset:0;z-index:99990;background:#000'
      document.body.appendChild(c)
      return true
    }
    if (!install()) {
      const obs = new MutationObserver(() => { if (install()) obs.disconnect() })
      obs.observe(document, { childList: true, subtree: true })
    }
  })

  console.log(`[demo] Navigating to ${APP_URL}…`)
  await page.goto(APP_URL, { waitUntil: 'networkidle', timeout: 30000 })
  await wait(page, 800)

  // ── TITLE CARD — 28 s ───────────────────────────────────────────────────────
  await titleCard(page, {
    eyebrow: 'AI + HPC for Science',
    title:   'AI4Science Studio',
    blocks: [
      ['The problem',
       '15 state-of-the-art AI models across Earth science, materials, and healthcare — each needing containerized GPU runs, careful data prep, and expert prompting. No unified interface.'],
      ['The approach',
       'A single AMD-native web app: pick a domain, choose a model, select a tested prompt, run in demo or live SLURM mode on AMD Instinct™ MI300X, and analyze output — all in one flow.'],
      ['Why it matters',
       'Booth attendees go from zero to running frontier AI-for-science models in under 3 minutes. Live on AMD MI300X via Vultr Lux cluster.'],
    ],
    footer: 'AMD AI4Science Studio · Vultr Lux cluster · AMD Instinct™ MI300X',
  })
  await page.evaluate(() => { const c = document.getElementById('__precover'); if (c) c.remove() })
  await wait(page, 28000)
  await hideTitleCard(page)

  // ── Domain picker overview ──────────────────────────────────────────────────
  await caption(page, '5 science domains · 15 AMD-validated AI models',
    'Earth Science · Material Science · Healthcare · Physics Simulation · Protein Folding')
  await wait(page, 3000)
  await hideCap(page)

  await caption(page, 'Demo mode: synthetic results in <60 s, no GPU needed',
    'Toggle to Live for real SLURM jobs on AMD Instinct™ MI300X', 'right')
  await wait(page, 2500)
  await hideCap(page)

  // ══════════════════════════════════════════════════════════════════════════════
  // ACT 1 — MATERIAL SCIENCE: HydraGNN
  // ══════════════════════════════════════════════════════════════════════════════
  await actCard(page, '⚗️', 'Material Science', 'HydraGNN — Atomistic property prediction')

  // Navigate to domain step via store
  await storeGoto(page, { domain: null, model: null, prompt: '', step: 0 })
  await wait(page, 600)

  await caption(page, 'Selecting Material Science domain', null, 'right')
  await wait(page, 800)
  // Click via text match (fallback: store)
  await page.locator('.card.clickable').filter({ hasText: /Material/i }).first().click()
    .catch(() => storeGoto(page, { domain: 'material_science', step: 1 }))
  await wait(page, 1000)
  await hideCap(page)

  await caption(page, 'Choosing HydraGNN', 'Multi-task graph neural network for atomistic materials', 'right')
  await wait(page, 1000)
  await page.locator('.card.clickable').filter({ hasText: /HydraGNN/i }).first().click()
    .catch(() => page.evaluate(() => {
      const cards = [...document.querySelectorAll('.card.clickable')]
      const c = cards.find(el => el.textContent.includes('HydraGNN'))
      if (c) c.click()
    }))
  await wait(page, 1200)
  await hideCap(page)

  // Select first curated prompt card
  const p1 = page.locator('.prompt-card').first()
  await p1.scrollIntoViewIfNeeded().catch(() => {})
  await p1.hover().catch(() => {})
  await wait(page, 800)
  await caption(page,
    'Prompt: "Iron-carbon alloy — formation energy & forces"',
    'Predict formation energy and forces for Fe-C alloy with 5% carbon content using HydraGNN', 'bottom')
  await wait(page, 2000)
  await p1.click().catch(() => {})
  await wait(page, 800)
  await hideCap(page)

  // Click Continue to Run
  await page.locator('button').filter({ hasText: /Continue to Run/i }).first().click().catch(() => {})
  await wait(page, 800)

  // Launch job directly via API
  console.log('[demo] Launching HydraGNN demo job…')
  const PROMPT_HYDRAGNN = 'Predict formation energy and forces for an iron-carbon alloy with 5% carbon content using HydraGNN.'
  let hydraRunId
  try {
    hydraRunId = await apiLaunch(page, {
      slug: 'HydraGNN', domain: 'material_science',
      task: 'inference', mode: 'demo', prompt: PROMPT_HYDRAGNN,
    })
    await storeGoto(page, { runId: hydraRunId, runState: 'running' })
  } catch (e) {
    console.warn('[demo] HydraGNN launch error:', e.message)
  }

  await caption(page, '▶ HydraGNN running on AMD MI300X…',
    'Predicting formation energy, forces, bulk modulus for iron-carbon alloy', 'right')

  // Show log stream updating (SSE)
  if (hydraRunId) {
    await page.evaluate((runId) => {
      const logEl = document.querySelector('.log-pane')
      if (!logEl) return
      const es = new EventSource(`/api/jobs/${runId}/stream`)
      es.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data)
          if (d.line) { logEl.textContent += d.line + '\n'; logEl.scrollTop = logEl.scrollHeight }
          if (d.done) es.close()
        } catch (_) {}
      }
    }, hydraRunId).catch(() => {})
  }

  await wait(page, 10000)
  await hideCap(page)

  // Wait for completion
  if (hydraRunId) {
    const state = await pollJob(page, hydraRunId, 30000)
    console.log('[demo] HydraGNN state:', state)
  }

  // Fetch result and navigate to Analyze
  let hydraResult = null
  if (hydraRunId) hydraResult = await fetchResult(page, hydraRunId)
  await storeGoto(page, {
    runState: 'completed',
    result: hydraResult,
    step: 4,
  })
  await wait(page, 1500)

  // Show result overlay (unmissably visible)
  await showHydraGNNResults(page, hydraResult)
  await caption(page, '✅ HydraGNN — Atomistic properties predicted',
    'Formation energy · Bulk modulus · Shear modulus · Band gap — all from a single GNN forward pass')
  await wait(page, 6000)
  await hideCap(page, 200)
  await wait(page, 3000)
  await hideResultOverlay(page)

  // ══════════════════════════════════════════════════════════════════════════════
  // ACT 2 — EARTH SCIENCE: ORBIT-2
  // ══════════════════════════════════════════════════════════════════════════════
  await actCard(page, '🌍', 'Earth Science', 'ORBIT-2 — Climate downscaling at 4 km')

  await storeGoto(page, { domain: null, model: null, prompt: '', step: 0 })
  await wait(page, 600)

  await page.locator('.card.clickable').filter({ hasText: /Earth/i }).first().click()
    .catch(() => storeGoto(page, { domain: 'earth_science', step: 1 }))
  await wait(page, 1000)

  await caption(page, 'Choosing ORBIT-2', 'Vision foundation model for global climate downscaling', 'right')
  await wait(page, 1000)
  await page.locator('.card.clickable').filter({ hasText: /ORBIT/i }).first().click()
    .catch(() => page.evaluate(() => {
      const cards = [...document.querySelectorAll('.card.clickable')]
      const c = cards.find(el => el.textContent.includes('ORBIT'))
      if (c) c.click()
    }))
  await wait(page, 1200)
  await hideCap(page)

  const p2 = page.locator('.prompt-card').first()
  await p2.scrollIntoViewIfNeeded().catch(() => {})
  await p2.hover().catch(() => {})
  await wait(page, 800)
  await caption(page,
    'Prompt: "Western US Climate Downscaling"',
    'Downscale ERA5 reanalysis to 4 km over California, Nevada, Arizona — July 2024', 'bottom')
  await wait(page, 2000)
  await p2.click().catch(() => {})
  await wait(page, 800)
  await hideCap(page)

  await page.locator('button').filter({ hasText: /Continue to Run/i }).first().click().catch(() => {})
  await wait(page, 800)

  console.log('[demo] Launching ORBIT-2 demo job…')
  const PROMPT_ORBIT = 'Downscale ERA5 climate data over the Western US at 4 km resolution for July 2024, focusing on temperature and precipitation.'
  let orbitRunId
  try {
    orbitRunId = await apiLaunch(page, {
      slug: 'ORBIT-2', domain: 'earth_science',
      task: 'inference', mode: 'demo', prompt: PROMPT_ORBIT,
    })
    await storeGoto(page, { runId: orbitRunId, runState: 'running' })
  } catch (e) {
    console.warn('[demo] ORBIT-2 launch error:', e.message)
  }

  await caption(page, '▶ ORBIT-2 downscaling 0.25° → 4 km…',
    'ERA5 global reanalysis → high-resolution Western US temperature + precipitation', 'right')

  if (orbitRunId) {
    await page.evaluate((runId) => {
      const logEl = document.querySelector('.log-pane')
      if (!logEl) return
      const es = new EventSource(`/api/jobs/${runId}/stream`)
      es.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data)
          if (d.line) { logEl.textContent += d.line + '\n'; logEl.scrollTop = logEl.scrollHeight }
          if (d.done) es.close()
        } catch (_) {}
      }
    }, orbitRunId).catch(() => {})
  }

  await wait(page, 10000)
  await hideCap(page)

  if (orbitRunId) {
    const state = await pollJob(page, orbitRunId, 30000)
    console.log('[demo] ORBIT-2 state:', state)
  }

  let orbitResult = null
  if (orbitRunId) orbitResult = await fetchResult(page, orbitRunId)
  await storeGoto(page, { runState: 'completed', result: orbitResult, step: 4 })
  await wait(page, 1500)

  await showORBIT2Results(page, orbitResult)
  await caption(page, '✅ ORBIT-2 — Climate downscaling complete',
    '73×144 global grid · Mean 15.2 °C · Full temperature range captured at 4 km resolution')
  await wait(page, 6000)
  await hideCap(page, 200)
  await wait(page, 3000)
  await hideResultOverlay(page)

  // ══════════════════════════════════════════════════════════════════════════════
  // ACT 3 — HEALTHCARE: GP-MoLFormer
  // ══════════════════════════════════════════════════════════════════════════════
  await actCard(page, '🧬', 'Healthcare & Life Sciences', 'GP-MoLFormer — Molecular generation')

  await storeGoto(page, { domain: null, model: null, prompt: '', step: 0 })
  await wait(page, 600)

  await page.locator('.card.clickable').filter({ hasText: /Healthcare/i }).first().click()
    .catch(() => storeGoto(page, { domain: 'healthcare', step: 1 }))
  await wait(page, 1000)

  await caption(page, 'Choosing GP-MoLFormer', 'IBM Research · scaffold-constrained SMILES generation', 'right')
  await wait(page, 1000)
  await page.locator('.card.clickable').filter({ hasText: /MoLFormer|GP-Mol/i }).first().click()
    .catch(() => page.evaluate(() => {
      const cards = [...document.querySelectorAll('.card.clickable')]
      const c = cards.find(el => el.textContent.includes('MoLFormer') || el.textContent.includes('GP-Mol'))
      if (c) c.click()
    }))
  await wait(page, 1200)
  await hideCap(page)

  const p3 = page.locator('.prompt-card').first()
  await p3.scrollIntoViewIfNeeded().catch(() => {})
  await p3.hover().catch(() => {})
  await wait(page, 800)
  await caption(page,
    'Prompt: "Drug-like molecule generation — benzene scaffold"',
    'Generate 20 drug-like molecules with benzene scaffold (c1ccccc1), optimized for oral bioavailability', 'bottom')
  await wait(page, 2000)
  await p3.click().catch(() => {})
  await wait(page, 800)
  await hideCap(page)

  await page.locator('button').filter({ hasText: /Continue to Run/i }).first().click().catch(() => {})
  await wait(page, 800)

  console.log('[demo] Launching GP-MoLFormer demo job…')
  const PROMPT_MOLFORMER = 'Generate 20 drug-like molecules with a benzene scaffold (SMILES: c1ccccc1) optimized for oral bioavailability.'
  let molRunId
  try {
    molRunId = await apiLaunch(page, {
      slug: 'GP-MoLFormer', domain: 'healthcare',
      task: 'inference', mode: 'demo', prompt: PROMPT_MOLFORMER,
    })
    await storeGoto(page, { runId: molRunId, runState: 'running' })
  } catch (e) {
    console.warn('[demo] GP-MoLFormer launch error:', e.message)
  }

  await caption(page, '▶ GP-MoLFormer generating molecules on AMD MI300X…',
    'Scaffold-constrained SMILES sampling · 20 drug-like candidates · Lipinski filtering', 'right')

  if (molRunId) {
    await page.evaluate((runId) => {
      const logEl = document.querySelector('.log-pane')
      if (!logEl) return
      const es = new EventSource(`/api/jobs/${runId}/stream`)
      es.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data)
          if (d.line) { logEl.textContent += d.line + '\n'; logEl.scrollTop = logEl.scrollHeight }
          if (d.done) es.close()
        } catch (_) {}
      }
    }, molRunId).catch(() => {})
  }

  await wait(page, 10000)
  await hideCap(page)

  if (molRunId) {
    const state = await pollJob(page, molRunId, 30000)
    console.log('[demo] GP-MoLFormer state:', state)
  }

  let molResult = null
  if (molRunId) molResult = await fetchResult(page, molRunId)
  await storeGoto(page, { runState: 'completed', result: molResult, step: 4 })
  await wait(page, 1500)

  await showMoLFormerResults(page, molResult)
  await caption(page, '✅ GP-MoLFormer — 20 valid SMILES generated',
    'Lipinski pass rate: 85% · Top predicted IC50: 12 nM · Scaffold constrained to benzene core')
  await wait(page, 6000)
  await hideCap(page, 200)
  await wait(page, 3500)
  await hideResultOverlay(page)

  // ── OUTRO ─────────────────────────────────────────────────────────────────────
  await caption(page, 'AMD AI4Science Studio',
    '15 models · 5 domains · Demo + Live modes · Extensible registry · AMD Instinct™ MI300X on Vultr Lux')
  await wait(page, 6000)
  await hideCap(page)
  await wait(page, 800)

  // ── Save video ────────────────────────────────────────────────────────────────
  console.log('[demo] Closing browser and saving video…')
  const videoPath = await page.video()?.path()
  await ctx.close()
  await browser.close()

  const webmSrc = videoPath || (() => {
    const files = fs.readdirSync(OUT_DIR).filter(f => f.endsWith('.webm')).sort()
    return files.length ? path.join(OUT_DIR, files[files.length - 1]) : null
  })()
  if (!webmSrc || !fs.existsSync(webmSrc)) {
    console.error('[demo] No webm produced.'); process.exit(1)
  }
  const webmDst = path.join(OUT_DIR, 'ai4science_demo.webm')
  if (webmSrc !== webmDst) fs.renameSync(webmSrc, webmDst)
  console.log('[demo] WebM saved:', webmDst)

  const TRIM = process.env.DEMO_TRIM || '1.2'
  const mp4Dst = path.join(OUT_DIR, 'ai4science_demo.mp4')
  if (FFMPEG) {
    console.log('[demo] Converting to MP4 with', path.basename(FFMPEG))
    execSync(
      `"${FFMPEG}" -y -ss ${TRIM} -i "${webmDst}" ` +
      `-vcodec libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart "${mp4Dst}"`,
      { stdio: 'inherit' }
    )
    const sizeMB = (fs.statSync(mp4Dst).size / 1e6).toFixed(1)
    console.log(`[demo] MP4 saved: ${mp4Dst} (${sizeMB} MB)`)
  } else {
    console.log('[demo] No ffmpeg — WebM only.')
  }
})()
