// Capture SHOT_TARGET to SHOT_OUT. Run from studio/demo (playwright resolves here).
// Invoked by shot-on-compute.sh on a compute node with chrome libs staged.
const { chromium } = require('playwright');
const T = process.env.SHOT_TARGET;
const OUT = process.env.SHOT_OUT;
const W = parseInt(process.env.SHOT_W || '1520', 10);
const H = parseInt(process.env.SHOT_H || '940', 10);
(async () => {
  const b = await chromium.launch({ channel: 'chromium', headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'] });
  const ctx = await b.newContext({ viewport: { width: W, height: H },
    deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  await p.goto(T, { waitUntil: 'networkidle' });
  await p.waitForTimeout(1000);
  await p.screenshot({ path: OUT });
  await b.close();
  console.log('[shot] wrote', OUT);
})().catch(e => { console.error('[shot] FAIL', e.message); process.exit(1); });
