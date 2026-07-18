/* Molten-Salt Blanket Digital Twin — live dashboard logic.
 *
 * The reduced 1D two-group slab model is ported to JS (identical formula
 * to twin/reduced_model.py) so the twin runs entirely in the browser. The
 * calibrated group constants, the learned delta-net residual (as a small
 * lookup/interp grid), and the OpenMC ground-truth points are baked into
 * twin_data.js by twin/export_web_data.py. If the delta grid is absent the
 * dashboard degrades gracefully to the calibrated reduced model alone.
 */
const D = window.TWIN_DATA;
const BAND = D.band_rel;               // honest 1D-vs-3D band (0.30)
const C = D.constants;                 // calibrated group constants
const MULTS = ["none", "Be", "Pb"];

let state = { li6: 0.30, tbl: 1.0, mult: "Be" };

// ---- reduced model (matches reduced_model.tbr_slab, two-channel) ----
function tbrReduced(tbl, li6, mult) {
  const e = Math.max(0, Math.min(1, li6));
  const teff = C.t_eff_fraction * tbl;
  const atten = 1 - Math.exp(-C.Sigma6_eff_100pct_per_m * teff);
  const f7 = C.f7_fast || 0;                 // fast 7Li floor
  const kpar = C.k_par_li6 || 1e-3;          // 6Li thermal half-saturation
  const thermal6 = (e / (e + kpar)) * atten;
  return C.f_mult[mult] * (f7 + thermal6) * (1 - C.f_loss);
}

// ---- delta-net residual (bilinear interp over baked grid) ----
function deltaResid(tbl, li6, mult) {
  const g = D.delta_grid && D.delta_grid[mult];
  if (!g) return 0;
  const { li6_axis, t_axis, resid } = g;
  const fx = clampIdx(li6, li6_axis), fy = clampIdx(tbl, t_axis);
  const x0 = fx.i, x1 = Math.min(fx.i + 1, li6_axis.length - 1), tx = fx.f;
  const y0 = fy.i, y1 = Math.min(fy.i + 1, t_axis.length - 1), ty = fy.f;
  const a = resid[y0][x0], b = resid[y0][x1], c = resid[y1][x0], d = resid[y1][x1];
  return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
}
function clampIdx(v, axis) {
  if (v <= axis[0]) return { i: 0, f: 0 };
  if (v >= axis[axis.length - 1]) return { i: axis.length - 2, f: 1 };
  for (let i = 0; i < axis.length - 1; i++)
    if (v >= axis[i] && v < axis[i + 1])
      return { i, f: (v - axis[i]) / (axis[i + 1] - axis[i]) };
  return { i: axis.length - 2, f: 1 };
}
function tbrTwin(tbl, li6, mult) { return tbrReduced(tbl, li6, mult) + deltaResid(tbl, li6, mult); }

// ---- charts ----
const AMD = "#ED1C24", AMDB = "#FF3B41", INFO = "#38bdf8", OK = "#21c77a";
Chart.defaults.color = "#a1a1aa";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.font.size = 16;
Chart.defaults.borderColor = "rgba(39,39,42,.6)";

// shared axis styling: bigger fonts, ~half the tick labels
const TICK = { font: { size: 15 } };
const AXT = { font: { size: 17, weight: "600" } };

function sweepSeries(mult, li6) {
  const xs = [], twin = [], lo = [], hi = [];
  for (let t = 0.3; t <= 1.5001; t += 0.05) {
    const v = tbrTwin(t, li6, mult);
    xs.push(+t.toFixed(2)); twin.push(v); lo.push(v * (1 - BAND)); hi.push(v * (1 + BAND));
  }
  return { xs, twin, lo, hi };
}
function openmcPoints(mult, li6) {
  return D.openmc.filter(r => r.multiplier === mult && Math.abs(r.li6 - li6) < 0.03)
    .map(r => ({ x: r.t_blanket, y: r.tbr }));
}

let sweepChart, fidChart, optChart;
function initCharts() {
  const s = sweepSeries(state.mult, state.li6);
  sweepChart = new Chart(document.getElementById("sweepChart"), {
    data: {
      labels: s.xs,
      datasets: [
        { type: "line", label: "band hi", data: s.hi, borderWidth: 0, pointRadius: 0,
          backgroundColor: "rgba(237,28,36,.10)", fill: "+1" },
        { type: "line", label: "band lo", data: s.lo, borderWidth: 0, pointRadius: 0, fill: false },
        { type: "line", label: "Twin TBR", data: s.twin, borderColor: AMD, borderWidth: 3,
          pointRadius: 0, tension: .25 },
        { type: "scatter", label: "OpenMC", data: openmcPoints(state.mult, state.li6),
          borderColor: INFO, backgroundColor: INFO, pointRadius: 5 },
      ],
    },
    options: {
      maintainAspectRatio: false, animation: { duration: 300 },
      scales: {
        x: { title: { display: true, text: "blanket thickness (m)", ...AXT },
          ticks: { ...TICK, callback(v, i) { return i % 2 === 0 ? this.getLabelForValue(v) : ""; } } },
        y: { title: { display: true, text: "TBR", ...AXT }, suggestedMin: 0.6, suggestedMax: 1.8,
          ticks: { ...TICK, maxTicksLimit: 4 } } },
      plugins: { legend: { labels: { filter: i => !i.text.startsWith("band") } },
        annotation: false },
    },
    plugins: [reqLinePlugin],
  });

  fidChart = new Chart(document.getElementById("fidChart"), {
    type: "scatter",
    data: {
      datasets: [
        { label: "reduced (prior)", data: D.fidelity.reduced, backgroundColor: "#a1a1aa", pointRadius: 4 },
        { label: "twin (calibrated)", data: D.fidelity.twin, backgroundColor: AMD, pointRadius: 4 },
        { label: "ideal", type: "line", data: D.fidelity.ideal, borderColor: OK,
          borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
      ],
    },
    options: { maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "OpenMC TBR", ...AXT }, ticks: { ...TICK, maxTicksLimit: 5 } },
        y: { title: { display: true, text: "model TBR", ...AXT }, ticks: { ...TICK, maxTicksLimit: 5 } } } },
  });

  optChart = new Chart(document.getElementById("optChart"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "best TBR so far", data: [], borderColor: AMD, borderWidth: 2.5, pointRadius: 0, tension: .2 },
      { label: "required (1.05)", data: [], borderColor: OK, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
    ] },
    options: { maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "iteration", ...AXT }, ticks: { ...TICK, maxTicksLimit: 6 } },
        y: { title: { display: true, text: "TBR", ...AXT }, suggestedMin: 0.8, suggestedMax: 1.6,
          ticks: { ...TICK, maxTicksLimit: 5 } } } },
  });
}

// draw the required-TBR line on the sweep chart
const reqLinePlugin = {
  id: "reqLine",
  afterDraw(chart) {
    const { ctx, chartArea: a, scales } = chart;
    const y = scales.y.getPixelForValue(D.tbr_required);
    ctx.save(); ctx.strokeStyle = OK; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke();
    ctx.fillStyle = OK; ctx.font = "14px Inter";
    ctx.fillText("required " + D.tbr_required, a.left + 6, y - 5); ctx.restore();
  },
};

// ---- live update ----
function update() {
  const tbr = tbrTwin(state.tbl, state.li6, state.mult);
  const lo = tbr * (1 - BAND), hi = tbr * (1 + BAND);
  document.getElementById("tbrbig").textContent = tbr.toFixed(2);
  document.getElementById("bandtxt").textContent = `band ${lo.toFixed(2)} – ${hi.toFixed(2)}`;
  const v = document.getElementById("verdict");
  if (lo >= D.tbr_required)
    v.innerHTML = `<span class="badge badge-ok">SELF-SUFFICIENT</span>
      <div class="text-sm text-slate-400 mt-2">lower band edge ≥ required</div>`;
  else if (tbr >= D.tbr_required)
    v.innerHTML = `<span class="badge badge-warn">MARGINAL</span>
      <div class="text-sm text-slate-400 mt-2">point value OK, band edge below required</div>`;
  else
    v.innerHTML = `<span class="badge" style="background:rgba(237,28,36,.16);color:#ff8f93;border:1px solid rgba(237,28,36,.45)">SUB-BREEDING</span>
      <div class="text-sm text-slate-400 mt-2">increase enrichment / thickness / multiplier</div>`;

  const s = sweepSeries(state.mult, state.li6);
  sweepChart.data.labels = s.xs;
  sweepChart.data.datasets[0].data = s.hi;
  sweepChart.data.datasets[1].data = s.lo;
  sweepChart.data.datasets[2].data = s.twin;
  sweepChart.data.datasets[3].data = openmcPoints(state.mult, state.li6);
  sweepChart.update();
}

// ---- optimization animation (replays baked DE history) ----
function runOptimize() {
  const out = document.getElementById("optout");
  const rec = D.optimization.recommended;
  const hist = (D.optimization.per_multiplier[state.mult] || rec).history || [];
  optChart.data.labels = []; optChart.data.datasets[0].data = []; optChart.data.datasets[1].data = [];
  let best = -1, i = 0;
  const timer = setInterval(() => {
    if (i >= hist.length) {
      clearInterval(timer);
      out.innerHTML = `<span class="badge badge-ok">RECOMMENDED</span> ${rec.multiplier} multiplier ·
        Li-6 <b>${(rec.li6_opt * 100).toFixed(1)}%</b> · blanket <b>${rec.t_blanket_opt.toFixed(2)} m</b>
        → TBR <b>${rec.tbr_opt.toFixed(3)}</b> (lo ${rec.tbr_lo_opt.toFixed(3)})`;
      // apply recommended design to the sliders
      setLi6(rec.li6_opt); setTbl(rec.t_blanket_opt); setMult(rec.multiplier);
      return;
    }
    best = Math.max(best, hist[i].tbr);
    optChart.data.labels.push(i);
    optChart.data.datasets[0].data.push(best);
    optChart.data.datasets[1].data.push(D.tbr_required);
    optChart.update();
    i++;
  }, 90);
}

// ---- controls wiring ----
function setLi6(v) { state.li6 = v; document.getElementById("li6").value = (v * 100).toFixed(1);
  document.getElementById("li6val").textContent = (v * 100).toFixed(1) + "%"; update(); }
function setTbl(v) { state.tbl = v; document.getElementById("tbl").value = v.toFixed(2);
  document.getElementById("tval").textContent = v.toFixed(2) + " m"; update(); }
function setMult(m) { state.mult = m;
  document.querySelectorAll(".seg").forEach(s => s.classList.toggle("active", s.dataset.mult === m)); update(); }

document.getElementById("li6").addEventListener("input", e => setLi6(+e.target.value / 100));
document.getElementById("tbl").addEventListener("input", e => setTbl(+e.target.value));
document.querySelectorAll(".seg").forEach(s => s.addEventListener("click", () => setMult(s.dataset.mult)));
document.getElementById("optbtn").addEventListener("click", runOptimize);

initCharts();
update();
