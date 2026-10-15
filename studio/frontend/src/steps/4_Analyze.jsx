import { useEffect, useRef, useState, useMemo } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { StepHeader, Spinner, Stat, StatGrid } from '../components/ui'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, ReferenceLine, Cell, Brush,
} from 'recharts'

// ── 3D molecular viewer (3Dmol.js) ──────────────────────────────────────────
const _CPK = {
  H: '#ffffff', C: '#909090', N: '#3050f8', O: '#ff0d0d', F: '#90e050',
  Na: '#ab5cf2', Mg: '#8aff00', Al: '#bfa6a6', Si: '#f0c8a0', P: '#ff8000',
  S: '#ffff30', Cl: '#1ff01f', K: '#8f40d4', Ca: '#3dff00', Ti: '#bfc2c7',
  V: '#a6a6ab', Cr: '#8a99c7', Mn: '#9c7ac7', Fe: '#e06633', Co: '#f090a0',
  Ni: '#50d050', Cu: '#c88033', Zn: '#7d80b0', Ga: '#c28f8f', Ge: '#668f8f',
  As: '#bd80e3', Se: '#ffa100', Br: '#a62929', Y: '#94ffff', Zr: '#94e0e0',
  Nb: '#73c2c9', Mo: '#54b5b5', Ru: '#248f8f', Rh: '#0a7d8c', Pd: '#006985',
  Ag: '#c0c0c0', Sn: '#668080', Sb: '#9e63b5', Te: '#d47a00', I: '#940094',
  Ba: '#00c900', La: '#70d4ff', Hf: '#4dc2ff', Ta: '#4da6ff', W: '#2194d6',
  Re: '#267dab', Os: '#266696', Ir: '#175487', Pt: '#d0d0e0', Au: '#ffd123',
  Hg: '#b8b8d0', Pb: '#575961', Bi: '#9e4fb5',
}

// Covalent radii (Angstrom) used to determine atom-atom bonds.
const _COV_R = {
  H:0.31, He:0.28, Li:1.28, Be:0.96, B:0.84, C:0.76, N:0.71, O:0.66, F:0.57,
  Ne:0.58, Na:1.66, Mg:1.41, Al:1.21, Si:1.11, P:1.07, S:1.05, Cl:1.02, Ar:1.06,
  K:2.03, Ca:1.76, Sc:1.70, Ti:1.60, V:1.53, Cr:1.39, Mn:1.61, Fe:1.32, Co:1.26,
  Ni:1.24, Cu:1.32, Zn:1.22, Ga:1.22, Ge:1.20, As:1.19, Se:1.20, Br:1.20, Kr:1.16,
  Rb:2.20, Sr:1.95, Y:1.90, Zr:1.75, Nb:1.64, Mo:1.54, Ru:1.44, Rh:1.42, Pd:1.39,
  Ag:1.45, Cd:1.44, In:1.42, Sn:1.39, Sb:1.39, Te:1.38, I:1.39, Xe:1.40,
  Cs:2.44, Ba:2.15, La:2.07, Hf:1.75, Ta:1.70, W:1.62, Re:1.51, Os:1.44,
  Ir:1.41, Pt:1.36, Au:1.36, Hg:1.32, Pb:1.46, Bi:1.48,
}
// Minimum-image displacement between atoms a and b under a periodic cell.
// cell is [[ax,ay,az],[bx,by,bz],[cx,cy,cz]] (rows = lattice vectors, A). Search
// the 27 neighboring images and return the shortest a->b vector. Without a cell,
// returns the plain difference.
function _minImageDelta(a, b, cell) {
  let dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z
  if (!cell) return [dx, dy, dz]
  let best = [dx, dy, dz], bestD2 = dx*dx + dy*dy + dz*dz
  for (let i = -1; i <= 1; i++)
    for (let j = -1; j <= 1; j++)
      for (let k = -1; k <= 1; k++) {
        if (i === 0 && j === 0 && k === 0) continue
        const sx = i*cell[0][0] + j*cell[1][0] + k*cell[2][0]
        const sy = i*cell[0][1] + j*cell[1][1] + k*cell[2][1]
        const sz = i*cell[0][2] + j*cell[1][2] + k*cell[2][2]
        const vx = dx + sx, vy = dy + sy, vz = dz + sz
        const d2 = vx*vx + vy*vy + vz*vz
        if (d2 < bestD2) { bestD2 = d2; best = [vx, vy, vz] }
      }
  return best
}
function _bonded(a, b, cell) {
  const ra = _COV_R[a.element] ?? 1.5, rb = _COV_R[b.element] ?? 1.5
  const [dx, dy, dz] = _minImageDelta(a, b, cell)
  return (dx*dx + dy*dy + dz*dz) < (ra + rb + 0.4) ** 2
}
function _computeBonds(atoms, cell) {
  const bonds = []
  for (let i = 0; i < atoms.length; i++)
    for (let j = i+1; j < atoms.length; j++)
      if (_bonded(atoms[i], atoms[j], cell)) bonds.push([i, j])
  return bonds
}
// Unwrap a periodic structure into a connected cluster: BFS over minimum-image
// bonds, shifting each neighbor to the image nearest its already-placed parent.
// This turns a wrapped crystal cell (atoms scattered across the box, bonds
// crossing boundaries) into a chemically sensible connected fragment so
// ball-and-stick renders correctly. Returns new atom objects with shifted x/y/z.
function _unwrapAtoms(atoms, cell) {
  if (!cell || atoms.length === 0) return atoms
  const bonds = _computeBonds(atoms, cell)
  const adj = atoms.map(() => [])
  bonds.forEach(([i, j]) => { adj[i].push(j); adj[j].push(i) })
  const out = atoms.map(a => ({ ...a }))
  const placed = new Array(atoms.length).fill(false)
  for (let s = 0; s < atoms.length; s++) {
    if (placed[s]) continue
    placed[s] = true
    const queue = [s]
    while (queue.length) {
      const p = queue.shift()
      for (const q of adj[p]) {
        if (placed[q]) continue
        // Shift q to the periodic image closest to its parent p (already placed).
        const [dx, dy, dz] = _minImageDelta(out[p], atoms[q], cell)
        out[q].x = out[p].x + dx
        out[q].y = out[p].y + dy
        out[q].z = out[p].z + dz
        placed[q] = true
        queue.push(q)
      }
    }
  }
  return out
}

// Detect usable WebGL. Zscaler Browser Isolation (and some remote/headless
// contexts) render HTML/CSS fine but strip WebGL, which leaves 3Dmol's canvas
// blank. In that case we fall back to a 2D SVG projection so the structure is
// always visible.
function hasWebGL() {
  try {
    const c = document.createElement('canvas')
    const gl = c.getContext('webgl') || c.getContext('experimental-webgl')
    if (!gl) return false
    // A context object alone is not enough: Zscaler Browser Isolation returns a
    // stub context that has no real renderer, so 3Dmol draws nothing. Require a
    // working shader compile + a real (non-empty) renderer string.
    const dbg = gl.getExtension('WEBGL_debug_renderer_info')
    const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)
    if (!renderer) return false
    const sh = gl.createShader(gl.VERTEX_SHADER)
    gl.shaderSource(sh, 'void main(){gl_Position=vec4(0.0);}')
    gl.compileShader(sh)
    const ok = gl.getShaderParameter(sh, gl.COMPILE_STATUS)
    return !!ok
  } catch { return false }
}

// 2D fallback: orthographic projection onto the two highest-variance axes.
function Molecule2D({ atoms, height, cell }) {
  const W = 600, H = height
  const pad = 36
  const xs = atoms.map(a => a.x), ys = atoms.map(a => a.y), zs = atoms.map(a => a.z)
  // Pick the two axes with the largest spread for the most informative view.
  const spread = arr => Math.max(...arr) - Math.min(...arr)
  const axes = [['x', spread(xs)], ['y', spread(ys)], ['z', spread(zs)]]
    .sort((a, b) => b[1] - a[1]).slice(0, 2).map(a => a[0])
  const ax = axes[0], ay = axes[1]
  const av = (a, k) => a[k]
  const aX = atoms.map(a => av(a, ax)), aY = atoms.map(a => av(a, ay))
  const minX = Math.min(...aX), maxX = Math.max(...aX)
  const minY = Math.min(...aY), maxY = Math.max(...aY)
  const rangeX = (maxX - minX) || 1, rangeY = (maxY - minY) || 1
  const sx = v => pad + ((v - minX) / rangeX) * (W - 2 * pad)
  const sy = v => pad + ((maxY - v) / rangeY) * (H - 2 * pad)
  // Depth (third axis) drives radius so nearer atoms look larger.
  const az = ['x', 'y', 'z'].find(k => k !== ax && k !== ay)
  const aZ = atoms.map(a => av(a, az))
  const minZ = Math.min(...aZ), maxZ = Math.max(...aZ), rangeZ = (maxZ - minZ) || 1
  const drawn = atoms.map((a, i) => ({ a, i, depth: (av(a, az) - minZ) / rangeZ }))
    .sort((p, q) => p.depth - q.depth)   // far first, near last (painter's order)
  const bonds = _computeBonds(atoms, cell)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}
      style={{ borderRadius: '.5rem', border: '1px solid #27272a', background: '#0a0a0b' }}>
      {/* Bonds drawn first (behind atoms) */}
      {bonds.map(([i, j]) => (
        <line key={`b${i}-${j}`}
          x1={sx(av(atoms[i], ax))} y1={sy(av(atoms[i], ay))}
          x2={sx(av(atoms[j], ax))} y2={sy(av(atoms[j], ay))}
          stroke="#4a4a52" strokeWidth="2.5" strokeLinecap="round" />
      ))}
      {drawn.map(({ a, i, depth }) => {
        const r = 12 + depth * 12
        const col = _CPK[a.element] || '#dd77ff'
        return (
          <g key={i}>
            <circle cx={sx(av(a, ax))} cy={sy(av(a, ay))} r={r}
              fill={col} stroke="rgba(0,0,0,.5)" strokeWidth="1" opacity={0.55 + depth * 0.45} />
            <text x={sx(av(a, ax))} y={sy(av(a, ay)) + 4} textAnchor="middle"
              fontSize="11" fontWeight="700"
              fill={['#ffffff', '#ffff30', '#8aff00', '#3dff00', '#90e050'].includes(col) ? '#000' : '#fff'}>
              {a.element}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function MoleculeViewer({ atoms, height = 340, formula, sublabel, cell }) {
  const hostRef = useRef(null)
  const viewerRef = useRef(null)
  // Default to the 2D SVG view: it renders everywhere, including inside Zscaler
  // Browser Isolation which serves a stub WebGL context that passes feature
  // detection but draws nothing. Users can opt into 3D via the toggle when WebGL
  // genuinely works (e.g. VSCode local port-forward, direct browser).
  const [use2D, setUse2D] = useState(true)
  // Set when the user explicitly clicks "3D view". In that case we skip the
  // conservative hasWebGL() pre-check (which false-negatives in VSCode webviews /
  // remote port-forwards) and let 3Dmol actually try — falling back to 2D only if
  // it throws at runtime.
  const [force3D, setForce3D] = useState(false)

  // These are periodic DFT crystal cells. Unwrap into a connected cluster so
  // bonds don't stretch across the box (fixes the "disconnected atoms" look).
  const viewAtoms = useMemo(() => _unwrapAtoms(atoms || [], cell), [atoms, cell])

  useEffect(() => {
    let cancelled = false
    if (use2D) return               // 2D mode: skip 3Dmol entirely
    if (!viewAtoms || viewAtoms.length === 0) return
    // Only auto-revert on the strict WebGL check when the user did NOT explicitly
    // ask for 3D. If they clicked the toggle, attempt render regardless.
    if (!force3D && !hasWebGL()) { setUse2D(true); return }
    if (!hostRef.current) return
    ;(async () => {
      try {
        const mod = await import('3dmol')
        const $3Dmol = mod.default || mod
        if (cancelled || !hostRef.current) return
        hostRef.current.innerHTML = ''
        const viewer = $3Dmol.createViewer(hostRef.current, {
          backgroundColor: '#0a0a0b',
        })
        viewerRef.current = viewer
        const model = viewer.addModel()
        // Explicit bonds via minimum-image convention (periodic crystal), so 3Dmol
        // draws the correct connectivity rather than distance-guessing on the
        // wrapped cell. 3Dmol's GLModel has no public addBond(); bonds must be
        // supplied on each atom as parallel `bonds`/`bondOrder` index arrays at
        // addAtoms time. (The old model.addBond() call threw a TypeError that the
        // catch below swallowed — silently reverting every 3D attempt to 2D.)
        const bondPairs = _computeBonds(viewAtoms, null)  // viewAtoms already unwrapped
        const nb = viewAtoms.map(() => ({ bonds: [], bondOrder: [] }))
        bondPairs.forEach(([i, j]) => {
          nb[i].bonds.push(j); nb[i].bondOrder.push(1)
          nb[j].bonds.push(i); nb[j].bondOrder.push(1)
        })
        model.addAtoms(viewAtoms.map((a, i) => ({
          elem: a.element, x: a.x, y: a.y, z: a.z, serial: i,
          bonds: nb[i].bonds, bondOrder: nb[i].bondOrder,
        })))
        viewer.setStyle({}, {
          stick: { radius: 0.14, colorscheme: 'Jmol' },
          sphere: { scale: 0.30, colorscheme: 'Jmol' },
        })
        viewer.zoomTo()
        viewer.render()
        viewer.zoom(1.2, 600)
        viewer.spin('y', 0.5)
      } catch (e) {
        // 3Dmol/WebGL failed at runtime -> fall back to 2D. Log so a real failure
        // is diagnosable instead of a silent revert.
        console.error('[MoleculeViewer] 3Dmol render failed, falling back to 2D:', e)
        if (!cancelled) setUse2D(true)
      }
    })()
    return () => {
      cancelled = true
      try { viewerRef.current?.clear?.() } catch { /* noop */ }
    }
  }, [viewAtoms, use2D, force3D])

  if (!atoms || atoms.length === 0) {
    return (
      <div style={{ padding: '1rem', color: '#52525b', fontSize: '.82rem', textAlign: 'center',
        background: 'rgba(5,5,6,0.8)', borderRadius: '.5rem' }}>
        No atomic coordinates in this result.
      </div>
    )
  }

  // Composition legend (element → count)
  const counts = {}
  atoms.forEach(a => { counts[a.element] = (counts[a.element] || 0) + 1 })
  const uniq = Object.keys(counts)
  return (
    <div>
      <div style={{ position: 'relative' }}>
        {use2D ? (
          <Molecule2D atoms={viewAtoms} height={height} />
        ) : (
          <div ref={hostRef} style={{ position: 'relative', width: '100%', height,
            borderRadius: '.5rem', overflow: 'hidden', border: '1px solid #27272a',
            background: '#0a0a0b' }} />
        )}
        {/* Formula label overlay on the viewer */}
        {formula && (
          <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 10,
            background: 'rgba(10,10,11,0.8)', border: '1px solid #27272a',
            borderRadius: '.4rem', padding: '.3rem .6rem', backdropFilter: 'blur(4px)',
            pointerEvents: 'none' }}>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: '#7dd3fc', letterSpacing: '.02em' }}>
              {formula}
            </div>
            {sublabel && <div style={{ fontSize: '.62rem', color: '#a1a1aa' }}>{sublabel}</div>}
          </div>
        )}
        {/* Manual 2D/3D toggle — guarantees a visible structure even if WebGL
            detection misjudges the environment (e.g. Zscaler isolation). */}
        <button
          onClick={() => {
            if (use2D) setForce3D(true)   // user explicitly wants 3D → bypass strict WebGL pre-check
            setUse2D(v => !v)
          }}
          style={{ position: 'absolute', top: 8, right: 8, zIndex: 10,
            background: 'rgba(10,10,11,0.85)', border: '1px solid #3f3f46',
            borderRadius: '.4rem', padding: '.25rem .6rem', cursor: 'pointer',
            fontSize: '.68rem', fontWeight: 700, color: '#a1a1aa' }}>
          {use2D ? '3D view' : '2D view'}
        </button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem', marginTop: '.5rem' }}>
        {uniq.map(el => (
          <span key={el} style={{ display: 'inline-flex', alignItems: 'center', gap: '.3rem',
            fontSize: '.72rem', color: '#a1a1aa' }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%',
              background: _CPK[el] || '#dd77ff', display: 'inline-block',
              border: '1px solid rgba(255,255,255,.2)' }} />
            {el}<span style={{ color: '#52525b' }}>×{counts[el]}</span>
          </span>
        ))}
        <span style={{ fontSize: '.72rem', color: '#52525b', marginLeft: 'auto' }}>
          drag to rotate · scroll to zoom
        </span>
      </div>
    </div>
  )
}

function WeatherViz({ result }) {
  const field = result.t2m_min !== undefined
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: '.75rem', marginBottom: '1rem' }}>
        {[
          { label: 'Mean Temp', value: result.t2m_mean ? `${(result.t2m_mean - 273.15).toFixed(1)} °C` : '—' },
          { label: 'Min Temp', value: result.t2m_min ? `${(result.t2m_min - 273.15).toFixed(1)} °C` : '—' },
          { label: 'Max Temp', value: result.t2m_max ? `${(result.t2m_max - 273.15).toFixed(1)} °C` : '—' },
          { label: 'Forecast Steps', value: result.forecast_steps ?? '—' },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
            <div className="section-label" style={{ marginBottom: '.3rem' }}>{s.label}</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#f5f5f7' }}>{s.value}</div>
          </div>
        ))}
      </div>
      <GridHeatmap runId={result._runId} />
    </div>
  )
}

function GridHeatmap({ runId }) {
  const [data, setData] = useState(null)
  useEffect(() => {
    if (!runId) return
    fetch(`/api/jobs/${runId}/files`)
      .then(r => r.json())
      .then(files => {
        const f = files.find(x => x.name === 'forecast_field.json')
        if (!f) return
        // Read file via a proxy — not available in plain API; fall back to summary
      })
  }, [runId])
  return (
    <div style={{ padding: '.75rem', background: 'rgba(5,5,6,0.8)', borderRadius: '.5rem',
      color: '#52525b', fontSize: '.8rem', textAlign: 'center' }}>
      Global temperature field (73×144 grid) — download forecast_field.json for full visualization
    </div>
  )
}

function DownscalingViz({ result, runId }) {
  const stat = (label, value) => (
    <div key={label} className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
      <div className="section-label" style={{ marginBottom: '.3rem' }}>{label}</div>
      <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f5f5f7' }}>{value}</div>
    </div>
  )
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(150px,1fr))', gap: '.75rem', marginBottom: '1rem' }}>
        {result.upscale_factor && stat('Downscaling', result.upscale_factor)}
        {result.psnr !== undefined && stat('PSNR', `${result.psnr} dB`)}
        {result.ssim !== undefined && stat('SSIM', result.ssim)}
        {result.prediction_shape && stat('Output Grid', result.prediction_shape.join(' × '))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        {['0_input.png', '0_prediction.png'].map((fn, i) => (
          <div key={fn} className="card" style={{ padding: '.75rem' }}>
            <div className="section-label" style={{ marginBottom: '.4rem' }}>
              {i === 0 ? `Low-res Input ${result.input_shape ? '(' + result.input_shape.join('×') + ')' : ''}`
                       : `Downscaled Prediction ${result.prediction_shape ? '(' + result.prediction_shape.join('×') + ')' : ''}`}
            </div>
            <img src={`/api/jobs/${runId}/files/${fn}`} alt={fn}
              style={{ width: '100%', imageRendering: 'pixelated', borderRadius: '.4rem', background: '#000' }} />
          </div>
        ))}
      </div>
      <div style={{ marginTop: '.75rem', fontSize: '.78rem', color: '#71717a' }}>
        Variable: {result.variable || '—'} · Predicted range: {result.prediction_min ?? '—'} to {result.prediction_max ?? '—'} (physical units, denormalized)
      </div>
    </div>
  )
}

function EnergyViz({ result }) {
  const atoms = result.atoms || []
  const err = result.abs_error_ev_per_atom
  const errColor = err == null ? '#f5f5f7' : err < 0.15 ? '#21c77a' : err < 0.35 ? '#f5a524' : '#ff8f93'
  return (
    <div>
      <div style={{ marginBottom: '.9rem', fontSize: '.95rem', color: '#f5f5f7' }}>
        Material: <span style={{ fontWeight: 800, color: '#7dd3fc' }}>{result.material_name || result.formula}</span>
        <span style={{ color: '#71717a' }}> · {result.formula} · {result.n_atoms} atoms</span>
        {result.model_variant && (
          <span className="badge badge-amd" style={{ marginLeft: '.5rem' }}>{result.model_variant} model</span>
        )}
      </div>
      {result.application && (
        <div style={{ marginBottom: '.9rem', padding: '.6rem .8rem',
          background: 'rgba(56,189,248,.06)', border: '1px solid rgba(56,189,248,.18)',
          borderRadius: '.5rem', fontSize: '.82rem', color: '#a1a1aa', lineHeight: 1.5 }}>
          <span style={{ color: '#7dd3fc', fontWeight: 700 }}>Why this material: </span>
          {result.application}
        </div>
      )}
      {result.cell && (
        <div style={{ marginBottom: '.9rem', fontSize: '.72rem', color: '#71717a' }}>
          Periodic crystal — the viewer shows one unit cell, with bonds resolved
          across periodic boundaries (minimum-image convention).
        </div>
      )}

      {/* Two columns: 3D structure | prediction stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px,1fr) minmax(260px,1fr)',
        gap: '1rem', alignItems: 'start' }}>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem' }}>Atomistic Structure</div>
          <MoleculeViewer atoms={atoms} cell={result.cell} formula={result.formula}
            sublabel={result.n_atoms ? `${result.n_atoms} atoms · held-out DFT structure` : null} />
        </div>

        <div>
          <StatGrid min={130}>
            {result.predicted_energy_ev_per_atom !== undefined &&
              <Stat label="Predicted Energy" value={`${result.predicted_energy_ev_per_atom}`} sub="eV/atom" color="#21c77a" />}
            {result.dft_reference_ev_per_atom !== undefined &&
              <Stat label="DFT Reference" value={`${result.dft_reference_ev_per_atom}`} sub="eV/atom (DFT label)" />}
            {err !== undefined &&
              <Stat label="Abs Error" value={`${err}`} sub="eV/atom" color={errColor} />}
          </StatGrid>

          {/* Predicted vs DFT bar comparison */}
          {result.predicted_energy_ev_per_atom !== undefined && result.dft_reference_ev_per_atom !== undefined && (
            <div className="card" style={{ padding: '.6rem .6rem .2rem', marginBottom: '.75rem' }}>
              <div className="section-label" style={{ marginBottom: '.3rem' }}>Prediction vs DFT</div>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart layout="vertical" data={[
                  { name: 'Predicted', v: result.predicted_energy_ev_per_atom, fill: '#21c77a' },
                  { name: 'DFT ref', v: result.dft_reference_ev_per_atom, fill: '#38bdf8' },
                ]} margin={{ left: 10, right: 20, top: 4, bottom: 4 }}>
                  <XAxis type="number" tick={{ fill: '#71717a', fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 11 }} width={70} />
                  <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 12 }} />
                  <Bar dataKey="v" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                    {[0, 1].map(i => <Cell key={i} fill={i === 0 ? '#21c77a' : '#38bdf8'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <StatGrid min={110}>
            {result.model_val_corr != null && <Stat label="Val Corr" value={result.model_val_corr} />}
            {result.model_val_r2 != null && <Stat label="Val R²" value={result.model_val_r2} />}
            {result.model_val_mae != null && <Stat label="Val MAE" value={result.model_val_mae} />}
          </StatGrid>
        </div>
      </div>

      <div style={{ fontSize: '.78rem', color: '#71717a', lineHeight: 1.5, marginTop: '.75rem' }}>
        {result.model} · {result.note}
      </div>
    </div>
  )
}

function MaterialsViz({ result }) {
  if (result.type === 'atomistic_properties') {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px,1fr) minmax(260px,1fr)',
        gap: '1rem', alignItems: 'start' }}>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem' }}>Atomistic Structure</div>
          <MoleculeViewer atoms={result.atoms || []} cell={result.cell} formula={result.formula}
            sublabel={result.n_atoms ? `${result.n_atoms} atoms · ${result.structure_type || 'crystal'}` : null} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(140px,1fr))', gap: '.75rem', alignContent: 'start' }}>
          {[
            { label: 'Formation Energy', value: `${result.formation_energy_eV_per_atom} eV/atom` },
            { label: 'Bulk Modulus', value: `${result.bulk_modulus_GPa} GPa` },
            { label: 'Shear Modulus', value: `${result.shear_modulus_GPa} GPa` },
            { label: 'Band Gap', value: `${result.band_gap_eV} eV` },
            { label: 'Magnetic Moment', value: `${result.magnetic_moment_muB} μB` },
            { label: 'Atoms', value: result.n_atoms },
          ].map(s => (
            <div key={s.label} className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
              <div className="section-label" style={{ marginBottom: '.3rem' }}>{s.label}</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#f5f5f7' }}>{s.value}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }
  if (result.type === 'crystal_generation') {
    const stable = result.structures?.filter(s => s.is_stable) || []
    return (
      <div>
        <div style={{ marginBottom: '.75rem', color: '#21c77a', fontWeight: 700 }}>
          {stable.length} / {result.n_generated} structures thermodynamically stable
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
          {result.structures?.map(s => (
            <div key={s.id} className="card" style={{ padding: '.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <span style={{ fontWeight: 700, color: '#f5f5f7', marginRight: '.5rem' }}>{s.formula}</span>
                  <span className="badge badge-muted">{s.space_group}</span>
                </div>
                <span className={`badge badge-${s.is_stable ? 'ok' : 'warn'}`}>
                  {s.is_stable ? 'Stable' : 'Unstable'}
                </span>
              </div>
              <div style={{ fontSize: '.78rem', color: '#94a3b8', marginTop: '.3rem' }}>
                a={s.a_A} Å, b={s.b_A} Å, c={s.c_A} Å &nbsp;·&nbsp; E={s.formation_energy_eV_per_atom} eV/atom
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }
  return <div style={{ color: '#52525b' }}>See output files for full results.</div>
}

function HealthcareViz({ result }) {
  if (result.type === 'molecule_generation') {
    return (
      <div>
        <div style={{ marginBottom: '.75rem', display: 'flex', gap: '1rem' }}>
          <div className="card" style={{ padding: '.75rem', textAlign: 'center', flex: 1 }}>
            <div className="section-label">Generated</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f5f5f7' }}>{result.n_valid}</div>
          </div>
          <div className="card" style={{ padding: '.75rem', textAlign: 'center', flex: 1 }}>
            <div className="section-label">Lipinski Pass</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#21c77a' }}>
              {(result.lipinski_pass_rate * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.35rem' }}>
          {result.molecules?.slice(0, 10).map(m => (
            <div key={m.id} className="card" style={{ padding: '.6rem .9rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
                <code style={{ fontSize: '.75rem', color: '#7dd3fc', flex: 1, overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.smiles}</code>
                <div style={{ display: 'flex', gap: '.4rem', flexShrink: 0 }}>
                  <span className="badge badge-muted">MW {m.MW}</span>
                  <span className="badge badge-muted">LogP {m.LogP}</span>
                  <span className={`badge badge-${m.Lipinski_pass ? 'ok' : 'warn'}`}>
                    {m.Lipinski_pass ? 'Lipinski ✓' : 'Lipinski ✗'}
                  </span>
                </div>
              </div>
            </div>
          ))}
          {(result.molecules?.length || 0) > 10 && (
            <div style={{ color: '#52525b', fontSize: '.8rem', textAlign: 'center', padding: '.3rem' }}>
              + {result.molecules.length - 10} more — see generated_molecules.json
            </div>
          )}
        </div>
      </div>
    )
  }
  if (result.type === 'segmentation') {
    return (
      <div>
        <div style={{ marginBottom: '.75rem', color: '#f5f5f7', fontWeight: 600 }}>
          Tumor volume: {result.tumor_volume_mL} mL
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '.75rem' }}>
          {Object.entries(result.dice_scores || {}).map(([cls, score]) => (
            <div key={cls} className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
              <div className="section-label" style={{ marginBottom: '.3rem' }}>
                {cls.replace('_', ' ')}
              </div>
              <div style={{ fontSize: '1.3rem', fontWeight: 800,
                color: score > 0.85 ? '#21c77a' : score > 0.75 ? '#f5a524' : '#ff8f93' }}>
                {score.toFixed(3)}
              </div>
              <div style={{ fontSize: '.7rem', color: '#52525b' }}>Dice</div>
            </div>
          ))}
        </div>
      </div>
    )
  }
  return <div style={{ color: '#52525b' }}>See output files for full results.</div>
}

function PhysicsViz({ result }) {
  if (!result.trajectory) return <div style={{ color: '#52525b' }}>No trajectory data.</div>
  const last = result.trajectory[result.trajectory.length - 1]
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '.75rem', marginBottom: '1rem' }}>
        {[
          { label: 'Steps', value: result.n_steps },
          { label: 'Grid Size', value: result.nx },
          { label: 'Final Energy', value: result.final_energy?.toFixed(4) },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
            <div className="section-label" style={{ marginBottom: '.3rem' }}>{s.label}</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#f5f5f7' }}>{s.value}</div>
          </div>
        ))}
      </div>
      <div style={{ padding: '.75rem', background: 'rgba(5,5,6,0.8)', borderRadius: '.5rem',
        fontSize: '.8rem', color: '#52525b' }}>
        Field snapshots at steps: {result.trajectory.map(t => t.step).join(', ')} — see rollout.json
      </div>
    </div>
  )
}

function TrainingConvergenceViz({ result }) {
  const r1 = result.runs?.['1gpu']
  const r8 = result.runs?.['8gpu']
  if (!r1 || !r8) return <div style={{ color: '#52525b' }}>Training data unavailable.</div>

  // Merge epochs by index for the line chart.
  const n = Math.max(r1.epochs.length, r8.epochs.length)
  const curve = Array.from({ length: n }, (_, i) => ({
    ep: i,
    val1: r1.epochs[i]?.val ?? null,
    val8: r8.epochs[i]?.val ?? null,
    train1: r1.epochs[i]?.train ?? null,
    train8: r8.epochs[i]?.train ?? null,
  }))

  const m1 = r1.metrics || {}, m8 = r8.metrics || {}
  const rows = [
    { k: 'Test correlation', a: m1.corr, b: m8.corr, better: 'high' },
    { k: 'Test R²', a: m1.r2, b: m8.r2, better: 'high' },
    { k: 'Test MAE (eV/atom)', a: m1.mae, b: m8.mae, better: 'low' },
    { k: 'Training structures', a: r1.n_samples, b: r8.n_samples, better: 'high', int: true },
  ]

  // Parity scatter for 8-GPU (best model).
  const parity = (result.parity?.['8gpu']?.pred || []).map((p, i) => ({
    pred: p, true: result.parity['8gpu'].true[i],
  }))
  const allv = parity.flatMap(d => [d.pred, d.true])
  const lo = allv.length ? Math.min(...allv) : -3
  const hi = allv.length ? Math.max(...allv) : 1

  const speedSamples = [
    { name: '1 GPU', v: r1.n_samples, fill: '#71717a' },
    { name: '8 GPU', v: r8.n_samples, fill: '#ED1C24' },
  ]

  return (
    <div>
      {/* Compact header row: description + key metrics inline */}
      <div style={{ display: 'flex', gap: '.75rem', marginBottom: '.75rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div className="card" style={{ padding: '.6rem .9rem', flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: '.8rem', color: '#a1a1aa', lineHeight: 1.5 }}>
            Real HydraGNN training on <b style={{ color: '#f5f5f7' }}>Alexandria DFT</b> data.
            8 GPUs → <b style={{ color: '#ED1C24' }}>{(r8.n_samples / r1.n_samples).toFixed(1)}×</b> more data,
            same wall-clock time.
          </div>
        </div>
        {/* Inline metric deltas */}
        {[
          { k: 'Corr', a: m1.corr, b: m8.corr, hi: true },
          { k: 'R²', a: m1.r2, b: m8.r2, hi: true },
          { k: 'MAE', a: m1.mae, b: m8.mae, hi: false },
        ].map(({ k, a, b, hi }) => {
          const win = hi ? b > a : b < a
          return (
            <div key={k} className="card" style={{ padding: '.5rem .7rem', textAlign: 'center', minWidth: 80 }}>
              <div className="section-label" style={{ fontSize: '.6rem' }}>{k}</div>
              <div style={{ fontSize: '.72rem', color: '#71717a' }}>{a?.toFixed(3)}</div>
              <div style={{ fontSize: '.78rem', fontWeight: 800, color: win ? '#21c77a' : '#ff8f93' }}>→ {b?.toFixed(3)}</div>
            </div>
          )
        })}
      </div>

      {/* Row 1: convergence chart (full width, compact height) */}
      <div className="card" style={{ padding: '.6rem .8rem', marginBottom: '.6rem' }}>
        <div className="section-label" style={{ marginBottom: '.3rem' }}>Validation Loss Convergence</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={curve} margin={{ top: 4, right: 12, bottom: 16, left: 0 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis dataKey="ep" tick={{ fill: '#71717a', fontSize: 10 }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -8, fill: '#71717a', fontSize: 10 }} />
            <YAxis tick={{ fill: '#71717a', fontSize: 10 }} domain={['auto', 'auto']} width={36} />
            <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="val1" name="1-GPU val" stroke="#f5a524" dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="monotone" dataKey="val8" name="8-GPU val" stroke="#38bdf8" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Row 2: left = bar + stats table, right = parity scatter */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.6rem' }}>
        {/* Left column: training size bar + comparison table */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
          <div className="card" style={{ padding: '.6rem .8rem' }}>
            <div className="section-label" style={{ marginBottom: '.3rem' }}>Training Structures / Epoch</div>
            <ResponsiveContainer width="100%" height={130}>
              <BarChart data={speedSamples} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
                <YAxis tick={{ fill: '#71717a', fontSize: 10 }} width={50} />
                <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 11 }} />
                <Bar dataKey="v" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {speedSamples.map((s, i) => <Cell key={i} fill={s.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {r8.throughput_it_s && (
              <div style={{ fontSize: '.68rem', color: '#52525b', marginTop: '.2rem', textAlign: 'center' }}>
                8-GPU: {r8.throughput_it_s} it/s/rank · {(r8.throughput_it_s * 8).toFixed(0)} it/s total
              </div>
            )}
          </div>

          {/* Stats comparison table (in left column) */}
          <div className="card" style={{ padding: '.6rem .8rem' }}>
            <div className="section-label" style={{ marginBottom: '.3rem' }}>Final Model Comparison</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.78rem' }}>
              <thead>
                <tr style={{ color: '#71717a', textAlign: 'right' }}>
                  <th style={{ textAlign: 'left', padding: '.25rem .4rem' }}>Metric</th>
                  <th style={{ padding: '.25rem .4rem' }}>1-GPU</th>
                  <th style={{ padding: '.25rem .4rem' }}>8-GPU</th>
                  <th style={{ padding: '.25rem .4rem' }}>Δ</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => {
                  const fmt = v => v == null ? '—' : r.int ? v.toLocaleString() : (+v).toFixed(3)
                  const win8 = r.better === 'high' ? r.b > r.a : r.b < r.a
                  return (
                    <tr key={r.k} style={{ borderTop: '1px solid #27272a', color: '#f5f5f7' }}>
                      <td style={{ textAlign: 'left', padding: '.28rem .4rem', color: '#a1a1aa' }}>{r.k}</td>
                      <td style={{ textAlign: 'right', padding: '.28rem .4rem' }}>{fmt(r.a)}</td>
                      <td style={{ textAlign: 'right', padding: '.28rem .4rem',
                        color: win8 ? '#21c77a' : '#f5f5f7', fontWeight: win8 ? 800 : 400 }}>{fmt(r.b)}</td>
                      <td style={{ textAlign: 'right', padding: '.28rem .4rem',
                        color: win8 ? '#21c77a' : '#ff8f93' }}>
                        {r.a && r.b ? (r.better === 'high'
                          ? `+${(((r.b - r.a) / Math.abs(r.a)) * 100).toFixed(0)}%`
                          : `−${(((r.a - r.b) / Math.abs(r.a)) * 100).toFixed(0)}%`) : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>{/* end left column */}

        {/* Right column: parity scatter */}
        <div className="card" style={{ padding: '.6rem .8rem' }}>
          <div className="section-label" style={{ marginBottom: '.3rem' }}>8-GPU: Predicted vs DFT</div>
          <ResponsiveContainer width="100%" height={310}>
            <ScatterChart margin={{ top: 4, right: 12, bottom: 16, left: 0 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="true" name="DFT" domain={[lo, hi]}
                tick={{ fill: '#71717a', fontSize: 10 }}
                label={{ value: 'DFT (eV/atom)', position: 'insideBottom', offset: -8, fill: '#71717a', fontSize: 10 }} />
              <YAxis type="number" dataKey="pred" name="Predicted" domain={[lo, hi]}
                tick={{ fill: '#71717a', fontSize: 10 }} width={40}
                label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fill: '#71717a', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 11 }}
                cursor={{ strokeDasharray: '3 3' }} />
              <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#52525b" strokeDasharray="4 4" />
              <Scatter data={parity} fill="#38bdf8" fillOpacity={0.5} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{ fontSize: '.68rem', color: '#52525b', textAlign: 'center', marginTop: '.2rem' }}>
            Dashed line = perfect prediction · val corr {m8.corr?.toFixed(3)}
          </div>
        </div>
      </div>{/* end row 2 grid */}
    </div>
  )
}

// ── System Telemetry Panel — GPU metrics captured by AMD Omnistat ─────────────
// Catalog-driven: the selectable metric list comes from /api/telemetry/catalog, so
// the user picks WHICH of Omnistat's ~25 aggregated metrics to plot (and how many)
// from a grouped dropdown. Peaks/means tiles + N interactive time-series render
// dynamically from the selection. Data is real (live cross-node query or the baked
// demo asset) — nothing synthesized.
const _CHART_COLORS = ['#ED1C24', '#f59e0b', '#38bdf8', '#a78bfa', '#21c77a',
  '#f472b6', '#94a3b8', '#eab308', '#34d399', '#60a5fa']
const _fmtVal = (v, unit) => {
  if (v == null) return '—'
  const a = Math.abs(v)
  const d = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3
  return `${v.toFixed(d)} ${unit}`
}

// Grouped multi-select dropdown of catalog metrics.
function MetricPicker({ catalog, selected, onChange }) {
  const [open, setOpen] = useState(false)
  const groups = catalog?.groups || []
  const byGroup = useMemo(() => {
    const m = {}
    for (const d of (catalog?.metrics || [])) (m[d.group] ||= []).push(d)
    return m
  }, [catalog])
  const toggle = key => onChange(selected.includes(key)
    ? selected.filter(k => k !== key) : [...selected, key])
  return (
    <div style={{ position: 'relative', marginBottom: '.9rem' }}>
      <button onClick={() => setOpen(o => !o)} className="btn btn-ghost"
        style={{ fontSize: '.8rem', border: '1px solid #27272a' }}>
        Metrics: {selected.length} selected ▾
      </button>
      {open && (
        <div className="card" style={{ position: 'absolute', zIndex: 50, marginTop: '.3rem',
          padding: '.6rem .8rem', maxHeight: 360, overflowY: 'auto', minWidth: 300,
          background: '#0f0f11', border: '1px solid #27272a' }}>
          {groups.map(g => (
            <div key={g} style={{ marginBottom: '.5rem' }}>
              <div className="section-label" style={{ fontSize: '.6rem', marginBottom: '.25rem' }}>{g}</div>
              {(byGroup[g] || []).map(d => (
                <label key={d.key} style={{ display: 'flex', alignItems: 'center', gap: '.4rem',
                  fontSize: '.78rem', color: '#d4d4d8', padding: '.12rem 0', cursor: 'pointer' }}>
                  <input type="checkbox" checked={selected.includes(d.key)}
                    onChange={() => toggle(d.key)} style={{ accentColor: '#ED1C24' }} />
                  {d.label} <span style={{ color: '#52525b' }}>({d.unit})</span>
                </label>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TelemetryPanel({ result, catalog, selected, setSelected, live }) {
  const tel = result.telemetry || {}
  const peaks = tel.peaks || {}
  const means = tel.means || {}
  const units = tel.units || {}
  const series = tel.series || {}

  // Descriptor lookup from the catalog (label/unit per key).
  const desc = useMemo(() => {
    const m = {}
    for (const d of (catalog?.metrics || [])) m[d.key] = d
    return m
  }, [catalog])

  // Which selected metrics actually have a series in this payload.
  const plotKeys = (selected || []).filter(k => (series[k] || []).length > 0)
  const rows = useMemo(() => {
    const t = series.t_s || []
    return t.map((tt, i) => {
      const row = { t: tt }
      for (const k of plotKeys) row[k] = series[k]?.[i] ?? null
      return row
    })
  }, [series, plotKeys.join(',')])

  const metrics = result.metrics || {}
  // Tiles: the selected metrics + energy (always interesting).
  const tileKeys = [...new Set([...(selected || []), 'energy_kj'])]

  return (
    <div>
      <div className="section-label" style={{ color: '#ED1C24', marginBottom: '.5rem' }}>
        SYSTEM TELEMETRY — AMD INSTINCT MI355X · {live ? 'LIVE' : 'PEEK UNDER THE HOOD'}
      </div>
      <div style={{ fontSize: '.78rem', color: '#a1a1aa', marginBottom: '.9rem' }}>
        GPU metrics captured by Omnistat — {result.n_gpus || 8} GPUs
        {result.runtime_s ? `, ${result.runtime_s}s` : ''}
        {result.epochs ? `, ${result.epochs} epochs` : ''}.
        {' '}Also collects raw perf counters, xGMI scale-up & network scale-out bandwidth, and host I/O.
      </div>

      <MetricPicker catalog={catalog} selected={selected} onChange={setSelected} />

      {/* Peak stat tiles (dynamic: selected metrics + energy) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
        gap: '.6rem', marginBottom: '1.1rem' }}>
        {tileKeys.map((k, i) => {
          const label = k === 'energy_kj' ? 'ENERGY' : (desc[k]?.label || k).toUpperCase()
          const unit = units[k] || desc[k]?.unit || ''
          return (
            <div key={k} className="card" style={{ padding: '.7rem', textAlign: 'center' }}>
              <div className="section-label" style={{ marginBottom: '.35rem', fontSize: '.6rem' }}>
                {k === 'energy_kj' ? '' : 'PEAK '}{label}
              </div>
              <div style={{ fontSize: '1.2rem', fontWeight: 900, color: _CHART_COLORS[i % _CHART_COLORS.length] }}>
                {peaks[k] != null ? _fmtVal(peaks[k], unit) : '—'}
              </div>
              {k !== 'energy_kj' && means[k] != null && (
                <div style={{ fontSize: '.6rem', color: '#71717a', marginTop: '.2rem' }}>
                  mean {_fmtVal(means[k], unit)}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Final accuracy (from the training's own validation.json) */}
      {(metrics.corr != null || metrics.mae != null) && (
        <div className="card" style={{ padding: '.6rem .9rem', marginBottom: '1.1rem',
          fontSize: '.8rem', color: '#a1a1aa' }}>
          Final model accuracy —
          {metrics.corr != null && <> corr <b style={{ color: '#f5f5f7' }}>{metrics.corr?.toFixed(4)}</b></>}
          {metrics.r2 != null && <> · R² <b style={{ color: '#f5f5f7' }}>{metrics.r2?.toFixed(4)}</b></>}
          {metrics.mae != null && <> · MAE <b style={{ color: '#f5f5f7' }}>{metrics.mae?.toFixed(4)}</b> eV/atom</>}
        </div>
      )}

      {/* One interactive time-series chart per selected metric that has data */}
      {plotKeys.length > 0 ? plotKeys.map((k, i) => {
        const label = desc[k]?.label || k
        const unit = units[k] || desc[k]?.unit || ''
        const color = _CHART_COLORS[i % _CHART_COLORS.length]
        return (
          <div key={k} className="card" style={{ padding: '.75rem', marginBottom: '.85rem' }}>
            <div className="section-label" style={{ marginBottom: '.4rem' }}>{label} ({unit})</div>
            <ResponsiveContainer width="100%" height={190}>
              <LineChart data={rows} margin={{ top: 5, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="t" tick={{ fill: '#71717a', fontSize: 10 }}
                  tickFormatter={v => `${v}s`} minTickGap={28} />
                <YAxis tick={{ fill: '#71717a', fontSize: 10 }} width={44} />
                <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 11 }}
                  labelFormatter={v => `t = ${v}s`} formatter={val => [`${val} ${unit}`, label]} />
                <Line type="monotone" dataKey={k} name={label} stroke={color}
                  dot={false} strokeWidth={1.6} isAnimationActive={false} />
                <Brush dataKey="t" height={18} stroke="#ED1C24" fill="#0a0a0b"
                  travellerWidth={8} tickFormatter={v => `${v}s`} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      }) : (
        <div style={{ color: '#52525b', fontSize: '.8rem' }}>
          {selected?.length ? 'No data yet for the selected metrics.' : 'Select metrics above to plot.'}
        </div>
      )}
    </div>
  )
}

// Fetches the metric catalog + owns metric selection, then renders TelemetryPanel.
// Used for the static/final (Results-view) case; PerformanceTab manages its own
// selection so it can drive the live query keys.
function useTelemetryCatalog() {
  const [catalog, setCatalog] = useState(null)
  useEffect(() => {
    let stop = false
    api.telemetryCatalog().then(c => { if (!stop) setCatalog(c) }).catch(() => {})
    return () => { stop = true }
  }, [])
  return catalog
}

function TelemetryPanelContainer({ result }) {
  const catalog = useTelemetryCatalog()
  const [selected, setSelected] = useState(null)
  // Seed selection from the payload's own series keys (what was actually harvested),
  // falling back to catalog defaults.
  useEffect(() => {
    if (selected) return
    const fromResult = Object.keys(result?.telemetry?.series || {}).filter(k => k !== 't_s')
    if (fromResult.length) setSelected(fromResult)
    else if (catalog?.defaults) setSelected(catalog.defaults)
  }, [catalog, result, selected])
  if (!catalog || !selected) return <Spinner size={24} />
  return <TelemetryPanel result={result} catalog={catalog} selected={selected} setSelected={setSelected} />
}

// ── Training Results — loss convergence + final accuracy (the Results tab for
//    an 8-GPU telemetry training run; telemetry itself lives on Performance). ──
function TrainingResultsView({ result }) {
  const curve = result.loss_curve || []
  const m = result.metrics || {}
  const hasCurve = curve.length > 0

  return (
    <div>
      {/* Final accuracy */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(120px,1fr))',
        gap: '.6rem', marginBottom: '1.1rem' }}>
        {[
          { k: 'Final test corr', v: m.corr != null ? m.corr.toFixed(4) : '—', c: '#21c77a' },
          { k: 'Final test R²', v: m.r2 != null ? m.r2.toFixed(4) : '—', c: '#38bdf8' },
          { k: 'Final MAE (eV/atom)', v: m.mae != null ? m.mae.toFixed(4) : '—', c: '#f59e0b' },
          { k: 'Epochs', v: result.epochs ?? '—', c: '#f5f5f7' },
        ].map(s => (
          <div key={s.k} className="card" style={{ padding: '.7rem', textAlign: 'center' }}>
            <div className="section-label" style={{ marginBottom: '.3rem', fontSize: '.62rem' }}>{s.k}</div>
            <div style={{ fontSize: '1.15rem', fontWeight: 900, color: s.c }}>{s.v}</div>
          </div>
        ))}
      </div>

      {/* Loss convergence */}
      <div className="card" style={{ padding: '.75rem' }}>
        <div className="section-label" style={{ marginBottom: '.4rem' }}>Loss convergence</div>
        {hasCurve ? (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={curve} margin={{ top: 5, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="ep" tick={{ fill: '#71717a', fontSize: 10 }}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#71717a', fontSize: 10 }} />
              <YAxis tick={{ fill: '#71717a', fontSize: 10 }} width={48} />
              <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 11 }}
                labelFormatter={v => `Epoch ${v}`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train" name="Train" stroke="#ED1C24" dot={false} strokeWidth={1.6} isAnimationActive={false} />
              <Line type="monotone" dataKey="val" name="Validation" stroke="#38bdf8" dot={false} strokeWidth={1.6} isAnimationActive={false} />
              <Line type="monotone" dataKey="test" name="Test" stroke="#a78bfa" dot={false} strokeWidth={1.6} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ color: '#52525b', fontSize: '.8rem' }}>
            Loss curve appears once the run completes.
          </div>
        )}
      </div>
    </div>
  )
}

// ── DC Temperature Downscaling Viz — Leaflet map with temperature overlay ────
function tempToHex(t, tmin, tmax) {
  const norm = Math.max(0, Math.min(1, (t - tmin) / (tmax - tmin || 1)))
  // Blue (cold) → white (mid) → red (hot)
  let r, g, b
  if (norm < 0.5) {
    const f = norm * 2
    r = Math.round(f * 255); g = Math.round(f * 255); b = 255
  } else {
    const f = (norm - 0.5) * 2
    r = 255; g = Math.round((1 - f) * 255); b = Math.round((1 - f) * 255)
  }
  return `rgba(${r},${g},${b},0.55)`
}

// Real DC-region geography, embedded as factual lat/lon vectors so the basemap
// renders under Zscaler Browser Isolation (external OSM tiles are stripped). All
// coordinates are real; nothing is fabricated.
const DC_CITIES = [
  { name: 'Washington DC', lat: 38.905, lon: -77.037, major: true },
  { name: 'Baltimore', lat: 39.290, lon: -76.612 },
  { name: 'Annapolis', lat: 38.979, lon: -76.492 },
  { name: 'Richmond', lat: 37.541, lon: -77.436 },
  { name: 'Fredericksburg', lat: 38.303, lon: -77.461 },
]
// Washington DC boundary diamond (the original 10x10 mile square corners).
const DC_DIAMOND = [
  [38.995, -77.041], [38.892, -76.909], [38.789, -77.041], [38.892, -77.120], [38.995, -77.041],
]

// Self-contained SVG temperature-grid heatmap. No external tiles/CSS, so it
// renders inside Zscaler Browser Isolation where Leaflet (unpkg CSS +
// OpenStreetMap tiles) is blocked. Shows the real temperature field with a DC
// marker, coastline from the land-sea mask, a lat/lon graticule, and real city
// reference markers so it reads as a proper map.
function SVGTempMap({ gridData, dcLat, dcLon, label, height = 360, landMask = null, geo = true }) {
  const lats = gridData.lat || [], lons = gridData.lon || [], grid = gridData.temp_c || []
  const mask = landMask || gridData.land_sea_mask || null
  const allTemps = grid.flat().filter(t => t !== null)
  const tmin = allTemps.length ? Math.min(...allTemps) : 15
  const tmax = allTemps.length ? Math.max(...allTemps) : 42
  const rows = grid.length, cols = grid[0]?.length || 0
  const W = 600, H = height, pad = 4
  const cw = (W - 2 * pad) / (cols || 1), ch = (H - 2 * pad) / (rows || 1)
  // lat[0] may be min or max; map so north (higher lat) is on top.
  const latAsc = lats.length > 1 && lats[1] > lats[0]
  const lonToX = lon => pad + ((lon - lons[0]) / ((lons[cols-1] - lons[0]) || 1)) * (W - 2*pad)
  const latToY = lat => {
    const f = (lat - lats[0]) / ((lats[rows-1] - lats[0]) || 1)
    return pad + (latAsc ? (1 - f) : f) * (H - 2*pad)
  }
  // Row index -> y (top of cell), honoring N-up orientation.
  const rowY = ri => (latAsc ? (rows - 1 - ri) : ri) * ch + pad
  const isLand = (ri, ci) => !mask || (mask[ri]?.[ci] ?? 1) >= 0.5
  // Graticule at whole-degree lat/lon lines within the field extent.
  const inLon = lon => lons.length > 1 && lon >= Math.min(lons[0], lons[cols-1]) && lon <= Math.max(lons[0], lons[cols-1])
  const inLat = lat => lats.length > 1 && lat >= Math.min(lats[0], lats[rows-1]) && lat <= Math.max(lats[0], lats[rows-1])
  const gratLons = [], gratLats = []
  if (geo && lons.length > 1) {
    for (let d = Math.ceil(Math.min(lons[0], lons[cols-1])); d <= Math.max(lons[0], lons[cols-1]); d++) gratLons.push(d)
    for (let d = Math.ceil(Math.min(lats[0], lats[rows-1])); d <= Math.max(lats[0], lats[rows-1]); d++) gratLats.push(d)
  }
  const inView = (lat, lon) => inLat(lat) && inLon(lon)
  // Coastline segments: boundary between a land cell and a sea neighbor (right/below).
  const coast = []
  if (mask) {
    for (let ri = 0; ri < rows; ri++) {
      for (let ci = 0; ci < cols; ci++) {
        const here = isLand(ri, ci)
        const x0 = pad + ci*cw, y0 = rowY(ri)
        if (ci + 1 < cols && here !== isLand(ri, ci+1))
          coast.push({ x1: x0+cw, y1: y0, x2: x0+cw, y2: y0+ch })
        if (ri + 1 < rows && here !== isLand(ri+1, ci)) {
          // shared horizontal edge between row ri and ri+1
          const yb = latAsc ? rowY(ri) : rowY(ri)+ch
          coast.push({ x1: x0, y1: yb, x2: x0+cw, y2: yb })
        }
      }
    }
  }
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}
        style={{ borderRadius: '.5rem', border: '1px solid #27272a', background: '#0a0a0b', display: 'block' }}>
        {/* Water underlay from the model's own land-sea mask (self-contained, no tiles). */}
        {mask && grid.map((row, ri) => row.map((t, ci) => {
          if (isLand(ri, ci)) return null
          return <rect key={`w-${ri}-${ci}`} x={pad + ci*cw} y={rowY(ri)} width={cw+0.5} height={ch+0.5}
            fill="#0f2540" />
        }))}
        {grid.map((row, ri) => row.map((t, ci) => {
          if (t === null) return null
          return <rect key={`${ri}-${ci}`} x={pad + ci*cw} y={rowY(ri)} width={cw+0.5} height={ch+0.5}
            fill={tempToHex(t, tmin, tmax)} />
        }))}
        {/* Coastline strokes along land/sea boundaries. */}
        {coast.map((s, i) => (
          <line key={`c-${i}`} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
            stroke="rgba(226,232,240,.7)" strokeWidth="1.2" />
        ))}
        {/* Lat/lon graticule (whole degrees), self-contained. */}
        {geo && gratLons.map(d => (
          <g key={`gx-${d}`}>
            <line x1={lonToX(d)} y1={pad} x2={lonToX(d)} y2={H-pad}
              stroke="rgba(255,255,255,.10)" strokeWidth="0.6" />
            <text x={lonToX(d) + 2} y={H - pad - 3} fill="rgba(255,255,255,.35)" fontSize="8">
              {Math.abs(d)}°W</text>
          </g>
        ))}
        {geo && gratLats.map(d => (
          <g key={`gy-${d}`}>
            <line x1={pad} y1={latToY(d)} x2={W-pad} y2={latToY(d)}
              stroke="rgba(255,255,255,.10)" strokeWidth="0.6" />
            <text x={pad + 2} y={latToY(d) - 2} fill="rgba(255,255,255,.35)" fontSize="8">
              {d}°N</text>
          </g>
        ))}
        {/* Washington DC boundary diamond (real district borders). */}
        {geo && lats.length > 1 && (
          <polyline points={DC_DIAMOND.map(([la, lo]) => `${lonToX(lo)},${latToY(la)}`).join(' ')}
            fill="none" stroke="rgba(255,255,255,.45)" strokeWidth="1" />
        )}
        {/* Real reference cities as small labeled dots. */}
        {geo && lats.length > 1 && DC_CITIES.filter(c => !c.major && inView(c.lat, c.lon)).map(c => (
          <g key={c.name}>
            <circle cx={lonToX(c.lon)} cy={latToY(c.lat)} r="2.5" fill="rgba(255,255,255,.8)" />
            <text x={lonToX(c.lon) + 4} y={latToY(c.lat) + 3} fill="rgba(255,255,255,.7)" fontSize="8.5"
              style={{ paintOrder: 'stroke', stroke: '#000', strokeWidth: 2 }}>{c.name}</text>
          </g>
        ))}
        {dcLat && dcLon && lats.length > 1 && (
          <g>
            <circle cx={lonToX(dcLon)} cy={latToY(dcLat)} r="7" fill="#fff" stroke="#ED1C24" strokeWidth="3" />
            <text x={lonToX(dcLon) + 12} y={latToY(dcLat) + 4} fill="#fff" fontSize="12" fontWeight="700"
              style={{ paintOrder: 'stroke', stroke: '#000', strokeWidth: 3 }}>Washington DC</text>
          </g>
        )}
      </svg>
      <div style={{ marginTop: '.4rem', display: 'flex', alignItems: 'center', gap: '.5rem' }}>
        <span style={{ fontSize: '.65rem', color: '#71717a', flexShrink: 0 }}>{tmin.toFixed(1)}°C</span>
        <div style={{ flex: 1, height: 10, borderRadius: 4,
          background: 'linear-gradient(to right, #0000ff, #ffffff, #ff0000)' }} />
        <span style={{ fontSize: '.65rem', color: '#71717a', flexShrink: 0 }}>{tmax.toFixed(1)}°C</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.6rem', color: '#52525b', marginTop: '.15rem' }}>
        <span><span style={{ color: '#ED1C24' }}>●</span> Washington DC</span>
        <span>{label} · {rows}×{cols} grid · real ERA5 temperature · geographic overlay</span>
      </div>
    </div>
  )
}

function LeafletTempMap({ gridData, dcLat, dcLon, label, height = 360 }) {
  const mapRef = useRef(null)
  const leafletRef = useRef(null)

  useEffect(() => {
    let L, map, cancelled = false
    ;(async () => {
      L = await import('leaflet')
      if (cancelled || !mapRef.current) return

      // Inject Leaflet CSS if not already present
      if (!document.getElementById('leaflet-css')) {
        const link = document.createElement('link')
        link.id = 'leaflet-css'; link.rel = 'stylesheet'
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
        document.head.appendChild(link)
      }

      if (leafletRef.current) { leafletRef.current.remove(); leafletRef.current = null }

      const lats = gridData.lat || [], lons = gridData.lon || [], grid = gridData.temp_c || []
      if (!lats.length) return

      const centerLat = (lats[0] + lats[lats.length-1]) / 2
      const centerLon = (lons[0] + lons[lons.length-1]) / 2
      const zoom = gridData.resolution_deg <= 0.15 ? 9 : 7

      map = L.default.map(mapRef.current, { zoomControl: true, scrollWheelZoom: false })
      leafletRef.current = map

      L.default.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        opacity: 0.85,
      }).addTo(map)

      map.setView([centerLat, centerLon], zoom)

      // Temperature overlay: one rectangle per grid cell
      const allTemps = grid.flat().filter(t => t !== null)
      const tmin = Math.min(...allTemps), tmax = Math.max(...allTemps)
      const stepLat = lats.length > 1 ? Math.abs(lats[1] - lats[0]) : 0.25
      const stepLon = lons.length > 1 ? Math.abs(lons[1] - lons[0]) : 0.25

      grid.forEach((row, ri) => {
        row.forEach((t, ci) => {
          if (t === null) return
          const lat = lats[ri], lon = lons[ci]
          L.default.rectangle(
            [[lat - stepLat/2, lon - stepLon/2], [lat + stepLat/2, lon + stepLon/2]],
            { color: 'none', fillColor: tempToHex(t, tmin, tmax), fillOpacity: 0.55,
              weight: 0, interactive: false }
          ).addTo(map)
        })
      })

      // DC marker
      if (dcLat && dcLon) {
        L.default.circleMarker([dcLat, dcLon], {
          radius: 8, color: '#ED1C24', weight: 3, fillColor: '#fff', fillOpacity: 0.9,
        }).bindTooltip(`Washington DC<br>${dcLat.toFixed(2)}°N ${Math.abs(dcLon).toFixed(2)}°W`, { permanent: false })
          .addTo(map)
      }
    })()
    return () => { cancelled = true; if (leafletRef.current) { leafletRef.current.remove(); leafletRef.current = null } }
  }, [gridData, dcLat, dcLon])

  const allTemps = (gridData.temp_c || []).flat().filter(t => t !== null)
  const tmin = allTemps.length ? Math.min(...allTemps) : 15
  const tmax = allTemps.length ? Math.max(...allTemps) : 42

  return (
    <div>
      <div ref={mapRef} style={{ width: '100%', height, borderRadius: '.5rem',
        border: '1px solid #27272a', overflow: 'hidden', background: '#0a0a0b' }} />
      {/* Color bar legend */}
      <div style={{ marginTop: '.4rem', display: 'flex', alignItems: 'center', gap: '.5rem' }}>
        <span style={{ fontSize: '.65rem', color: '#71717a', flexShrink: 0 }}>{tmin.toFixed(1)}°C</span>
        <div style={{ flex: 1, height: 10, borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            width: '100%', height: '100%',
            background: 'linear-gradient(to right, #0000ff, #ffffff, #ff0000)',
          }} />
        </div>
        <span style={{ fontSize: '.65rem', color: '#71717a', flexShrink: 0 }}>{tmax.toFixed(1)}°C</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.6rem', color: '#52525b', marginTop: '.15rem' }}>
        <span><span style={{ color: '#ED1C24' }}>●</span> Washington DC</span>
        <span>{label} · {(gridData.temp_c || []).length}×{(gridData.temp_c?.[0] || []).length} grid · OpenStreetMap</span>
      </div>
    </div>
  )
}

// ── ORBIT-2 four-act story, daisy-chained: pretrained -> OOD gap -> ───────────
//    first finetune (broad) -> second finetune (DC-targeted). Acts reveal one
//    at a time (auto-play), each with its own DC map vs the PRISM truth.
function ORBIT2StoryViz({ result }) {
  const m = result.metrics || {}
  const dc = m.dc_tmax_mae_c || {}
  const dcCore = m.dc_core_tmax_mae_c || {}
  const conus = m.conus_tmax_mae_c || {}
  const maps = result.maps || {}
  const acts = result.acts || []
  const dcLat = 38.89, dcLon = -77.04

  // Sequential reveal: `shown` = how many acts are visible; `active` = focused act.
  const [shown, setShown] = useState(1)
  const [active, setActive] = useState(0)
  const [playing, setPlaying] = useState(true)

  // Auto-advance one act at a time while playing.
  useEffect(() => {
    if (!playing) return
    if (shown >= acts.length) { setPlaying(false); return }
    const t = setTimeout(() => {
      setShown(s => Math.min(s + 1, acts.length))
      setActive(a => Math.min(a + 1, acts.length - 1))
    }, 3200)
    return () => clearTimeout(t)
  }, [playing, shown, acts.length])

  const replay = () => { setShown(1); setActive(0); setPlaying(true) }

  // Per-act accent color: gray (pretrained), red (OOD gap), green shades (finetunes).
  const actColor = ['#a1a1aa', '#ff8f93', '#4ade80', '#22c55e']
  const cur = acts[active] || {}
  const curMapKey = cur.map_key || 'pretrained'
  const curMae = dc[cur.stage === 'baseline' ? 'bilinear' : cur.stage] ?? dc[curMapKey]

  return (
    <div>
      <div style={{ marginBottom: '.5rem', fontSize: '.95rem', color: '#f5f5f7' }}>
        <strong>ORBIT-2 downscaling — a real robustness &amp; finetuning story</strong>
        <div style={{ fontSize: '.72rem', color: '#71717a', marginTop: '.2rem' }}>
          {result.task}
        </div>
        <div style={{ fontSize: '.68rem', color: '#4ade80', marginTop: '.15rem' }}>
          {result.provenance || 'Replay of real GPU runs — measured, not synthesized.'}
        </div>
      </div>

      {/* Progress stepper + playback controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', marginBottom: '.9rem', flexWrap: 'wrap' }}>
        {acts.map((a, i) => (
          <button key={a.n} onClick={() => { setActive(i); setShown(Math.max(shown, i + 1)); setPlaying(false) }}
            style={{
              display: 'flex', alignItems: 'center', gap: '.35rem', padding: '.28rem .6rem',
              borderRadius: '9999px', cursor: 'pointer', fontSize: '.7rem', fontWeight: 700,
              background: i === active ? actColor[i] : (i < shown ? 'rgba(255,255,255,.06)' : 'transparent'),
              border: '1px solid', borderColor: i === active ? actColor[i] : '#3f3f46',
              color: i === active ? '#0a0a0b' : (i < shown ? '#e4e4e7' : '#52525b'),
              opacity: i < shown ? 1 : 0.5,
            }}>
            <span style={{
              width: 16, height: 16, borderRadius: '50%', display: 'inline-flex',
              alignItems: 'center', justifyContent: 'center', fontSize: '.6rem',
              background: i === active ? '#0a0a0b' : actColor[i],
              color: i === active ? actColor[i] : '#0a0a0b',
            }}>{a.n}</span>
            {a.title}
          </button>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '.4rem' }}>
          <button className="btn btn-ghost" style={{ fontSize: '.72rem', padding: '.25rem .6rem' }}
            onClick={() => setPlaying(p => !p)} disabled={shown >= acts.length && !playing}>
            {playing ? '⏸ Pause' : (shown >= acts.length ? '⏵ Play' : '▶ Continue')}
          </button>
          <button className="btn btn-ghost" style={{ fontSize: '.72rem', padding: '.25rem .6rem' }}
            onClick={replay}>↺ Replay</button>
        </div>
      </div>

      {/* Active act narration card */}
      <div className="card" style={{ padding: '1rem', marginBottom: '1rem', borderLeft: `4px solid ${actColor[active]}` }}>
        <div style={{ fontSize: '.62rem', color: '#71717a', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.05em' }}>
          Act {cur.n} of {acts.length}
        </div>
        <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#f5f5f7', margin: '.25rem 0 .4rem' }}>
          {cur.title}
        </div>
        <div style={{ fontSize: '.85rem', color: '#d4d4d8', lineHeight: 1.5, marginBottom: '.5rem' }}>
          {cur.text}
        </div>
        {cur.detail && (
          <div style={{ fontSize: '.78rem', color: '#a1a1aa', marginBottom: '.5rem' }}>{cur.detail}</div>
        )}
        <div style={{ fontSize: '.95rem', fontWeight: 800, color: actColor[active] }}>{cur.stat}</div>
      </div>

      {/* Two maps side by side: this act's prediction vs the PRISM truth */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '.75rem' }}>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem', fontSize: '.68rem' }}>
            {curMapKey === 'coarse' ? 'Coarse input (what the model sees)'
              : `${maps[curMapKey]?.label || cur.title}${curMae !== undefined ? ` — MAE ${curMae}°C` : ''}`}
          </div>
          {maps[curMapKey] && (
            <SVGTempMap gridData={maps[curMapKey]} dcLat={dcLat} dcLon={dcLon}
              label={maps[curMapKey].label} height={300} />
          )}
        </div>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem', fontSize: '.68rem' }}>
            {maps.truth?.label || 'Real observations (fine grid)'}
          </div>
          {maps.truth && (
            <SVGTempMap gridData={maps.truth} dcLat={dcLat} dcLon={dcLon}
              label={maps.truth.label} height={300} />
          )}
        </div>
      </div>

      {/* Running scoreboard: DC tmax MAE (whole window + urban core) per stage */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(150px,1fr))', gap: '.6rem', marginBottom: '.5rem' }}>
        {[
          { label: 'Pretrained (OOD)', key: 'pretrained', color: '#a1a1aa', reveal: 1 },
          { label: 'Bilinear baseline', key: 'bilinear', color: '#71717a', reveal: 2 },
          { label: 'Physics-residual head', key: 'urban', color: '#22c55e', reveal: 3 },
          { label: '+ analytic physics', key: 'physics', color: '#16a34a', reveal: 4 },
        ].map(s => (
          <div key={s.key} className="card" style={{ padding: '.6rem', textAlign: 'center',
            opacity: shown >= s.reveal ? 1 : 0.25, transition: 'opacity .4s' }}>
            <div className="section-label" style={{ marginBottom: '.2rem', fontSize: '.58rem' }}>{s.label}</div>
            <div style={{ fontSize: '1.02rem', fontWeight: 800, color: s.color }}>
              {dc[s.key] !== undefined ? `${dc[s.key]}°C` : '—'}
            </div>
            {dcCore[s.key] !== undefined && (
              <div style={{ fontSize: '.62rem', color: '#a1a1aa', marginTop: '.15rem' }}>
                core {dcCore[s.key]}°C
              </div>
            )}
          </div>
        ))}
      </div>

      {shown >= acts.length && dcCore.urban !== undefined && dcCore.bilinear !== undefined && (
        <div style={{ fontSize: '.82rem', color: '#22c55e', margin: '.5rem 0 1rem', fontWeight: 700 }}>
          ✓ A ~10k-param physics-residual head (GHSL urban density + distance-to-water) cuts the
          urban-core error {dcCore.bilinear}°C → {dcCore.urban}°C — beating bilinear by{' '}
          {Math.round(100 * (dcCore.bilinear - dcCore.urban) / dcCore.bilinear)}% exactly where sub-grid
          physics lives. Known physics learned as a residual, on independent Open-Meteo data.
        </div>
      )}

      <div style={{ fontSize: '.72rem', color: '#52525b' }}>
        PRISM-trained ORBIT-2 8M ViT applied to independent Open-Meteo ERA5 DC data (never seen in
        training) — a true out-of-distribution test. 4× super-resolution · DC urban-core tmax MAE:
        bilinear {dcCore.bilinear}°C → physics-residual head {dcCore.urban}°C. The head corrects the
        bilinear field with EU JRC GHSL built-up surface + a coastal distance-to-water field; it beats
        bilinear both at the urban core and across the whole land window
        ({dc.bilinear}°C → {dc.urban}°C), largest exactly where sub-grid physics lives.
        All errors measured on land.
      </div>
    </div>
  )
}

function DCDownscalingViz({ result }) {
  const coarse = result.coarse || {}
  const fine = result.fine || {}
  const activeEvent = result.event_key || 'july16_2024'

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: '.75rem', fontSize: '.95rem', color: '#f5f5f7' }}>
        <strong>{result.label || result.date}</strong>
        {result.peak_temp_c && (
          <span style={{ marginLeft: '.75rem', color: '#ff8f93', fontWeight: 700 }}>
            Peak {result.peak_temp_c}°C / {result.peak_temp_f}°F
          </span>
        )}
      </div>

      {/* Event selector */}
      {result.available_events && Object.keys(result.available_events).length > 1 && (
        <div style={{ marginBottom: '1rem', fontSize: '.78rem', color: '#71717a' }}>
          <span style={{ marginRight: '.5rem' }}>Dataset:</span>
          {Object.entries(result.available_events).map(([k, label]) => (
            <span key={k} style={{
              marginRight: '.5rem', padding: '.2rem .55rem', borderRadius: '9999px',
              background: activeEvent === k ? '#ED1C24' : 'transparent',
              border: '1px solid', borderColor: activeEvent === k ? '#ED1C24' : '#3f3f46',
              color: activeEvent === k ? '#fff' : '#a1a1aa', fontSize: '.7rem',
            }}>{label}</span>
          ))}
          <span style={{ color: '#52525b' }}>— re-run to switch</span>
        </div>
      )}

      {/* Stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(130px,1fr))', gap: '.65rem', marginBottom: '1rem' }}>
        {[
          { label: 'Date', value: result.date || '—' },
          { label: 'Peak Temp', value: result.peak_temp_c ? `${result.peak_temp_c}°C` : '—', color: '#ff8f93' },
          { label: 'Coarse Grid', value: coarse.label || `${coarse.resolution_deg}°` },
          { label: 'Fine Grid', value: fine.label || `${fine.resolution_deg}°` },
          { label: 'Coarse Peak', value: coarse.peak_temp_c ? `${coarse.peak_temp_c}°C` : '—' },
          { label: 'Fine Peak', value: fine.peak_temp_c ? `${fine.peak_temp_c}°C` : '—', color: '#ff8f93' },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: '.65rem', textAlign: 'center' }}>
            <div className="section-label" style={{ marginBottom: '.25rem', fontSize: '.62rem' }}>{s.label}</div>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: s.color || '#f5f5f7' }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Two Leaflet maps side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '.75rem' }}>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem', fontSize: '.68rem' }}>
            Coarse — {coarse.label || '0.25° ERA5'} (input)
          </div>
          <SVGTempMap gridData={coarse} dcLat={result.dc?.lat} dcLon={result.dc?.lon}
            label={coarse.label || 'ERA5 0.25°'} height={320} />
        </div>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem', fontSize: '.68rem' }}>
            Fine — {fine.label || '0.1° ERA5-Land'} (ORBIT-2 output)
          </div>
          <SVGTempMap gridData={fine} dcLat={result.dc?.lat} dcLon={result.dc?.lon}
            label={fine.label || 'ERA5-Land 0.1°'} height={320} />
        </div>
      </div>

      <div style={{ fontSize: '.72rem', color: '#52525b' }}>
        Real ERA5 data · {result.source || 'Open-Meteo'} · ORBIT-2 4× super-resolution · map © OpenStreetMap contributors
      </div>
    </div>
  )
}

// ── GP-MoLFormer Pair-Tuning Viz ─────────────────────────────────────────────
function MolefineTuneViz({ result }) {
  const b = result.before || {}, a = result.after || {}
  const prop = (result.property || 'qed').toUpperCase()
  const epochs = result.epochs || []

  function delta(av, bv, higher = true) {
    if (av == null || bv == null) return null
    const d = av - bv
    const better = higher ? d > 0 : d < 0
    return { d: Math.abs(d).toFixed(3), better, pct: Math.abs(d / (bv || 1) * 100).toFixed(0) }
  }

  const metrics = [
    { label: 'QED Mean', bv: b.qed_mean, av: a.qed_mean, higher: true },
    { label: 'logP Mean', bv: b.logp_mean, av: a.logp_mean, higher: false },
    { label: 'Lipinski Pass', bv: b.lipinski_pass_rate, av: a.lipinski_pass_rate, higher: true, pct: true },
  ]

  const curveDat = epochs.map(e => ({ ep: e.ep, loss: e.loss }))

  return (
    <div>
      <div style={{ marginBottom: '.75rem', fontSize: '.88rem', color: '#a1a1aa', lineHeight: 1.5 }}>
        <strong style={{ color: '#f5f5f7' }}>GP-MoLFormer pair-tuning</strong>
        {' — '} property optimized: <span style={{ color: '#38bdf8', fontWeight: 700 }}>{prop}</span>
        {result.num_epochs && ` · ${result.num_epochs} epochs`}
      </div>

      {/* Before / after comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(170px,1fr))', gap: '.75rem', marginBottom: '1rem' }}>
        {metrics.map(({ label, bv, av, higher, pct }) => {
          const d = delta(av, bv, higher)
          return (
            <div key={label} className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
              <div className="section-label" style={{ marginBottom: '.3rem' }}>{label}</div>
              <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '.6rem', color: '#52525b' }}>Before</div>
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: '#71717a' }}>
                    {bv != null ? (pct ? `${(bv*100).toFixed(0)}%` : bv.toFixed(3)) : '—'}
                  </div>
                </div>
                <div style={{ fontSize: '1.2rem', color: d?.better ? '#21c77a' : '#f5a524' }}>
                  {d?.better ? '↑' : '↓'}
                </div>
                <div>
                  <div style={{ fontSize: '.6rem', color: '#52525b' }}>After</div>
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: d?.better ? '#21c77a' : '#f5a524' }}>
                    {av != null ? (pct ? `${(av*100).toFixed(0)}%` : av.toFixed(3)) : '—'}
                  </div>
                </div>
              </div>
              {d && (
                <div style={{ fontSize: '.65rem', color: d.better ? '#21c77a' : '#f5a524', marginTop: '.2rem' }}>
                  {d.better ? '+' : '-'}{d.d} ({d.pct}%)
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        {/* Loss curve */}
        {curveDat.length > 0 && (
          <div className="card" style={{ padding: '.75rem' }}>
            <div className="section-label" style={{ marginBottom: '.4rem' }}>Pair-tuning loss</div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={curveDat} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                <XAxis dataKey="ep" tick={{ fill: '#71717a', fontSize: 10 }} />
                <YAxis tick={{ fill: '#71717a', fontSize: 10 }} domain={['auto','auto']} />
                <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 11 }} />
                <Line type="monotone" dataKey="loss" stroke="#38bdf8" dot={false}
                  strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Before/after molecule samples */}
        <div className="card" style={{ padding: '.75rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem' }}>Generated molecules (sample)</div>
          {(['before','after']).map(stage => {
            const mols = result[stage]?.molecules?.slice(0, 4) || []
            return (
              <div key={stage} style={{ marginBottom: '.5rem' }}>
                <div style={{ fontSize: '.68rem', fontWeight: 700, color: stage === 'before' ? '#71717a' : '#21c77a',
                  marginBottom: '.25rem', textTransform: 'uppercase', letterSpacing: '.05em' }}>{stage}</div>
                {mols.map((m, i) => (
                  <div key={i} style={{ marginBottom: '.2rem', display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', gap: '.5rem' }}>
                    <code style={{ fontSize: '.65rem', color: '#7dd3fc', flex: 1,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {m.smiles}
                    </code>
                    {m.qed != null && (
                      <span className={`badge badge-${m.qed > 0.7 ? 'ok' : m.qed > 0.5 ? 'info' : 'muted'}`}
                        style={{ fontSize: '.6rem', flexShrink: 0 }}>
                        QED {m.qed.toFixed(3)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      </div>

      <div style={{ fontSize: '.75rem', color: '#52525b', lineHeight: 1.5 }}>
        {result.model || 'GP-MoLFormer'} · Pair-tuning PEFT (IBM Research) ·
        Backbone frozen; only soft-prompt tokens are trained
      </div>
    </div>
  )
}

function ResultView({ result, runId }) {
  if (!result) return null
  switch (result.type) {
    case 'downscaling': return <DownscalingViz result={result} runId={runId} />
    case 'atomistic_energy': return <EnergyViz result={result} />
    case 'training_convergence': return <TrainingConvergenceViz result={result} />
    case 'training_telemetry': return <TelemetryPanelContainer result={result} />
    case 'dc_downscaling': return <DCDownscalingViz result={result} />
    case 'orbit2_story': return <ORBIT2StoryViz result={result} />
    case 'molecule_finetune': return <MolefineTuneViz result={result} />
    case 'weather_forecast': return <WeatherViz result={result} />
    case 'atomistic_properties':
    case 'crystal_generation': return <MaterialsViz result={result} />
    case 'molecule_generation':
    case 'segmentation': return <HealthcareViz result={result} />
    case 'physics_rollout': return <PhysicsViz result={result} />
    default: return (
      <pre style={{ color: '#94a3b8', fontSize: '.8rem', overflow: 'auto' }}>
        {JSON.stringify(result, null, 2)}
      </pre>
    )
  }
}

// Frontend safety-net denylist mirroring the backend log filter (Part D).
const _LOG_DENY = [
  /Sorry!\s+You were supposed to get help/i,
  /Couldn't open the help file/i,
  /help-btl-vader\.txt|btl_vader/i,
  /PMIx?\b.*(WARNING|not found)/i,
]
const _cleanLog = lines => (lines || []).filter(l => !_LOG_DENY.some(re => re.test(l)))

// Live GPU telemetry while a job runs: polls the live cross-node endpoint (Omnistat
// VM on the compute node) every 2.5s and renders growing charts. Omnistat's first
// push is ~1 min in, so we show a "warming up" state until data arrives, then live
// charts, then the harvested final telemetry on completion.
function PerformanceTab({ runId, result, runState }) {
  const catalog = useTelemetryCatalog()
  const [selected, setSelected] = useState(null)
  const [liveTel, setLiveTel] = useState(null)     // {status, telemetry?, node?}
  const [log, setLog] = useState([])

  // Seed selection from catalog defaults (or the final result's own keys).
  useEffect(() => {
    if (selected) return
    const fromResult = Object.keys(result?.telemetry?.series || {}).filter(k => k !== 't_s')
    if (fromResult.length) setSelected(fromResult)
    else if (catalog?.defaults) setSelected(catalog.defaults)
  }, [catalog, result, selected])

  const running = runState === 'running' || runState === 'queued' || runState === 'pending'

  // Poll the live endpoint (keyed to the current selection) + the log while running.
  // Completion detection is owned by StepAnalyze (runs regardless of active tab).
  useEffect(() => {
    if (!runId || !selected) return
    let stop = false
    const tick = async () => {
      try {
        const t = await api.jobTelemetryLive(runId, selected)
        if (!stop && t) setLiveTel(t)
      } catch { /* ignore */ }
      try {
        const l = await api.jobLog(runId)
        if (!stop && l?.lines) setLog(_cleanLog(l.lines).slice(-14))
      } catch { /* ignore */ }
    }
    tick()
    if (running && !result?.telemetry) {
      const id = setInterval(tick, 2500)
      return () => { stop = true; clearInterval(id) }
    }
    return () => { stop = true }
  }, [runId, running, selected, result])

  if (!catalog || !selected) return <Spinner size={24} />

  // Final (completed) telemetry takes precedence; else live payload; else warmup.
  const finalTel = result?.telemetry ? result : null
  const liveShown = liveTel?.status === 'live' && liveTel.telemetry
    ? { telemetry: liveTel.telemetry, n_gpus: result?.n_gpus, runtime_s: liveTel.telemetry.runtime_s }
    : null
  const shown = finalTel || liveShown

  return (
    <div>
      {shown ? (
        <TelemetryPanel result={shown} catalog={catalog} selected={selected}
          setSelected={setSelected} live={!finalTel} />
      ) : (
        <div>
          <MetricPicker catalog={catalog} selected={selected} onChange={setSelected} />
          <div className="card" style={{ padding: '1rem', marginBottom: '1rem', display: 'flex',
            alignItems: 'center', gap: '.75rem' }}>
            <Spinner size={18} />
            <div style={{ fontSize: '.85rem', color: '#a1a1aa' }}>
              {liveTel?.status === 'starting'
                ? `Omnistat is collecting on ${liveTel.node || 'the MI355X node'} — first telemetry push lands ~1 min into the run; live charts appear then.`
                : 'Waiting for the job to start on a compute node…'}
            </div>
          </div>
        </div>
      )}
      {/* Streaming log (noise-filtered) below the charts */}
      {log.length > 0 && (
        <pre style={{ background: 'rgba(5,5,6,0.8)', borderRadius: '.5rem', padding: '.75rem',
          fontSize: '.72rem', color: '#7dd3fc', overflow: 'auto', maxHeight: 220, marginTop: '1rem' }}>
          {log.join('\n')}
        </pre>
      )}
    </div>
  )
}

export function StepAnalyze() {
  const { runId, result, model, setStep, runState, setResult, setRunState } = useStore()
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(true)
  // Performance tab is the default and is available immediately once a run exists.
  const [tab, setTab] = useState('performance')

  useEffect(() => {
    if (!runId) { setLoading(false); return }
    api.jobFiles(runId).then(setFiles).finally(() => setLoading(false))
  }, [runId, runState])

  // Completion detection lives HERE (not in a tab component) so it runs regardless
  // of which tab is active: the Run step navigates here mid-run for live training,
  // so this step owns harvesting the final result (loss curve + telemetry + metrics)
  // when the SLURM job finishes. Polls until terminal.
  useEffect(() => {
    if (!runId) return
    const running = runState === 'running' || runState === 'queued' || runState === 'pending'
    if (!running || result?.loss_curve) return
    let stop = false
    const id = setInterval(async () => {
      try {
        const job = await api.job(runId)
        if (stop) return
        if (job?.state === 'completed' || job?.state === 'failed') {
          if (job.result) setResult(job.result)
          setRunState(job.state)
          clearInterval(id)
        }
      } catch { /* ignore */ }
    }, 3000)
    return () => { stop = true; clearInterval(id) }
  }, [runId, runState, result])

  const tabBtn = (id, label) => (
    <button onClick={() => setTab(id)}
      style={{
        padding: '.5rem 1.1rem', fontSize: '.85rem', fontWeight: 700, cursor: 'pointer',
        background: 'none', border: 'none', borderBottom: tab === id ? '2px solid #ED1C24' : '2px solid transparent',
        color: tab === id ? '#f5f5f7' : '#71717a',
      }}>{label}</button>
  )

  return (
    <div>
      <StepHeader
        title="Analyze"
        sub={`${model?.name || model?.slug} — performance & output`}
      />
      <button className="btn btn-ghost" style={{ marginBottom: '1rem', fontSize: '.8rem' }}
        onClick={() => setStep(3)}>← Back</button>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '.5rem', borderBottom: '1px solid #27272a',
        marginBottom: '1.2rem' }}>
        {tabBtn('performance', 'Performance')}
        {tabBtn('results', 'Results')}
      </div>

      {tab === 'performance' && (
        <PerformanceTab runId={runId} result={result} runState={runState} />
      )}

      {tab === 'results' && (
        <div>
          {loading && <Spinner size={28} />}
          {result && (
            <div style={{ marginBottom: '1.5rem' }}>
              <div className="section-label" style={{ marginBottom: '.75rem' }}>Output Summary</div>
              {/* Training runs: show loss convergence + accuracy here (telemetry is on
                  the Performance tab). Everything else uses the standard result view. */}
              {result.type === 'training_telemetry'
                ? <TrainingResultsView result={result} />
                : <ResultView result={result} runId={runId} />}
            </div>
          )}
          {files.length > 0 && (
            <div>
              <div className="section-label" style={{ marginBottom: '.6rem' }}>Output Files</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
                {files.map(f => (
                  <div key={f.name} className="card" style={{ padding: '.6rem 1rem', display: 'flex',
                    justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontWeight: 600, fontSize: '.88rem', color: '#f5f5f7' }}>{f.name}</span>
                      <span style={{ fontSize: '.72rem', color: '#52525b', marginLeft: '.5rem' }}>
                        {(f.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                    <a href={`/api/jobs/${runId}/files/${f.name}`}
                      style={{ color: '#7dd3fc', fontSize: '.75rem' }}
                      download={f.name}>Download</a>
                  </div>
                ))}
              </div>
            </div>
          )}
          {!result && !loading && (
            <div style={{ color: '#52525b', textAlign: 'center', padding: '2rem' }}>
              Results appear here when the run completes.
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
        <button className="btn btn-ghost" onClick={() => setStep(0)}>
          ← Start new run
        </button>
      </div>
    </div>
  )
}
