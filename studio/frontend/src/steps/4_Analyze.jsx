import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { StepHeader, Spinner, Stat, StatGrid } from '../components/ui'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, ReferenceLine, Cell,
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

function MoleculeViewer({ atoms, height = 340 }) {
  const hostRef = useRef(null)
  const viewerRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    if (!atoms || atoms.length === 0 || !hostRef.current) return
    ;(async () => {
      const mod = await import('3dmol')
      const $3Dmol = mod.default || mod
      if (cancelled || !hostRef.current) return
      hostRef.current.innerHTML = ''
      const viewer = $3Dmol.createViewer(hostRef.current, {
        backgroundColor: '#0a0a0b',
      })
      viewerRef.current = viewer
      const model = viewer.addModel()
      model.addAtoms(atoms.map((a, i) => ({
        elem: a.element, x: a.x, y: a.y, z: a.z, serial: i,
      })))
      // Infer bonds from interatomic distances (addAtoms doesn't do this itself).
      try { model.assignBonds?.() } catch { /* older 3dmol: bonds via distance below */ }
      // Per-element sphere colors + sticks for bonds.
      viewer.setStyle({}, {
        sphere: { scale: 0.30, colorscheme: 'Jmol' },
        stick: { radius: 0.14, colorscheme: 'Jmol' },
      })
      viewer.zoomTo()
      viewer.render()
      viewer.zoom(1.2, 600)
      viewer.spin('y', 0.5)
    })()
    return () => {
      cancelled = true
      try { viewerRef.current?.clear?.() } catch { /* noop */ }
    }
  }, [atoms])

  if (!atoms || atoms.length === 0) {
    return (
      <div style={{ padding: '1rem', color: '#52525b', fontSize: '.82rem', textAlign: 'center',
        background: 'rgba(5,5,6,0.8)', borderRadius: '.5rem' }}>
        No atomic coordinates in this result.
      </div>
    )
  }

  // Composition legend (unique elements)
  const uniq = [...new Set(atoms.map(a => a.element))]
  return (
    <div>
      <div ref={hostRef} style={{ position: 'relative', width: '100%', height,
        borderRadius: '.5rem', overflow: 'hidden', border: '1px solid #27272a',
        background: '#0a0a0b' }} />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem', marginTop: '.5rem' }}>
        {uniq.map(el => (
          <span key={el} style={{ display: 'inline-flex', alignItems: 'center', gap: '.3rem',
            fontSize: '.72rem', color: '#a1a1aa' }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%',
              background: _CPK[el] || '#dd77ff', display: 'inline-block',
              border: '1px solid rgba(255,255,255,.2)' }} />
            {el}
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
        Material: <span style={{ fontWeight: 800, color: '#7dd3fc' }}>{result.formula}</span>
        <span style={{ color: '#71717a' }}> · {result.n_atoms} atoms</span>
        {result.model_variant && (
          <span className="badge badge-amd" style={{ marginLeft: '.5rem' }}>{result.model_variant} model</span>
        )}
      </div>

      {/* Two columns: 3D structure | prediction stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px,1fr) minmax(260px,1fr)',
        gap: '1rem', alignItems: 'start' }}>
        <div className="card" style={{ padding: '.6rem' }}>
          <div className="section-label" style={{ marginBottom: '.4rem' }}>Atomistic Structure</div>
          <MoleculeViewer atoms={atoms} />
        </div>

        <div>
          <StatGrid min={130}>
            {result.predicted_energy_ev_per_atom !== undefined &&
              <Stat label="Predicted Energy" value={`${result.predicted_energy_ev_per_atom}`} sub="eV/atom" color="#21c77a" />}
            {result.dft_reference_ev_per_atom !== undefined &&
              <Stat label="DFT Reference" value={`${result.dft_reference_ev_per_atom}`} sub="eV/atom (experiment)" />}
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
          <MoleculeViewer atoms={result.atoms || []} />
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
      <div style={{ marginBottom: '1rem', fontSize: '.9rem', color: '#a1a1aa', lineHeight: 1.5 }}>
        Real HydraGNN energy-model training on Alexandria DFT data. Scaling to 8 GPUs lets us
        train on <b style={{ color: '#f5f5f7' }}>{(r8.n_samples / r1.n_samples).toFixed(1)}×</b> more
        data in comparable wall-clock, improving accuracy.
      </div>

      {/* Loss convergence */}
      <div className="card" style={{ padding: '.8rem', marginBottom: '1rem' }}>
        <div className="section-label" style={{ marginBottom: '.5rem' }}>Validation Loss Convergence</div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve} margin={{ top: 6, right: 16, bottom: 6, left: 0 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis dataKey="ep" tick={{ fill: '#71717a', fontSize: 11 }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#71717a', fontSize: 11 }} />
            <YAxis tick={{ fill: '#71717a', fontSize: 11 }} domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="val1" name="1-GPU val" stroke="#f5a524" dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="monotone" dataKey="val8" name="8-GPU val" stroke="#38bdf8" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        {/* Scaling: data throughput */}
        <div className="card" style={{ padding: '.8rem' }}>
          <div className="section-label" style={{ marginBottom: '.5rem' }}>Training Set Size (per epoch)</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={speedSamples} margin={{ top: 6, right: 16, bottom: 6, left: 0 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
              <YAxis tick={{ fill: '#71717a', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 12 }} />
              <Bar dataKey="v" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                {speedSamples.map((s, i) => <Cell key={i} fill={s.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {r8.throughput_it_s && (
            <div style={{ fontSize: '.74rem', color: '#52525b', marginTop: '.3rem', textAlign: 'center' }}>
              8-GPU throughput ≈ {r8.throughput_it_s} it/s/rank ({(r8.throughput_it_s * 8).toFixed(0)} it/s aggregate)
            </div>
          )}
        </div>

        {/* Parity scatter (8-GPU) */}
        <div className="card" style={{ padding: '.8rem' }}>
          <div className="section-label" style={{ marginBottom: '.5rem' }}>8-GPU: Predicted vs DFT (held-out)</div>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ top: 6, right: 16, bottom: 6, left: 0 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="true" name="DFT" domain={[lo, hi]}
                tick={{ fill: '#71717a', fontSize: 10 }}
                label={{ value: 'DFT (eV/atom)', position: 'insideBottom', offset: -2, fill: '#71717a', fontSize: 10 }} />
              <YAxis type="number" dataKey="pred" name="Predicted" domain={[lo, hi]}
                tick={{ fill: '#71717a', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#141416', border: '1px solid #27272a', fontSize: 12 }}
                cursor={{ strokeDasharray: '3 3' }} />
              <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#52525b" strokeDasharray="4 4" />
              <Scatter data={parity} fill="#38bdf8" fillOpacity={0.5} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Final stats comparison table */}
      <div className="card" style={{ padding: '.8rem' }}>
        <div className="section-label" style={{ marginBottom: '.5rem' }}>Final Model Comparison</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.85rem' }}>
          <thead>
            <tr style={{ color: '#71717a', textAlign: 'right' }}>
              <th style={{ textAlign: 'left', padding: '.3rem .5rem' }}>Metric</th>
              <th style={{ padding: '.3rem .5rem' }}>1-GPU</th>
              <th style={{ padding: '.3rem .5rem' }}>8-GPU</th>
              <th style={{ padding: '.3rem .5rem' }}>Δ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => {
              const fmt = v => v == null ? '—' : r.int ? v.toLocaleString() : (+v).toFixed(3)
              const win8 = r.better === 'high' ? r.b > r.a : r.b < r.a
              return (
                <tr key={r.k} style={{ borderTop: '1px solid #27272a', color: '#f5f5f7' }}>
                  <td style={{ textAlign: 'left', padding: '.35rem .5rem', color: '#a1a1aa' }}>{r.k}</td>
                  <td style={{ textAlign: 'right', padding: '.35rem .5rem' }}>{fmt(r.a)}</td>
                  <td style={{ textAlign: 'right', padding: '.35rem .5rem',
                    color: win8 ? '#21c77a' : '#f5f5f7', fontWeight: win8 ? 800 : 400 }}>{fmt(r.b)}</td>
                  <td style={{ textAlign: 'right', padding: '.35rem .5rem',
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
    </div>
  )
}

function ResultView({ result, runId }) {
  if (!result) return null
  switch (result.type) {
    case 'downscaling': return <DownscalingViz result={result} runId={runId} />
    case 'atomistic_energy': return <EnergyViz result={result} />
    case 'training_convergence': return <TrainingConvergenceViz result={result} />
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

export function StepAnalyze() {
  const { runId, result, model, setStep } = useStore()
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!runId) { setLoading(false); return }
    api.jobFiles(runId).then(setFiles).finally(() => setLoading(false))
  }, [runId])

  return (
    <div>
      <StepHeader
        title="Results"
        sub={`${model?.name || model?.slug} — output analysis`}
      />
      <button className="btn btn-ghost" style={{ marginBottom: '1.2rem', fontSize: '.8rem' }}
        onClick={() => setStep(3)}>← Back</button>

      {loading && <Spinner size={28} />}

      {result && (
        <div style={{ marginBottom: '1.5rem' }}>
          <div className="section-label" style={{ marginBottom: '.75rem' }}>Output Summary</div>
          <ResultView result={result} runId={runId} />
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
          No results yet. Go back and run a job first.
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
