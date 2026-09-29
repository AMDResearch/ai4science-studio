import { useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { Spinner } from './ui'

const DOMAINS = [
  { value: 'earth_science', label: 'Earth Science' },
  { value: 'material_science', label: 'Material Science' },
  { value: 'healthcare', label: 'Healthcare & Life Sciences' },
  { value: 'physics_simulation', label: 'Physics Simulation' },
  { value: 'protein_folding', label: 'Protein Folding' },
]

export function AddModelModal() {
  const { setAddModelOpen, demoLock } = useStore()
  const [form, setForm] = useState({
    slug: '', name: '', domain: 'earth_science', task: '',
    hf_id: '', license: 'Apache-2.0', container_image: '',
    vram_gb: '', validated_hardware: 'MI300X',
    tasks_available: 'inference',
    prompts: ['', '', ''],
  })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)

  function set(k, v) { setForm(f => ({ ...f, [k]: v })) }

  async function handleSubmit(e) {
    e.preventDefault()
    if (demoLock) { setError('Registering models is disabled on the read-only demo site.'); return }
    setSaving(true); setError('')
    try {
      const curated_prompts = form.prompts
        .filter(p => p.trim())
        .map((text, i) => ({ label: `Prompt ${i + 1}`, text, task: 'inference' }))
      await api.addModel({
        slug: form.slug.replace(/\s+/g, '-'),
        name: form.name,
        domain: form.domain,
        task: form.task,
        hf_id: form.hf_id,
        license: form.license,
        container_image: form.container_image,
        vram_gb: form.vram_gb ? parseFloat(form.vram_gb) : null,
        validated_hardware: form.validated_hardware.split(',').map(s => s.trim()).filter(Boolean),
        tasks_available: form.tasks_available.split(',').map(s => s.trim()).filter(Boolean),
        curated_prompts,
      })
      setDone(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(8px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem',
    }} onClick={() => setAddModelOpen(false)}>
      <div className="card" style={{ maxWidth: 560, width: '100%', padding: '1.5rem', boxSizing: 'border-box' }}
        onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
          <div className="gradient-text" style={{ fontWeight: 800, fontSize: '1.1rem' }}>Register New Model</div>
          <button className="btn btn-ghost" style={{ padding: '.2rem .6rem' }} onClick={() => setAddModelOpen(false)}>✕</button>
        </div>

        {done ? (
          <div style={{ textAlign: 'center', padding: '2rem 0' }}>
            <div style={{ fontSize: '2rem', marginBottom: '.5rem' }}>✅</div>
            <div style={{ fontWeight: 700, color: '#21c77a' }}>Model registered!</div>
            <div style={{ color: '#a1a1aa', fontSize: '.85rem', margin: '.5rem 0 1rem' }}>
              "{form.name}" added to {form.domain.replace('_', ' ')}.
            </div>
            <button className="btn btn-primary" onClick={() => { setAddModelOpen(false); window.location.reload() }}>
              Reload to see it
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
            {demoLock && (
              <div style={{ padding: '.5rem .75rem', borderRadius: '.5rem', fontSize: '.8rem',
                background: 'rgba(237,28,36,.08)', border: '1px solid rgba(237,28,36,.3)', color: '#ff8f93' }}>
                🔒 Read-only demo — model registration is disabled. Fields are shown for reference only.
              </div>
            )}
            <fieldset disabled={demoLock} style={{ border: 'none', padding: 0, margin: 0,
              display: 'flex', flexDirection: 'column', gap: '.75rem',
              opacity: demoLock ? 0.55 : 1 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.75rem' }}>
              <div>
                <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>Slug (ID)</label>
                <input className="field" required placeholder="e.g. MyModel" value={form.slug}
                  onChange={e => set('slug', e.target.value)} />
              </div>
              <div>
                <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>Display Name</label>
                <input className="field" required placeholder="e.g. My Model" value={form.name}
                  onChange={e => set('name', e.target.value)} />
              </div>
            </div>
            <div>
              <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>Domain</label>
              <select className="field" value={form.domain} onChange={e => set('domain', e.target.value)}>
                {DOMAINS.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
            <div>
              <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>Task Description</label>
              <input className="field" required placeholder="e.g. Molecular generation from SMILES scaffolds" value={form.task}
                onChange={e => set('task', e.target.value)} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.75rem' }}>
              <div>
                <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>HuggingFace ID</label>
                <input className="field" placeholder="org/model-name" value={form.hf_id}
                  onChange={e => set('hf_id', e.target.value)} />
              </div>
              <div>
                <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>License</label>
                <input className="field" placeholder="Apache-2.0" value={form.license}
                  onChange={e => set('license', e.target.value)} />
              </div>
            </div>
            <div>
              <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>Container Image</label>
              <input className="field" placeholder="rocm/pytorch:rocm7.2.2_..." value={form.container_image}
                onChange={e => set('container_image', e.target.value)} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.75rem' }}>
              <div>
                <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>VRAM (GB)</label>
                <input className="field" type="number" placeholder="e.g. 24" value={form.vram_gb}
                  onChange={e => set('vram_gb', e.target.value)} />
              </div>
              <div>
                <label className="section-label" style={{ display: 'block', marginBottom: '.3rem' }}>Validated Hardware</label>
                <input className="field" placeholder="MI300X, MI210" value={form.validated_hardware}
                  onChange={e => set('validated_hardware', e.target.value)} />
              </div>
            </div>
            <div>
              <label className="section-label" style={{ display: 'block', marginBottom: '.4rem' }}>Curated Prompts (up to 3)</label>
              {[0, 1, 2].map(i => (
                <textarea key={i} className="field" rows={2}
                  style={{ marginBottom: i < 2 ? '.4rem' : 0, resize: 'vertical', fontSize: '.8rem' }}
                  placeholder={`Prompt ${i + 1} — describe a concrete task for this model`}
                  value={form.prompts[i]}
                  onChange={e => {
                    const p = [...form.prompts]; p[i] = e.target.value
                    set('prompts', p)
                  }} />
              ))}
            </div>
            </fieldset>
            {error && <div style={{ color: '#ff8f93', fontSize: '.8rem' }}>{error}</div>}
            <div style={{ display: 'flex', gap: '.75rem', justifyContent: 'flex-end' }}>
              <button type="button" className="btn btn-ghost" onClick={() => setAddModelOpen(false)}>
                {demoLock ? 'Close' : 'Cancel'}
              </button>
              <button type="submit" className="btn btn-primary" disabled={saving || demoLock}
                style={demoLock ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}>
                {saving ? <Spinner size={14} /> : null} Register Model
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}
