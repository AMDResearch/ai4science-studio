import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { StepHeader, Spinner, Empty } from '../components/ui'

export function StepModel() {
  const { domain, model, setModel, setStep } = useStore()
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!domain) return
    setLoading(true)
    api.domainModels(domain).then(setModels).finally(() => setLoading(false))
  }, [domain])

  function select(m) {
    setModel(m)
    setStep(2)
  }

  const domainLabel = domain?.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()) || ''

  return (
    <div>
      <StepHeader
        title={`${domainLabel} Models`}
        sub="Select a model to run. All models are optimized for AMD MI-series GPUs."
      />
      <button className="btn btn-ghost" style={{ marginBottom: '1.2rem', fontSize: '.8rem' }}
        onClick={() => setStep(0)}>← Back</button>

      {loading && <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}><Spinner size={32} /></div>}
      {!loading && models.length === 0 && <Empty msg={`No models found in ${domainLabel}.`} />}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
        {models.map(m => (
          <div key={m.slug}
            className={`card clickable${model?.slug === m.slug ? ' selected' : ''}`}
            style={{ padding: '1rem 1.2rem', cursor: 'pointer' }}
            onClick={() => select(m)}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 800, fontSize: '1rem', color: '#f5f5f7', marginBottom: '.25rem' }}>
                  {m.name || m.slug}
                </div>
                <div style={{ fontSize: '.82rem', color: '#a1a1aa', lineHeight: 1.4, marginBottom: '.5rem' }}>
                  {m.task}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.35rem' }}>
                  {m.tasks_available?.map(t => (
                    <span key={t} className="badge badge-info">{t}</span>
                  ))}
                  {m.validated_hardware?.map(hw => (
                    <span key={hw} className="badge badge-amd">{hw}</span>
                  ))}
                  {m.license && (
                    <span className="badge badge-muted">{m.license}</span>
                  )}
                </div>
                {m.container_images?.length > 0 && (
                  <div style={{ marginTop: '.55rem', display: 'flex', flexDirection: 'column', gap: '.25rem' }}>
                    {m.container_images.map(img => (
                      <div key={img} style={{
                        fontSize: '.68rem', color: '#71717a', fontFamily: 'monospace',
                        display: 'flex', alignItems: 'center', gap: '.35rem',
                      }}>
                        <span style={{ color: '#f5a524' }}>▣</span>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div style={{ flexShrink: 0, textAlign: 'right' }}>
                {m.vram_gb && (
                  <div style={{ fontSize: '.75rem', color: '#a1a1aa' }}>
                    <span style={{ color: '#f5a524', fontWeight: 700 }}>{m.vram_gb}</span> GB VRAM
                  </div>
                )}
                {m.hf_id && m.hf_id !== 'N/A' && (
                  <div style={{ fontSize: '.65rem', color: '#52525b', marginTop: '.2rem' }}>
                    🤗 {m.hf_id}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
