import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { StepHeader, PromptCard, Spinner } from '../components/ui'

export function StepConfigure() {
  const { model, domain, prompt, setPrompt, customPrompt, setCustomPrompt,
          promptError, setPromptError, mode, setStep,
          task, setTask, modelVariant, setModelVariant } = useStore()
  const [curated, setCurated] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCustom, setShowCustom] = useState(false)
  const [validating, setValidating] = useState(false)

  useEffect(() => {
    if (!model) return
    setLoading(true)
    api.prompts(model.slug, domain)
      .then(setCurated)
      .finally(() => setLoading(false))
  }, [model, domain])

  async function handleProceed() {
    const active = showCustom ? customPrompt : prompt
    if (!active.trim()) { setPromptError('Please select or enter a prompt.'); return }
    setValidating(true)
    try {
      const v = await api.validatePrompt(model.slug, domain, active)
      if (!v.ok) { setPromptError(v.message); return }
      setPromptError('')
      if (showCustom) setPrompt(customPrompt)
      setStep(3)
    } catch (e) {
      setPromptError(e.message)
    } finally {
      setValidating(false)
    }
  }

  const activeText = showCustom ? customPrompt : prompt

  return (
    <div>
      <StepHeader
        title={`Configure: ${model?.name || model?.slug}`}
        sub={`${model?.task || ''} — choose a prompt and set parameters.`}
      />
      <button className="btn btn-ghost" style={{ marginBottom: '1.2rem', fontSize: '.8rem' }}
        onClick={() => setStep(1)}>← Back</button>

      {/* Mode banner */}
      <div style={{
        marginBottom: '1.2rem', padding: '.6rem 1rem',
        background: mode === 'demo' ? 'rgba(33,199,122,.08)' : 'rgba(237,28,36,.08)',
        border: `1px solid ${mode === 'demo' ? 'rgba(33,199,122,.3)' : 'rgba(237,28,36,.3)'}`,
        borderRadius: '.6rem', fontSize: '.82rem',
        color: mode === 'demo' ? '#6ee7b7' : '#ff8f93',
      }}>
        {mode === 'demo'
          ? '🟢 Demo mode — synthetic results, runs in <60 s on login node'
          : '🔴 Live mode — real SLURM job on Vultr Lux cluster (requires GPU allocation)'}
      </div>

      {/* HydraGNN: task + trained-model selectors */}
      {model?.slug === 'HydraGNN' && (
        <div style={{ marginBottom: '1.2rem' }}>
          <div className="section-label" style={{ marginBottom: '.5rem' }}>Task</div>
          <div style={{ display: 'flex', gap: '.5rem', marginBottom: task === 'inference' ? '.9rem' : 0 }}>
            {[
              { id: 'inference', label: 'Inference', sub: 'Predict energy on a real structure' },
              { id: 'train', label: 'Training scaling', sub: '1-GPU vs 8-GPU convergence' },
            ].map(t => (
              <div key={t.id}
                className={`card clickable${task === t.id ? ' selected' : ''}`}
                onClick={() => setTask(t.id)}
                style={{ flex: 1, padding: '.7rem .9rem', userSelect: 'none' }}>
                <div style={{ fontWeight: 700, fontSize: '.88rem', color: '#f5f5f7' }}>{t.label}</div>
                <div style={{ fontSize: '.74rem', color: '#a1a1aa', marginTop: '.15rem' }}>{t.sub}</div>
              </div>
            ))}
          </div>

          {/* Trained-model variant — only for live inference */}
          {task === 'inference' && mode === 'live' && (
            <>
              <div className="section-label" style={{ margin: '.3rem 0 .5rem' }}>Trained model</div>
              <div style={{ display: 'flex', gap: '.5rem' }}>
                {[
                  { id: '8gpu', label: '8-GPU model', sub: '268k structures · corr 0.79' },
                  { id: '1gpu', label: '1-GPU model', sub: '40k structures · corr 0.75' },
                ].map(v => (
                  <div key={v.id}
                    className={`card clickable${modelVariant === v.id ? ' selected' : ''}`}
                    onClick={() => setModelVariant(v.id)}
                    style={{ flex: 1, padding: '.7rem .9rem', userSelect: 'none' }}>
                    <div style={{ fontWeight: 700, fontSize: '.88rem', color: '#f5f5f7' }}>{v.label}</div>
                    <div style={{ fontSize: '.74rem', color: '#a1a1aa', marginTop: '.15rem' }}>{v.sub}</div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Curated prompts */}
      <div style={{ marginBottom: '1.2rem' }}>
        <div className="section-label" style={{ marginBottom: '.6rem' }}>
          {model?.slug === 'HydraGNN' && task === 'train' ? 'Training-scaling Demo' : 'Suggested Prompts'}
        </div>
        {loading && <Spinner size={20} />}
        {!loading && curated.length === 0 && (
          <div style={{ color: '#52525b', fontSize: '.85rem' }}>No curated prompts for this model. Use custom prompt below.</div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
          {curated
            .filter(p => model?.slug !== 'HydraGNN' ||
              (task === 'train' ? p.task === 'train' : p.task !== 'train'))
            .map((p, i) => (
            <PromptCard
              key={i}
              label={p.label}
              text={p.text}
              task={p.task}
              selected={!showCustom && prompt === p.text}
              onClick={() => { setPrompt(p.text); setShowCustom(false); setPromptError('') }}
            />
          ))}
        </div>
      </div>

      {/* Custom prompt accordion */}
      <div style={{ marginBottom: '1.2rem' }}>
        <button className="btn btn-ghost" style={{ fontSize: '.8rem', marginBottom: showCustom ? '.6rem' : 0 }}
          onClick={() => setShowCustom(v => !v)}>
          {showCustom ? '▲' : '▼'} Custom prompt
        </button>
        {showCustom && (
          <textarea
            className="field" rows={5}
            style={{ resize: 'vertical', marginTop: '.4rem', fontSize: '.85rem' }}
            placeholder="Describe exactly what you want to run — include key parameters like date range, region, composition, or molecular scaffold..."
            value={customPrompt}
            onChange={e => { setCustomPrompt(e.target.value); setPromptError('') }}
          />
        )}
      </div>

      {/* Active prompt preview */}
      {activeText && !showCustom && (
        <div style={{ marginBottom: '1.2rem', padding: '.75rem 1rem',
          background: 'rgba(56,189,248,.06)', border: '1px solid rgba(56,189,248,.2)',
          borderRadius: '.6rem', fontSize: '.82rem', color: '#94a3b8', lineHeight: 1.5 }}>
          <span style={{ color: '#7dd3fc', fontWeight: 700 }}>Active: </span>{activeText}
        </div>
      )}

      {promptError && (
        <div style={{ marginBottom: '1rem', color: '#ff8f93', fontSize: '.85rem',
          padding: '.5rem .85rem', background: 'rgba(237,28,36,.08)', borderRadius: '.5rem',
          border: '1px solid rgba(237,28,36,.3)' }}>
          {promptError}
        </div>
      )}

      <button className="btn btn-primary"
        disabled={!activeText.trim() || validating}
        onClick={handleProceed}>
        {validating ? <Spinner size={14} /> : null}
        {validating ? 'Validating...' : 'Continue to Run →'}
      </button>
    </div>
  )
}
