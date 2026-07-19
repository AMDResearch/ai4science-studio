import { useStore } from '../store'
import { RecordButton } from './RecordButton'

const STEPS = ['Domain', 'Model', 'Configure', 'Run', 'Analyze']

export function AppShell({ children }) {
  const { step, setStep, mode, setMode, addModelOpen, setAddModelOpen,
          domain, model, view, setView } = useStore()

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(10,2,3,0.85)', backdropFilter: 'blur(12px)',
        borderBottom: '1px solid #27272a',
        padding: '.75rem 1.5rem',
        display: 'flex', alignItems: 'center', gap: '1.2rem',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', flexShrink: 0 }}>
          <img src="/AMD-Logo.png" alt="AMD"
            style={{ height: 28, objectFit: 'contain' }}
            onError={e => {
              e.target.style.display = 'none'
              e.target.nextSibling && (e.target.nextSibling.style.display = '')
            }} />
          <div style={{
            display: 'none', width: 32, height: 32, borderRadius: '.4rem',
            background: '#ED1C24', alignItems: 'center', justifyContent: 'center',
            fontWeight: 900, fontSize: '1rem', color: '#fff', letterSpacing: '-1px',
          }}>AI</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '.9rem', color: '#f5f5f7', lineHeight: 1 }}>AI4Science Studio</div>
            <div style={{ fontSize: '.6rem', color: '#6b7280', letterSpacing: '.08em', textTransform: 'uppercase' }}>AMD</div>
          </div>
        </div>

        {/* Step pills (hidden when in catalog view) */}
        <nav style={{ display: 'flex', gap: '.35rem', flex: 1, flexWrap: 'wrap', alignItems: 'center' }}>
          {view === 'catalog' ? (
            <button className="step-pill active" style={{ border: 'none', cursor: 'default' }}>
              Model Catalog
            </button>
          ) : (
            STEPS.map((label, i) => {
              const cls = i < step ? 'done' : i === step ? 'active' : 'future'
              return (
                <button key={i} className={`step-pill ${cls}`}
                  onClick={() => i < step && setStep(i)}
                  style={{ border: 'none', cursor: i < step ? 'pointer' : 'default' }}>
                  {i < step ? '✓ ' : `${i+1}. `}{label}
                </button>
              )
            })
          )}
          {/* Catalog toggle */}
          <button
            onClick={() => setView(view === 'catalog' ? 'wizard' : 'catalog')}
            style={{
              marginLeft: '.5rem', padding: '.25rem .65rem', borderRadius: '9999px',
              fontSize: '.72rem', fontWeight: 700, letterSpacing: '.04em',
              border: '1px solid',
              background: view === 'catalog' ? '#ED1C24' : 'transparent',
              borderColor: view === 'catalog' ? '#ED1C24' : '#52525b',
              color: view === 'catalog' ? '#fff' : '#a1a1aa',
              cursor: 'pointer', flexShrink: 0,
              transition: 'all .15s',
            }}>
            {view === 'catalog' ? '← Wizard' : '⊞ Catalog'}
          </button>
        </nav>

        {/* Demo / Live toggle */}
        <div className="mode-toggle">
          <button className={`mode-btn${mode === 'demo' ? ' active-demo' : ''}`}
            onClick={() => setMode('demo')}>Demo</button>
          <button className={`mode-btn${mode === 'live' ? ' active-live' : ''}`}
            onClick={() => setMode('live')}>Live</button>
        </div>

        {/* Add model */}
        <button className="btn btn-ghost" style={{ flexShrink: 0, fontSize: '.75rem', padding: '.3rem .8rem' }}
          onClick={() => setAddModelOpen(true)}>+ Model</button>

        <RecordButton />
      </header>

      {/* Main */}
      <main style={{ flex: 1, padding: '2rem 1.5rem', maxWidth: 1100, margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
        {children}
      </main>

      {/* Footer */}
      <footer style={{ padding: '.75rem 1.5rem', borderTop: '1px solid #1a1a1c',
        fontSize: '.7rem', color: '#52525b', display: 'flex', justifyContent: 'space-between' }}>
        <span>AMD AI4Science Studio — powered by <a href="https://github.com/AMDResearch/ai4science-studio"
          style={{ color: '#6b7280', textDecoration: 'none' }}>ai4science-studio</a></span>
        <span>{domain ? `Domain: ${domain}` : ''}{model ? ` · ${model.name || model.slug}` : ''}</span>
      </footer>
    </div>
  )
}
