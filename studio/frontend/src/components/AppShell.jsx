import { useStore } from '../store'
import { RecordButton } from './RecordButton'

const STEPS = ['Domain', 'Model', 'Configure', 'Run', 'Analyze']

export function AppShell({ children }) {
  const { step, setStep, mode, setMode, addModelOpen, setAddModelOpen,
          domain, model, view, setView, demoLock } = useStore()

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(10,2,3,0.85)', backdropFilter: 'blur(12px)',
        borderBottom: '1px solid #27272a',
        // Extra right padding keeps the Demo/Live, +Model and Record controls clear
        // of the fixed Zscaler "Browser Isolation" overlay pinned to the top-right.
        padding: '.75rem 1.5rem',
        paddingRight: 'clamp(1.5rem, 24vw, 360px)',
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

        {/* Demo / Live toggle. Demo-locked (read-only) deployments gray out Live. */}
        <div className="mode-toggle">
          <button className={`mode-btn${mode === 'demo' ? ' active-demo' : ''}`}
            onClick={() => setMode('demo')}>Demo</button>
          <button className={`mode-btn${mode === 'live' ? ' active-live' : ''}`}
            onClick={() => !demoLock && setMode('live')}
            disabled={demoLock}
            title={demoLock ? 'Live mode is disabled on the read-only demo site' : 'Run live on the cluster'}
            style={demoLock ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}>
            Live{demoLock ? ' 🔒' : ''}
          </button>
        </div>

        {/* Add model — button stays available; inputs inside are disabled on the
            read-only demo site (see AddModelModal). */}
        <button className="btn btn-ghost" style={{ flexShrink: 0, fontSize: '.75rem', padding: '.3rem .8rem' }}
          onClick={() => setAddModelOpen(true)}>+ Model</button>

        <RecordButton />
      </header>

      {/* Read-only demo disclaimer banner (demo site only) */}
      {demoLock && (
        <div style={{
          background: 'rgba(237,28,36,.1)', borderBottom: '1px solid rgba(237,28,36,.35)',
          color: '#ffb3b7', fontSize: '.82rem', textAlign: 'center',
          padding: '.5rem 1rem', lineHeight: 1.4,
        }}>
          🔒 <b style={{ color: '#fff' }}>Demo preview</b> — this read-only site demonstrates the
          AI4Science Studio <b>workflow</b> with pre-computed results. Live cluster runs are disabled,
          and the showcased applications are <b>not performance-optimized</b>.
        </div>
      )}

      {/* Main: persistent LUX left panel + wizard content */}
      <div style={{ flex: 1, display: 'flex',
        width: '100%', boxSizing: 'border-box', gap: '1.5rem', padding: '2rem 1.5rem' }}>
        {/* Left panel — LUX supercomputer, centered in the whole left region,
            with ~10vh of space at the top. */}
        <aside style={{ flex: '0 0 34%', display: 'flex', flexDirection: 'column',
          alignItems: 'center', position: 'sticky', top: '4.5rem',
          alignSelf: 'flex-start', paddingTop: '10vh' }}>
          {/* Inner block constrained to the image width so all text wraps within it. */}
          <div style={{ width: '100%', maxWidth: 320, display: 'flex', flexDirection: 'column',
            gap: '1.1rem', textAlign: 'center' }}>
            {/* Big, bold LUX title above the image */}
            <div style={{ fontSize: '1.35rem', fontWeight: 900, lineHeight: 1.25,
              color: '#f5f5f7', letterSpacing: '-.01em' }}>
              LUX — First AMD and US Sovereign AI Infrastructure
            </div>
            <img src="/lux.jpg" alt="LUX — First AMD and US Sovereign AI Infrastructure"
              style={{ width: '100%', borderRadius: '.75rem', border: '1px solid #27272a',
                objectFit: 'cover', boxShadow: '0 8px 28px rgba(0,0,0,.5)' }}
              onError={e => { e.target.style.display = 'none' }} />
            <div style={{ fontSize: '.82rem', fontWeight: 700, color: '#a1a1aa', letterSpacing: '.02em' }}>
              US DOE Genesis Mission · ORNL · AMD · HPE
            </div>
            <div style={{ fontSize: '.9rem', fontWeight: 700, lineHeight: 1.45,
              color: '#c8a24a', letterSpacing: '.01em' }}>
              Reference Design for an end-to-end platform and service for building
              open models and Agentic workflows
            </div>
          </div>
        </aside>

        {/* Wizard content */}
        <main style={{ flex: 1, minWidth: 0, maxWidth: 1100 }}>
          {children}
        </main>
      </div>

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
