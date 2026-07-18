export function Spinner({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      style={{ animation: 'spin 0.8s linear infinite' }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      <circle cx="12" cy="12" r="10" stroke="rgba(237,28,36,.2)" strokeWidth="3" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="#ED1C24" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

export function StepHeader({ title, sub }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div className="section-label" style={{ marginBottom: '.3rem' }}>AMD AI4Science Studio</div>
      <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800, color: '#f5f5f7' }}>{title}</h2>
      {sub && <p style={{ margin: '.4rem 0 0', color: '#a1a1aa', fontSize: '.9rem' }}>{sub}</p>}
    </div>
  )
}

export function SelectCard({ icon, title, sub, badge, selected, onClick, style }) {
  return (
    <div
      className={`card clickable${selected ? ' selected' : ''}`}
      onClick={onClick}
      style={{ padding: '1rem 1.2rem', userSelect: 'none', ...style }}
    >
      {icon && <div style={{ fontSize: '1.6rem', marginBottom: '.4rem' }}>{icon}</div>}
      <div style={{ fontWeight: 700, fontSize: '.95rem', color: '#f5f5f7', marginBottom: '.2rem' }}>
        {title}
      </div>
      {sub && <div style={{ fontSize: '.78rem', color: '#a1a1aa', lineHeight: 1.4 }}>{sub}</div>}
      {badge && <div style={{ marginTop: '.5rem' }}>
        <span className={`badge badge-${badge.type || 'muted'}`}>{badge.label}</span>
      </div>}
    </div>
  )
}

export function Empty({ msg = 'Nothing here yet.' }) {
  return (
    <div style={{ color: '#52525b', textAlign: 'center', padding: '2rem', fontSize: '.9rem' }}>
      {msg}
    </div>
  )
}

export function PromptCard({ label, text, task, selected, onClick }) {
  return (
    <div
      className={`prompt-card${selected ? ' selected' : ''}`}
      onClick={onClick}
    >
      <div style={{ display: 'flex', gap: '.5rem', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div style={{ fontWeight: 700, fontSize: '.85rem', color: '#f5f5f7', marginBottom: '.35rem' }}>
          {label}
        </div>
        {task && <span className="badge badge-info" style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>{task}</span>}
      </div>
      <div style={{ fontSize: '.78rem', color: '#94a3b8', lineHeight: 1.5 }}>{text}</div>
    </div>
  )
}

export function Stat({ label, value, color, sub }) {
  return (
    <div className="card" style={{ padding: '.75rem', textAlign: 'center' }}>
      <div className="section-label" style={{ marginBottom: '.3rem' }}>{label}</div>
      <div style={{ fontSize: '1.2rem', fontWeight: 800, color: color || '#f5f5f7' }}>{value}</div>
      {sub && <div style={{ fontSize: '.7rem', color: '#52525b', marginTop: '.15rem' }}>{sub}</div>}
    </div>
  )
}

export function StatGrid({ children, min = 150 }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fill,minmax(${min}px,1fr))`,
      gap: '.75rem', marginBottom: '1rem' }}>
      {children}
    </div>
  )
}

export function StatusBadge({ state }) {
  const map = {
    idle:      { type: 'muted',  label: 'Idle' },
    pending:   { type: 'warn',   label: 'Pending' },
    queued:    { type: 'warn',   label: 'Queued' },
    running:   { type: 'info',   label: 'Running' },
    completed: { type: 'ok',     label: 'Completed' },
    failed:    { type: 'amd',    label: 'Failed' },
  }
  const { type, label } = map[state] || { type: 'muted', label: state }
  return <span className={`badge badge-${type}`}>{label}</span>
}
