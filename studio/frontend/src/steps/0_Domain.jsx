import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { StepHeader, SelectCard, Spinner } from '../components/ui'

export function StepDomain() {
  const { domain, setDomain, setStep } = useStore()
  const [domains, setDomains] = useState([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')

  useEffect(() => {
    api.domains().then(setDomains).catch(e => setErr(e.message)).finally(() => setLoading(false))
  }, [])

  function select(d) {
    setDomain(d.slug)
    setStep(1)
  }

  return (
    <div>
      <StepHeader
        title="Choose a Science Domain"
        sub="Select the scientific field you want to explore with AMD-optimized AI models."
      />
      {loading && <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}><Spinner size={32} /></div>}
      {err && <div style={{ color: '#ff8f93', marginBottom: '1rem' }}>{err}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
        {domains.map(d => (
          <SelectCard
            key={d.slug}
            icon={d.icon}
            title={d.title}
            sub={`${d.model_count} model${d.model_count !== 1 ? 's' : ''}`}
            selected={domain === d.slug}
            onClick={() => select(d)}
          />
        ))}
      </div>
    </div>
  )
}
