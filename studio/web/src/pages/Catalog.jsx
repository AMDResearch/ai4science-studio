import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api.js'

export default function Catalog() {
  const [models, setModels] = useState([])
  const [filters, setFilters] = useState({ domains: [], tasks: [], licenses: [] })
  const [err, setErr] = useState('')
  const [query, setQuery] = useState('')
  const [domain, setDomain] = useState('')
  const [task, setTask] = useState('')
  const [license, setLicense] = useState('')
  const nav = useNavigate()

  useEffect(() => {
    api.listModels().then((d) => setModels(d.models)).catch((e) => setErr(String(e)))
    api.filters().then(setFilters).catch(() => {})
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return models.filter((m) => {
      if (domain && m.domain !== domain) return false
      if (task && !(m.tasks_available || []).includes(task)) return false
      if (license && m.license !== license) return false
      if (q) {
        const hay = [m.name, m.slug, m.task, m.hf_id, m.domain,
          (m.tasks_available || []).join(' ')].join(' ').toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [models, query, domain, task, license])

  return (
    <div>
      <h1>Model catalog</h1>
      {err && <div className="alert err">{err}</div>}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row">
          <div className="col" style={{ flex: 2, minWidth: 200 }}>
            <label className="caption">Search</label>
            <input type="text" placeholder="name, task, hf id…"
              value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div className="col" style={{ flex: 1, minWidth: 140 }}>
            <label className="caption">Domain</label>
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              <option value="">All</option>
              {filters.domains.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div className="col" style={{ flex: 1, minWidth: 140 }}>
            <label className="caption">Task</label>
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="">All</option>
              {filters.tasks.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="col" style={{ flex: 1, minWidth: 140 }}>
            <label className="caption">License</label>
            <select value={license} onChange={(e) => setLicense(e.target.value)}>
              <option value="">All</option>
              {filters.licenses.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="caption" style={{ marginBottom: 8 }}>
        {filtered.length} of {models.length} models
      </div>

      <div className="grid">
        {filtered.map((m) => (
          <div key={m.slug} className="card">
            <h3>{m.name}</h3>
            <div className="caption">{m.domain} | {m.license || 'license: n/a'}</div>
            <p style={{ margin: '8px 0' }}>{m.task || <span className="muted">no description</span>}</p>
            <div>{(m.tasks_available || []).map((t) => <span key={t} className="badge">{t}</span>)}</div>
            {m.has_disclaimer && <div className="alert warn" style={{ marginTop: 8 }}>Usage disclaimer applies</div>}
            <div style={{ marginTop: 10 }}>
              <button className="small" onClick={() => nav(`/models/${encodeURIComponent(m.slug)}`)}>
                View details
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
