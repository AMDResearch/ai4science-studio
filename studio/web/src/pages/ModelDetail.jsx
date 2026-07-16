import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../api.js'

export default function ModelDetail() {
  const { slug } = useParams()
  const [m, setM] = useState(null)
  const [err, setErr] = useState('')
  const nav = useNavigate()

  useEffect(() => {
    setM(null)
    api.getModel(slug).then(setM).catch((e) => setErr(String(e)))
  }, [slug])

  if (err) return <div className="alert err">{err}</div>
  if (!m) return <div className="spinner">Loading…</div>

  return (
    <div>
      <h1>{m.name}</h1>
      <div className="caption">{m.domain} | slug: <code>{m.slug}</code></div>

      {m.schema_notes && m.schema_notes.length > 0 && (
        <details className="alert warn" style={{ marginTop: 8 }}>
          <summary>model.yaml schema notes ({m.schema_notes.length})</summary>
          {m.schema_notes.map((e, i) => <div key={i} className="caption">{e}</div>)}
        </details>
      )}

      {m.disclaimer && <div className="alert warn">{m.disclaimer}</div>}

      <div className="two-col" style={{ marginTop: 12 }}>
        <div className="card">
          <div><b>Task:</b> {m.task || '—'}</div>
          <div><b>License:</b> {m.license || '—'}</div>
          {m.hf_id && m.hf_id !== 'N/A'
            ? <div><b>Hugging Face:</b> <a href={`https://huggingface.co/${m.hf_id}`} target="_blank" rel="noreferrer"><code>{m.hf_id}</code></a></div>
            : <div><b>Weights:</b> {m.weight_source || 'see model card'}</div>}
        </div>
        <div className="card">
          {m.upstream_code && <div><b>Code:</b> <a href={m.upstream_code} target="_blank" rel="noreferrer">{m.upstream_code}</a></div>}
          {m.paper && <div><b>Paper:</b> <a href={m.paper} target="_blank" rel="noreferrer">{m.paper}</a></div>}
          {m.validated_hardware && m.validated_hardware.length > 0 &&
            <div><b>Validated hardware:</b> {m.validated_hardware.join(', ')}</div>}
          {m.vram_gb && <div><b>VRAM:</b> {m.vram_gb} GB</div>}
        </div>
      </div>

      {m.container_image && m.container_image.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <b>Container image(s):</b>
          {m.container_image.map((img, i) => <pre key={i} className="code">{img}</pre>)}
        </div>
      )}

      {m.data_source && <p><b>Data source:</b> {m.data_source}</p>}

      {m.model_variants && m.model_variants.length > 0 && (
        <div>
          <b>Variants:</b>
          <ul>{m.model_variants.map((v, i) =>
            <li key={i}><code>{v.name || '?'}</code> — {v.conditioning || ''}</li>)}</ul>
        </div>
      )}

      <h2>Recipes</h2>
      {(!m.recipes || m.recipes.length === 0) && <div className="alert info">No recipes defined for this model.</div>}
      {(m.recipes || []).map((r) => (
        <div key={r.task} className="card" style={{ marginBottom: 10 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div>
              <b>{r.task}</b> — {r.description || ''}
              {r.recipe_path && <div className="caption">{r.recipe_path}</div>}
            </div>
            <div>
              {r.runnable
                ? <button className="primary small"
                    onClick={() => nav(`/run?model=${encodeURIComponent(m.slug)}&task=${encodeURIComponent(r.task)}`)}>
                    Run
                  </button>
                : <button className="small" disabled title="No single sbatch script; run via the CLI/agent.">
                    Agent-only
                  </button>}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
