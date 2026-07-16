import { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'

// Mirror scaffold.slug_from_hf: org/model -> org__model (public names pass through).
function slugFromHf(hf) {
  if (hf && hf.includes('/') && hf !== 'N/A') return hf.replaceAll('/', '__')
  return hf || ''
}

export default function AddModel() {
  const [domains, setDomains] = useState([])
  const [hf, setHf] = useState('')
  const [slug, setSlug] = useState('')
  const [slugEdited, setSlugEdited] = useState(false)
  const [domain, setDomain] = useState('')
  const [license, setLicense] = useState('')
  const [task, setTask] = useState('')
  const [upstream, setUpstream] = useState('')
  const [paper, setPaper] = useState('')
  const [container, setContainer] = useState('')
  const [result, setResult] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    api.scaffoldDomains().then((d) => { setDomains(d.domains); setDomain(d.domains[0] || '') })
      .catch((e) => setErr(String(e)))
  }, [])

  // Auto-derive slug from HF id until the user edits it manually.
  const derivedSlug = useMemo(() => slugFromHf(hf), [hf])
  useEffect(() => { if (!slugEdited) setSlug(derivedSlug) }, [derivedSlug, slugEdited])

  async function create() {
    setErr(''); setResult(null)
    try {
      const res = await api.createModel({
        slug, domain, hf_id: hf, license, task,
        upstream_code: upstream, paper, container_image: container,
      })
      setResult(res)
    } catch (e) {
      setErr(e.detail && e.detail.errors ? e.detail.errors.join('; ') : String(e))
    }
  }

  const canCreate = slug && domain && task && license

  return (
    <div>
      <h1>Add a model</h1>
      <p className="muted">
        Scaffold a new model into the repo from <code>_template/</code>. This writes files
        locally; review with <code>git diff</code> and open a PR yourself.
      </p>
      {err && <div className="alert err">{err}</div>}

      <div className="two-col">
        <div>
          <div className="field"><label>Hugging Face id</label>
            <input type="text" placeholder="org/model  (or N/A)" value={hf} onChange={(e) => setHf(e.target.value)} /></div>
          <div className="field"><label>Slug (folder name)</label>
            <input type="text" value={slug}
              onChange={(e) => { setSlug(e.target.value); setSlugEdited(true) }} />
            <div className="desc">org/model → org__model, or a public name.</div></div>
          <div className="field"><label>Domain</label>
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              {domains.map((d) => <option key={d} value={d}>{d}</option>)}
            </select></div>
          <div className="field"><label>License</label>
            <input type="text" placeholder="Apache-2.0 / MIT / link" value={license} onChange={(e) => setLicense(e.target.value)} /></div>
        </div>
        <div>
          <div className="field"><label>Task</label>
            <input type="text" placeholder="one-line description" value={task} onChange={(e) => setTask(e.target.value)} /></div>
          <div className="field"><label>Upstream code URL</label>
            <input type="text" placeholder="https://github.com/..." value={upstream} onChange={(e) => setUpstream(e.target.value)} /></div>
          <div className="field"><label>Paper URL</label>
            <input type="text" placeholder="https://arxiv.org/abs/..." value={paper} onChange={(e) => setPaper(e.target.value)} /></div>
          <div className="field"><label>Container image (optional)</label>
            <input type="text" placeholder="rocm/pytorch:..." value={container} onChange={(e) => setContainer(e.target.value)} /></div>
        </div>
      </div>

      <button className="primary" onClick={create} disabled={!canCreate}>Create model</button>
      {!canCreate && <div className="alert info" style={{ marginTop: 8 }}>Fill slug, domain, task and license to enable creation.</div>}

      {result && (
        <div className="alert ok" style={{ marginTop: 12 }}>
          Created <code>{result.path}</code> and registered in models.yaml.
          <ol style={{ marginBottom: 0 }}>
            <li>Fill in <code>model.yaml</code> (recipes, env_vars, validated_hardware).</li>
            <li>Add recipe docs under <code>recipes/</code> and scripts under <code>examples/</code>.</li>
            <li>Add an <code>ACKNOWLEDGEMENTS.md</code> entry.</li>
            <li>Review with <code>git diff</code> and open a PR.</li>
          </ol>
        </div>
      )}
    </div>
  )
}
