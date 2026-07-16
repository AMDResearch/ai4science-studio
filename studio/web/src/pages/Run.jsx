import { useEffect, useMemo, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { api } from '../api.js'

const CUSTOM = '__custom__'

export default function Run() {
  const [params] = useSearchParams()
  const nav = useNavigate()
  const [models, setModels] = useState([])
  const [slug, setSlug] = useState(params.get('model') || '')
  const [task, setTask] = useState(params.get('task') || '')
  const [form, setForm] = useState(null)
  const [err, setErr] = useState('')

  // form state
  const [values, setValues] = useState({})       // field name -> value
  const [customFlags, setCustomFlags] = useState({}) // field name -> using custom
  const [scaling, setScaling] = useState({ nodes: 1, ntasks_per_node: 8, gpus: 8, time: '00:30:00' })
  const [slurm, setSlurm] = useState({ partition: '', account: '', qos: '' })
  const [ack, setAck] = useState(false)
  const [orbitSynthetic, setOrbitSynthetic] = useState(true)

  const [preview, setPreview] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitMsg, setSubmitMsg] = useState(null)

  // Load model list once; default slug/task if not provided.
  useEffect(() => {
    api.listModels().then((d) => {
      const runnable = d.models.filter((m) => (m.runnable_tasks || []).length > 0)
      setModels(runnable)
      if (!slug && runnable.length) setSlug(runnable[0].slug)
    }).catch((e) => setErr(String(e)))
  }, [])

  const currentModel = useMemo(() => models.find((m) => m.slug === slug), [models, slug])

  // When slug changes, pick a valid task.
  useEffect(() => {
    if (currentModel) {
      const tasks = currentModel.runnable_tasks || []
      if (!tasks.includes(task)) setTask(tasks[0] || '')
    }
  }, [slug, currentModel])

  // Load the form descriptor whenever slug+task are set.
  useEffect(() => {
    if (!slug || !task) { setForm(null); return }
    setForm(null); setPreview(null); setSubmitMsg(null); setErr('')
    api.runForm(slug, task).then((fd) => {
      setForm(fd)
      const v = {}, cf = {}
      for (const field of fd.fields) {
        v[field.name] = field.default || ''
        cf[field.name] = false
      }
      setValues(v); setCustomFlags(cf)
      setScaling({ ...fd.scaling })
      setSlurm({ ...fd.slurm_defaults, qos: fd.slurm_defaults.qos || '' })
      setAck(!fd.disclaimer)
      setOrbitSynthetic(true)
    }).catch((e) => setErr(String(e)))
  }, [slug, task])

  // Which fields are hidden (ORBIT-2 data-mode branch).
  const hidden = useMemo(() => {
    const h = new Set()
    if (form && form.orbit2) {
      const o = form.orbit2
      for (const n of (orbitSynthetic ? o.hidden_when_synthetic : o.hidden_when_real)) h.add(n)
    }
    return h
  }, [form, orbitSynthetic])

  // Build the env dict (mirrors the Streamlit "only emit changed/required, never empty" rule).
  function buildEnv() {
    const env = {}
    if (!form) return env
    // ORBIT-2 synthetic override + shared dir seed.
    if (form.orbit2 && orbitSynthetic) Object.assign(env, form.orbit2.synthetic_env)
    if (form.shared_dir_seed) env.AI4S_SHARED_DIR = form.shared_dir_seed
    for (const field of form.fields) {
      if (hidden.has(field.name)) continue
      let v = values[field.name]
      if (field.widget === 'checkbox') v = values[field.name] ? '1' : '0'
      if (v === undefined || v === null) v = ''
      const def = field.widget === 'checkbox' ? field.default : field.default
      if (field.required && v) env[field.name] = v
      else if (v && String(v) !== String(def)) env[field.name] = v
    }
    return env
  }

  function currentRunReq() {
    return {
      slug, task,
      partition: slurm.partition, account: slurm.account, qos: slurm.qos,
      nodes: Number(scaling.nodes), ntasks_per_node: Number(scaling.ntasks_per_node),
      gpus: Number(scaling.gpus), time: scaling.time,
      env: buildEnv(), disclaimer_ack: ack,
    }
  }

  async function doPreview() {
    setErr(''); setSubmitMsg(null)
    try { setPreview(await api.preview(currentRunReq())) }
    catch (e) { setErr(String(e)) }
  }

  async function doSubmit() {
    setErr(''); setSubmitting(true); setSubmitMsg(null)
    try {
      const res = await api.submit(currentRunReq())
      setSubmitMsg({ ok: true, jobid: res.jobid })
    } catch (e) {
      setSubmitMsg({ ok: false, msg: String(e), data: e.data })
    } finally { setSubmitting(false) }
  }

  // Required fields still empty (block submit).
  const missingRequired = useMemo(() => {
    if (!form) return []
    return form.fields.filter((f) =>
      f.required && !hidden.has(f.name) && !values[f.name]).map((f) => f.name)
  }, [form, values, hidden])

  const blockers = useMemo(() => {
    const b = []
    if (!ack) b.push('acknowledge the disclaimer')
    if (missingRequired.length) b.push('fill required fields: ' + missingRequired.join(', '))
    if (form && !form.can_submit) b.push('sbatch not available on this host (dry-run only)')
    if (!slurm.partition || !slurm.account) b.push('set SLURM partition and account')
    if (form && !form.runtime_ok) b.push(`cluster runtime '${form.runtime}' is not Apptainer`)
    return b
  }, [ack, missingRequired, form, slurm])

  function setVal(name, v) { setValues((s) => ({ ...s, [name]: v })) }

  return (
    <div>
      <h1>Run a model</h1>
      {err && <div className="alert err">{err}</div>}

      <div className="row">
        <div className="col" style={{ flex: 1 }}>
          <label className="caption">Model</label>
          <select value={slug} onChange={(e) => setSlug(e.target.value)}>
            {models.map((m) => <option key={m.slug} value={m.slug}>{m.name}</option>)}
          </select>
        </div>
        <div className="col" style={{ flex: 1 }}>
          <label className="caption">Recipe</label>
          <select value={task} onChange={(e) => setTask(e.target.value)}>
            {(currentModel ? currentModel.runnable_tasks : []).map((t) =>
              <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </div>

      {!form && slug && task && <p className="spinner">Loading form…</p>}

      {form && !form.runtime_ok && (
        <div className="alert err">
          Cluster runtime is <code>{form.runtime}</code>. Studio v1 runs Apptainer only.
          Update it on the Cluster setup page.
        </div>
      )}

      {form && (
        <>
          {form.disclaimer && (
            <div className="alert warn">
              {form.disclaimer}
              <label className="inline" style={{ display: 'flex', marginTop: 8 }}>
                <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} />
                I acknowledge the above and accept responsibility for this run.
              </label>
            </div>
          )}

          {form.orbit2 && (
            <div className="field">
              <label>Data mode</label>
              <div className="row">
                <label className="inline"><input type="radio" checked={orbitSynthetic}
                  onChange={() => setOrbitSynthetic(true)} /> Synthetic (smoke test)</label>
                <label className="inline"><input type="radio" checked={!orbitSynthetic}
                  onChange={() => setOrbitSynthetic(false)} /> Real data</label>
              </div>
              {orbitSynthetic &&
                <div className="caption">Synthetic mode generates a tiny dataset and auto-downloads a checkpoint from HF.</div>}
            </div>
          )}

          <h2>Scaling</h2>
          <div className="row">
            {['nodes', 'ntasks_per_node', 'gpus'].map((k) => (
              <div className="col" key={k}>
                <label className="caption">{k === 'ntasks_per_node' ? 'Tasks/node' : k === 'gpus' ? 'GPUs/node' : 'Nodes'}</label>
                <input type="number" min="1" value={scaling[k]}
                  onChange={(e) => setScaling({ ...scaling, [k]: e.target.value })} />
              </div>
            ))}
            <div className="col"><label className="caption">Walltime</label>
              <input type="text" value={scaling.time}
                onChange={(e) => setScaling({ ...scaling, time: e.target.value })} /></div>
          </div>

          <h2>SLURM</h2>
          <div className="row">
            {['partition', 'account', 'qos'].map((k) => (
              <div className="col" key={k} style={{ flex: 1 }}>
                <label className="caption">{k[0].toUpperCase() + k.slice(1)}</label>
                <input type="text" value={slurm[k]} onChange={(e) => setSlurm({ ...slurm, [k]: e.target.value })} />
              </div>
            ))}
          </div>

          <h2>Parameters</h2>
          {form.shared_dir_seed &&
            <div className="caption">AI4S_SHARED_DIR = <code>{form.shared_dir_seed}</code> (from cluster config)</div>}

          {form.fields.filter((f) => !hidden.has(f.name)).map((field) => (
            <Field key={field.name} field={field}
              value={values[field.name]} onChange={(v) => setVal(field.name, v)}
              useCustom={customFlags[field.name]}
              setUseCustom={(b) => setCustomFlags((s) => ({ ...s, [field.name]: b }))} />
          ))}

          <div className="row" style={{ marginTop: 16 }}>
            <button onClick={doPreview}>Dry-run preview</button>
            <button className="primary" onClick={doSubmit} disabled={blockers.length > 0 || submitting}>
              {submitting ? 'Submitting…' : 'Submit job'}
            </button>
          </div>

          {blockers.length > 0 &&
            <div className="alert info" style={{ marginTop: 8 }}>To submit: {blockers.join('; ')}.</div>}

          {preview && (
            <div style={{ marginTop: 12 }}>
              <h3>Preview (dry-run)</h3>
              <pre className="code">{preview.preview}</pre>
              <div className="caption">Output: {preview.resolved_outfile || '(none)'}</div>
              {preview.blockers && preview.blockers.length > 0 &&
                <div className="alert info">Blockers: {preview.blockers.join('; ')}</div>}
            </div>
          )}

          {submitMsg && (submitMsg.ok
            ? <div className="alert ok" style={{ marginTop: 12 }}>
                Submitted batch job {submitMsg.jobid}. <a onClick={() => nav('/jobs')} style={{ cursor: 'pointer' }}>View in Jobs →</a>
              </div>
            : <div className="alert err" style={{ marginTop: 12 }}>Submission failed: {submitMsg.msg}</div>)}
        </>
      )}
    </div>
  )
}

function Field({ field, value, onChange, useCustom, setUseCustom }) {
  const label = field.name + (field.required ? ' *' : '')

  if (field.widget === 'checkbox') {
    return (
      <div className="field">
        <label className="inline">
          <input type="checkbox" checked={value === true || value === '1'}
            onChange={(e) => onChange(e.target.checked)} /> {label}
        </label>
        {field.description && <div className="desc">{field.description}</div>}
      </div>
    )
  }

  if (field.widget === 'select' && !field.is_path) {
    return (
      <div className="field">
        <label>{label}</label>
        {field.description && <div className="desc">{field.description}</div>}
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {field.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </div>
    )
  }

  if (field.is_path) {
    const opts = field.suggestions || []
    return (
      <div className="field">
        <label>{label}</label>
        {field.description && <div className="desc">{field.description}</div>}
        <select value={useCustom ? CUSTOM : value}
          onChange={(e) => {
            if (e.target.value === CUSTOM) { setUseCustom(true); onChange('') }
            else { setUseCustom(false); onChange(e.target.value) }
          }}>
          {opts.map((o) => (
            <option key={o.value} value={o.value}>
              {o.exists === null ? o.value : `${o.exists ? '✓ ' : '⚠ '}${o.value}`}
            </option>
          ))}
          <option value={CUSTOM}>✎ Enter a custom path…</option>
        </select>
        {useCustom &&
          <input type="text" placeholder="custom path" value={value}
            style={{ marginTop: 6 }} onChange={(e) => onChange(e.target.value)} />}
      </div>
    )
  }

  // number / text
  return (
    <div className="field">
      <label>{label}</label>
      {field.description && <div className="desc">{field.description}</div>}
      <input type="text" value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  )
}
