import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api.js'

const STATE_ICON = {
  PENDING: '🟡', RUNNING: '🟢', COMPLETING: '🟢', COMPLETED: '✅', GONE: '✅',
  FAILED: '🔴', OUT_OF_MEMORY: '🔴', NODE_FAIL: '🔴', CANCELLED: '⚪',
  TIMEOUT: '🔴', UNKNOWN: '❔',
}
const STATE_HELP = {
  COMPLETED: 'Finished successfully.',
  FAILED: 'The job exited with an error — check the log below.',
  OUT_OF_MEMORY: 'Killed: out of memory.',
  TIMEOUT: 'Killed: exceeded the walltime limit.',
  CANCELLED: 'Cancelled.', NODE_FAIL: 'A compute node failed.',
  RUNNING: 'Currently running.', PENDING: 'Queued, waiting for resources.',
}

export default function Jobs() {
  const [jobs, setJobs] = useState(null)
  const [err, setErr] = useState('')
  const [auto, setAuto] = useState(false)
  const [logs, setLogs] = useState({})   // jobid -> log text
  const nav = useNavigate()

  const load = useCallback(() => {
    api.listJobs().then((d) => setJobs(d.jobs)).catch((e) => setErr(String(e)))
  }, [])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!auto) return
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [auto, load])

  async function toggleLog(jid) {
    if (logs[jid] !== undefined) { setLogs((s) => ({ ...s, [jid]: undefined })); return }
    try {
      const d = await api.jobLog(jid)
      setLogs((s) => ({ ...s, [jid]: d.log || '(no log output yet)' }))
    } catch (e) { setLogs((s) => ({ ...s, [jid]: String(e) })) }
  }

  async function cancel(jid) { await api.cancelJob(jid); load() }
  async function remove(jid) { await api.removeJob(jid); load() }
  function rerun(rec) { nav(`/run?model=${encodeURIComponent(rec.model)}&task=${encodeURIComponent(rec.recipe)}`) }

  if (err) return <div className="alert err">{err}</div>
  if (jobs === null) return <div className="spinner">Loading…</div>
  if (jobs.length === 0) return (
    <div><h1>Jobs</h1><div className="alert info">No jobs submitted yet. Launch one from the Run page.</div></div>
  )

  return (
    <div>
      <h1>Jobs</h1>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="small" onClick={load}>Refresh</button>
        <label className="inline"><input type="checkbox" checked={auto}
          onChange={(e) => setAuto(e.target.checked)} /> Auto-refresh (10s)</label>
      </div>

      {jobs.map((rec) => {
        const jid = rec.jobid || '?'
        const state = rec.state || rec.last_state || 'UNKNOWN'
        const icon = STATE_ICON[state] || '❔'
        const code = rec.exit_code
        const codeStr = code && state !== 'COMPLETED' ? `  (exit ${code})` : ''
        const metrics = rec.metrics || {}
        const isSyn = rec.model === 'ORBIT-2' && rec.params && rec.params.ORBIT2_USE_SYNTHETIC === '1'
        const helpLine = STATE_HELP[state]
        const alertCls = state === 'COMPLETED' ? 'ok' : (icon === '🔴' ? 'err' : 'info')
        return (
          <div key={jid} className="card" style={{ marginBottom: 12 }}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <b>{icon} {jid} — {rec.model} / {rec.recipe} — {state}{codeStr}</b>
            </div>
            {helpLine && <div className={`alert ${alertCls}`}>{helpLine}</div>}

            <div className="row" style={{ margin: '8px 0' }}>
              <div className="metric"><div className="val">{(rec.overrides || {}).nodes ?? '?'}</div><div className="lbl">Nodes</div></div>
              <div className="metric"><div className="val">{(rec.overrides || {}).gpus ?? '?'}</div><div className="lbl">GPUs/node</div></div>
              <div className="caption">Submitted: {rec.submit_time || '?'}<br />Partition: {(rec.overrides || {}).partition || '?'}</div>
            </div>

            {Object.keys(metrics).length > 0 && (
              <div>
                <h3>Results</h3>
                <div className="metrics">
                  {Object.entries(metrics).map(([k, v]) =>
                    <div className="metric" key={k}><div className="val">{v}</div><div className="lbl">{k}</div></div>)}
                </div>
                {isSyn && <div className="caption">Synthetic-data smoke test — these confirm the pipeline runs end-to-end, not model accuracy. Use real ERA5/PRISM data for meaningful PSNR/SSIM.</div>}
              </div>
            )}

            {rec.params && Object.keys(rec.params).length > 0 && (
              <details><summary>Parameters</summary>
                <pre className="code">{JSON.stringify(rec.params, null, 2)}</pre></details>
            )}

            <div className="caption">Log: <code>{rec.outfile || '(none)'}</code></div>
            {rec.outfile && (
              <div style={{ marginTop: 6 }}>
                <button className="small" onClick={() => toggleLog(jid)}>
                  {logs[jid] !== undefined ? 'Hide log' : 'Show full log'}
                </button>
                {logs[jid] !== undefined && <pre className="code">{logs[jid]}</pre>}
              </div>
            )}

            <div className="row" style={{ marginTop: 10 }}>
              {['PENDING', 'RUNNING', 'COMPLETING'].includes(state) &&
                <button className="small danger" onClick={() => cancel(jid)}>Cancel</button>}
              <button className="small" onClick={() => rerun(rec)}>Re-run</button>
              <button className="small" onClick={() => remove(jid)}>Remove</button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
