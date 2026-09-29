import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { StepHeader, StatusBadge, Spinner } from '../components/ui'

export function StepRun() {
  const { model, domain, prompt, customPrompt, mode, partition, setPartition,
          runId, setRunId, runState, setRunState, setResult, setStep,
          params, resetRun, task, modelVariant } = useStore()
  const [log, setLog] = useState([])
  const [launching, setLaunching] = useState(false)
  const [partitions, setPartitions] = useState([])
  const logRef = useRef(null)
  const esRef = useRef(null)
  const pollRef = useRef(null)

  const activePrompt = customPrompt || prompt

  useEffect(() => {
    api.slurmPartitions().then(d => {
      if (d.partitions) setPartitions(d.partitions)
      // Preselect the site default (AI4S_SLURM_PARTITION) if nothing is chosen yet.
      if (!useStore.getState().partition && d.default) setPartition(d.default)
    }).catch(() => {})
  }, [])

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  // Cleanup SSE on unmount
  useEffect(() => () => {
    try { esRef.current?.close() } catch { /* noop */ }
    if (pollRef.current) clearInterval(pollRef.current)
  }, [])

  async function launch() {
    setLaunching(true)
    setLog([])
    resetRun()
    try {
      const { run_id } = await api.launchJob({
        slug: model.slug,
        domain,
        task: task || 'inference',
        mode,
        prompt: activePrompt,
        params: { ...params, model_variant: modelVariant },
        partition,
      })
      setRunId(run_id)
      setRunState('running')

      // Live training runs stream their telemetry on the Analyze › Performance tab,
      // so jump there right after launch (the charts warm up ~1 min in, once
      // Omnistat's first push lands). The Analyze step owns completion detection, so
      // leaving this step early does not drop the final harvest. Demo runs and live
      // inference stay here and advance on completion via finalize() below.
      if (mode !== 'demo' && (task === 'train')) {
        setStep(4)
        return
      }

      // Finalize once — whichever path (SSE 'done' or polling) sees completion first.
      let finalized = false
      const finalize = async () => {
        if (finalized) return
        const job = await api.job(run_id)
        if (job.state !== 'completed' && job.state !== 'failed') return
        finalized = true
        try { esRef.current?.close() } catch { /* noop */ }
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
        setRunState(job.state)
        if (job.result) setResult(job.result)
        if (job.state === 'completed') setStep(4)
      }

      // Primary: SSE stream for live logs + completion.
      const es = new EventSource(`/api/jobs/${run_id}/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        const data = JSON.parse(e.data)
        if (data.line) setLog(l => [...l, data.line])
        if (data.done) finalize()
      }
      es.onerror = () => { try { es.close() } catch { /* noop */ } }

      // Fallback: poll job status every 2s. Zscaler Browser Isolation often kills
      // long-lived SSE streams, so this guarantees results still load.
      pollRef.current = setInterval(() => { finalize().catch(() => {}) }, 2000)
    } catch (e) {
      setLog(l => [...l, `[error] ${e.message}`])
      setRunState('failed')
    } finally {
      setLaunching(false)
    }
  }

  return (
    <div>
      <StepHeader
        title="Run"
        sub={`${model?.name || model?.slug} — ${
          mode === 'demo'
            ? ((model?.slug === 'ORBIT-2' && task === 'story') || model?.slug === 'HydraGNN'
                ? 'Demo (replay of real runs)' : 'Demo (synthetic)')
            : 'Live SLURM'} mode`}
      />
      <button className="btn btn-ghost" style={{ marginBottom: '1.2rem', fontSize: '.8rem' }}
        onClick={() => setStep(2)}>← Back</button>

      {/* Config summary */}
      <div className="card" style={{ padding: '1rem', marginBottom: '1.2rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.75rem', fontSize: '.85rem' }}>
          <div>
            <span style={{ color: '#52525b' }}>Model: </span>
            <span style={{ color: '#f5f5f7', fontWeight: 600 }}>{model?.name || model?.slug}</span>
          </div>
          <div>
            <span style={{ color: '#52525b' }}>Mode: </span>
            <span className={`badge badge-${mode === 'demo' ? 'ok' : 'amd'}`}>{mode.toUpperCase()}</span>
            {model?.slug === 'HydraGNN' && (
              <span className="badge badge-info" style={{ marginLeft: '.4rem' }}>
                {task === 'train' ? 'training scaling' : `inference · ${modelVariant}`}
              </span>
            )}
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <span style={{ color: '#52525b' }}>Prompt: </span>
            <span style={{ color: '#94a3b8' }}>{activePrompt.slice(0, 120)}{activePrompt.length > 120 ? '...' : ''}</span>
          </div>
        </div>
      </div>

      {/* Partition picker (live mode only) */}
      {mode === 'live' && (
        <div style={{ marginBottom: '1.2rem' }}>
          <label className="section-label" style={{ display: 'block', marginBottom: '.4rem' }}>SLURM Partition</label>
          <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
            {(partitions.length > 0 ? partitions
              : (partition ? [{ partition }] : [])).map(p => (
              <button key={p.partition}
                className={`btn ${partition === p.partition ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize: '.78rem', padding: '.3rem .7rem' }}
                onClick={() => setPartition(p.partition)}>
                {p.partition}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Launch button */}
      {runState === 'idle' && (
        <button className="btn btn-primary" style={{ marginBottom: '1.2rem' }}
          disabled={launching} onClick={launch}>
          {launching ? <Spinner size={14} /> : null}
          {launching ? 'Launching...' : mode === 'demo' ? '▶ Run Demo' : '▶ Submit SLURM Job'}
        </button>
      )}

      {/* Status */}
      {runState !== 'idle' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', marginBottom: '1rem' }}>
          <StatusBadge state={runState} />
          {runId && <span style={{ fontSize: '.75rem', color: '#52525b' }}>Run: {runId.slice(0, 8)}...</span>}
          {(runState === 'completed' || runState === 'failed') && (
            <button className="btn btn-ghost" style={{ fontSize: '.75rem', padding: '.25rem .6rem' }}
              onClick={() => { resetRun(); setLog([]); }}>Reset</button>
          )}
          {runState === 'completed' && (
            <button className="btn btn-primary" style={{ fontSize: '.75rem', padding: '.25rem .7rem' }}
              onClick={() => setStep(4)}>View Results →</button>
          )}
        </div>
      )}

      {/* Log */}
      {log.length > 0 && (
        <div className="log-pane" ref={logRef}>
          {log.join('\n')}
        </div>
      )}
    </div>
  )
}
