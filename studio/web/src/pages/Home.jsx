import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api.js'

export default function Home() {
  const [models, setModels] = useState(null)
  const [cluster, setCluster] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    api.listModels().then((d) => setModels(d)).catch((e) => setErr(String(e)))
    api.getCluster().then((d) => setCluster(d)).catch(() => {})
  }, [])

  return (
    <div>
      <h1>AI4Science Studio</h1>
      <p className="muted">
        A React UI for running open AI-for-science models on your own cluster.
        Jobs you launch run as <b>you</b>, via your cluster's SLURM scheduler.
      </p>
      {err && <div className="alert err">{err}</div>}

      <div className="grid" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="metric"><div className="val">{models ? models.count : '…'}</div>
            <div className="lbl">Models available</div></div>
        </div>
        <div className="card">
          <div className="metric">
            <div className="val">{cluster ? (cluster.exists ? 'found' : 'not set') : '…'}</div>
            <div className="lbl">Cluster config</div>
          </div>
          {cluster && <div className="caption">{cluster.active_path || 'Run Cluster setup first.'}</div>}
        </div>
        <div className="card">
          <div className="metric">
            <div className="val">{cluster ? (cluster.can_submit ? 'yes' : 'no') : '…'}</div>
            <div className="lbl">SLURM (sbatch)</div>
          </div>
          {cluster && <div className="caption">Host: {cluster.hostname}</div>}
        </div>
      </div>

      <h2>Getting started</h2>
      <ol className="muted">
        <li><Link to="/cluster">Cluster setup</Link> — auto-discover and save your cluster config (one time).</li>
        <li><Link to="/catalog">Catalog</Link> — browse models and pick a recipe.</li>
        <li><Link to="/run">Run</Link> — fill the form, dry-run to preview, then submit.</li>
        <li><Link to="/jobs">Jobs</Link> — watch status and tail logs.</li>
        <li><Link to="/add-model">Add model</Link> — scaffold a new model into the repo.</li>
      </ol>
    </div>
  )
}
