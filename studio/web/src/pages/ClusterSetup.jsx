import { useEffect, useState } from 'react'
import { api } from '../api.js'

export default function ClusterSetup() {
  const [cluster, setCluster] = useState(null)
  const [disc, setDisc] = useState(null)
  const [discovering, setDiscovering] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(null)
  const [err, setErr] = useState('')

  // Confirmed form fields.
  const [f, setF] = useState({
    gpu_arch: '', gpu_count: 8, vram_gb: 0,
    partition: '', account: '', scratch: '', runtime: 'apptainer',
    location: 'repo',
  })

  useEffect(() => {
    api.getCluster().then(setCluster).catch((e) => setErr(String(e)))
  }, [])

  async function discover() {
    setErr(''); setDiscovering(true); setSaved(null)
    try {
      const d = await api.discover()
      setDisc(d)
      const ex = (cluster && cluster.config) || {}
      const exS = ex.slurm || {}, exP = ex.paths || {}, exC = ex.containers || {}
      setF((prev) => ({
        ...prev,
        gpu_arch: d.gpu_arch || '',
        gpu_count: d.gpu_count || 8,
        vram_gb: d.vram_gb || 0,
        partition: (d.partitions && d.partitions[0]) || exS.partition || '',
        account: (d.accounts && d.accounts[0]) || exS.account || '',
        scratch: (d.scratch_dirs && d.scratch_dirs[0]) || exP.scratch || '',
        runtime: (d.runtimes && d.runtimes.includes('apptainer')) ? 'apptainer'
          : (d.runtimes && d.runtimes[0]) || exC.runtime || 'apptainer',
      }))
    } catch (e) { setErr(String(e)) } finally { setDiscovering(false) }
  }

  async function save() {
    setErr(''); setSaving(true)
    try {
      const payload = {
        gpu_arch: f.gpu_arch, gpu_count: Number(f.gpu_count), vram_gb: Number(f.vram_gb),
        partition: f.partition, account: f.account, scratch: f.scratch,
        runtime: f.runtime, location: f.location,
        scratch_local: (disc && disc.scratch_local) || '',
        internet: (disc && disc.internet) || false,
        proxy: (disc && disc.proxy) || '',
        net_ifaces: (disc && disc.net_ifaces) || [],
        ib_hcas: (disc && disc.ib_hcas) || [],
        perf_tools: (disc && disc.perf_tools) || '',
      }
      const res = await api.saveCluster(payload)
      setSaved(res)
      api.getCluster().then(setCluster).catch(() => {})
    } catch (e) { setErr(String(e)) } finally { setSaving(false) }
  }

  const set = (k) => (e) => setF({ ...f, [k]: e.target.value })

  return (
    <div>
      <h1>Cluster setup</h1>
      <p className="muted">
        Discover this cluster's settings and save them to <code>.cluster-config.yaml</code>.
        GUI equivalent of the <code>/init-cluster</code> agent command.
      </p>
      {err && <div className="alert err">{err}</div>}

      {cluster && (cluster.exists
        ? <div className="alert ok">Current config: <code>{cluster.active_path}</code></div>
        : <div className="alert warn">No cluster config found yet. Run discovery below.</div>)}

      <button className="primary" onClick={discover} disabled={discovering}>
        {discovering ? 'Probing…' : 'Discover cluster settings'}
      </button>

      {disc && (
        <div style={{ marginTop: 16 }}>
          <h2>Confirm settings</h2>
          <div className="two-col">
            <div className="card">
              <h3>GPU</h3>
              <div className="field"><label>Architecture</label>
                <input type="text" value={f.gpu_arch} onChange={set('gpu_arch')} />
                {disc.gpu_arch_label && <div className="caption">{disc.gpu_arch_label}</div>}</div>
              <div className="field"><label>GPUs per node</label>
                <input type="number" value={f.gpu_count} onChange={set('gpu_count')} /></div>
              <div className="field"><label>VRAM per GPU (GB)</label>
                <input type="number" value={f.vram_gb} onChange={set('vram_gb')} /></div>

              <h3>SLURM</h3>
              <div className="field"><label>Partition</label>
                <input type="text" list="partitions" value={f.partition} onChange={set('partition')} />
                <datalist id="partitions">{(disc.partitions || []).map((p) => <option key={p} value={p} />)}</datalist></div>
              <div className="field"><label>Account</label>
                <input type="text" list="accounts" value={f.account} onChange={set('account')} />
                <datalist id="accounts">{(disc.accounts || []).map((a) => <option key={a} value={a} />)}</datalist></div>
            </div>

            <div className="card">
              <h3>Container runtime</h3>
              <div className="field"><label>Runtime</label>
                <select value={f.runtime} onChange={set('runtime')}>
                  {(disc.runtimes && disc.runtimes.length ? disc.runtimes : ['apptainer']).map((r) =>
                    <option key={r} value={r}>{r}</option>)}
                </select>
                {f.runtime !== 'apptainer' &&
                  <div className="alert warn">Studio v1 runs Apptainer only. Docker support is planned.</div>}</div>

              <h3>Storage</h3>
              <div className="field"><label>Scratch / shared root</label>
                <input type="text" list="scratch" value={f.scratch} onChange={set('scratch')} />
                <datalist id="scratch">{(disc.scratch_dirs || []).map((s) => <option key={s} value={s} />)}</datalist>
                <div className="caption">Exported as AI4S_SHARED_DIR. Layout: &lt;scratch&gt;/images/ and &lt;scratch&gt;/models/&lt;Model&gt;/</div></div>

              <h3>Network</h3>
              <div className="caption">Internet: {disc.internet ? 'yes' : 'no'}</div>
              <div className="caption">Interfaces: {(disc.net_ifaces || []).join(', ') || '—'}</div>
              <div className="caption">IB HCAs: {(disc.ib_hcas || []).join(', ') || '—'}</div>
            </div>
          </div>

          {disc.sifs && disc.sifs.length > 0 && (
            <details className="card" style={{ marginTop: 12 }}>
              <summary>Discovered SIF images ({disc.sifs.length})</summary>
              {disc.sifs.map((s) => <pre key={s} className="code">{s}</pre>)}
            </details>
          )}

          <div className="field" style={{ marginTop: 12 }}>
            <label>Save location</label>
            <select value={f.location} onChange={set('location')}>
              <option value="repo">Repo root ({cluster && cluster.repo_config_path})</option>
              <option value="user">User home ({cluster && cluster.user_config_path})</option>
            </select>
          </div>

          <button className="primary" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save config'}
          </button>
          {saved && (
            <div style={{ marginTop: 12 }}>
              <div className="alert ok">Saved to <code>{saved.path}</code> (gitignored). Extra blocks preserved.</div>
              <pre className="code">{JSON.stringify(saved.config, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
