// Tiny fetch wrapper for the FastAPI backend. Uses same-origin relative URLs:
// in dev the Vite proxy forwards /api -> 127.0.0.1:8600; in prod FastAPI serves
// both the SPA and the API from the same port.

async function request(path, { method = 'GET', body } = {}) {
  const opts = { method, headers: {} }
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(path, opts)
  let data = null
  const text = await res.text()
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { raw: text }
  }
  if (!res.ok) {
    const detail = data && (data.detail || data.error) ? (data.detail || data.error) : res.statusText
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    err.status = res.status
    err.detail = detail
    err.data = data
    throw err
  }
  return data
}

export const api = {
  // catalog
  listModels: () => request('/api/models'),
  getModel: (slug) => request(`/api/models/${encodeURIComponent(slug)}`),
  filters: () => request('/api/filters'),

  // cluster
  getCluster: () => request('/api/cluster'),
  discover: () => request('/api/cluster/discover', { method: 'POST', body: {} }),
  saveCluster: (payload) => request('/api/cluster', { method: 'POST', body: payload }),

  // run
  runForm: (slug, task) =>
    request(`/api/models/${encodeURIComponent(slug)}/recipes/${encodeURIComponent(task)}/form`),
  preview: (payload) => request('/api/run/preview', { method: 'POST', body: payload }),
  submit: (payload) => request('/api/run/submit', { method: 'POST', body: payload }),

  // jobs
  listJobs: () => request('/api/jobs'),
  jobLog: (jobid) => request(`/api/jobs/${encodeURIComponent(jobid)}/log`),
  cancelJob: (jobid) => request(`/api/jobs/${encodeURIComponent(jobid)}/cancel`, { method: 'POST' }),
  removeJob: (jobid) => request(`/api/jobs/${encodeURIComponent(jobid)}`, { method: 'DELETE' }),

  // scaffold
  scaffoldDomains: () => request('/api/scaffold/domains'),
  createModel: (payload) => request('/api/models', { method: 'POST', body: payload }),
}
