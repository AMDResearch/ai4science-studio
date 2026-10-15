const BASE = '/api'

async function _fetch(path, opts = {}) {
  const r = await fetch(BASE + path, opts)
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText)
    throw new Error(`${r.status}: ${text}`)
  }
  return r.json()
}

export const api = {
  health: ()                    => _fetch('/health'),
  domains: ()                   => _fetch('/domains'),
  domainModels: (d)             => _fetch(`/domains/${d}/models`),
  model: (slug)                 => _fetch(`/models/${slug}`),
  prompts: (slug, domain)       => _fetch(`/models/${slug}/prompts?domain=${domain || ''}`),
  validatePrompt: (slug, domain, prompt) => _fetch(`/models/${slug}/validate-prompt`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt, domain}),
  }),
  addModel: (payload)           => _fetch('/models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }),
  launchJob: (payload)          => _fetch('/jobs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }),
  jobs: ()                      => _fetch('/jobs'),
  job: (id)                     => _fetch(`/jobs/${id}`),
  jobLog: (id)                  => _fetch(`/jobs/${id}/log`),
  jobFiles: (id)                => _fetch(`/jobs/${id}/files`),
  telemetryCatalog: ()          => _fetch('/telemetry/catalog'),
  jobTelemetryLive: (id, keys)  => _fetch(`/jobs/${id}/telemetry/live${keys && keys.length ? `?keys=${keys.join(',')}` : ''}`),
  slurmPartitions: ()           => _fetch('/slurm/partitions'),
  slurmQueue: ()                => _fetch('/slurm/queue'),
  convertVideo: (blob)          => {
    const fd = new FormData()
    fd.append('file', blob, 'recording.webm')
    return _fetch('/video/convert', {method: 'POST', body: fd})
  },
}
