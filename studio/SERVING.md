# AI4Science Studio — Remote Serving Runbook

Authoritative, tested procedure for running the Studio demo on the lux MI355X
cluster and reaching it from a laptop. This is the source of truth — do not
re-derive it each session.

## Topology

```
laptop browser                login node                 compute node (lux-mi355x-a6)
  localhost:5376  ── ssh -L ──► rad-vultr-login ── SLURM ──► vite   :5376  (frontend)
                                (job 17479 alloc)            uvicorn:8376 (backend)
                                                             vite proxies /api → 127.0.0.1:8376
```

- **Persistent allocation:** SLURM job `17479` (a `bash` step, partition `lux`,
  node `lux-mi355x-a6`) holds the node. All services attach to it with
  `srun --jobid=17479 --overlap`. If this job is gone, everything below fails —
  re-request an allocation first (see "If the allocation is gone").
- The login node **can** reach the compute node's `0.0.0.0` ports directly, so
  the laptop tunnels to the login node and traffic flows to the compute node.
- Compute nodes are **air-gapped** (no DNS/internet). Only the login node has
  network. Never try to `apt-get`/`pip install`/`npm install` on the compute node.

## Ports (fixed — do not change)

| Service  | Port  | Bind        | SLURM step name |
|----------|-------|-------------|-----------------|
| Frontend (vite) | 5376 | 0.0.0.0 | `hg_frontend` |
| Backend (uvicorn) | 8376 | 0.0.0.0 | `hg_backend` |

Frontend proxies `/api` → `http://127.0.0.1:8376` (see
[frontend/vite.config.js](frontend/vite.config.js)). Both processes run on the
**same** compute node so `127.0.0.1` works.

> Note: `launch-local.sh` uses different ports (5275/8275) for a purely local
> laptop run. The remote demo uses 5376/8376. Do not mix them.

## Public URL (Cloudflare tunnel)

The demo is served publicly at **https://ai4science-studio.axiomfabric.ai** via a
Cloudflare named tunnel (`ai4science-studio`). `cloudflared` runs on the **login
node** (compute nodes have no outbound DNS) and its ingress points at the
compute-node frontend by hostname (`http://lux-mi355x-a6:5376`), so no SSH tunnel
is involved for the public path.

- Do NOT confuse this with the apex `axiomfabric.ai`, which serves a separate
  Apache site — leave it alone.
- Full tunnel runbook + start command: `/shared/spannala/STUDIO_RUNBOOK.md` and
  `/shared/spannala/.cloudflared/config.yml`. If the studio moves to a new
  compute node, update the ingress `service:` hostname and restart cloudflared on
  the login node.
- Health check: `curl https://ai4science-studio.axiomfabric.ai/api/health` →
  `{"ok":true,...}`.

## SSH tunnel (from the laptop)

For local/dev access without the public tunnel:

```bash
ssh -N -L 5376:localhost:5376 rad-vultr-login
```

Then open **http://localhost:5376** in the laptop browser. `/api` calls are
proxied by vite to the backend automatically — no second tunnel needed.

## Start / restart the backend (uvicorn, port 8376)

Run these from the **login node**. The backend does NOT use `--reload`, so any
edit to `backend/*.py` requires a restart.

```bash
# 1. Kill the old step (login-node srun wrappers; scancel can't target the step)
pkill -f "job-name=hg_backend"; sleep 3

# 2. Confirm port 8376 is free on the compute node
srun --jobid=17479 --overlap bash -c "ss -ltn | grep -c 8376"   # expect 0 (or exit 1)

# 3. Relaunch (note the cd — uvicorn uses relative paths)
nohup srun --jobid=17479 --overlap --job-name=hg_backend bash -c \
  'cd /home/spannala/Projects/ai4science-studio/studio/backend && \
   HG_USE_PBC_EDGES=1 exec .venv/bin/python3 .venv/bin/uvicorn main:app \
   --host 0.0.0.0 --port 8376' \
  > /home/spannala/Projects/ai4science-studio/studio/backend/backend_restart.log 2>&1 &

# 4. Confirm startup
sleep 4; tail -6 /home/spannala/Projects/ai4science-studio/studio/backend/backend_restart.log
# expect: "Uvicorn running on http://0.0.0.0:8376"
```

## Start / restart the frontend (vite, port 5376)

The frontend has HMR, so edits to `frontend/src/**` are picked up live — you
rarely need to restart it. Restart only if HMR stops applying (NFS can break the
file watcher; `CHOKIDAR_USEPOLLING=true` mitigates it) or after changing
`vite.config.js` / installing deps.

```bash
pkill -f "job-name=hg_frontend"; sleep 2
nohup srun --jobid=17479 --overlap --job-name=hg_frontend bash -c \
  'cd /home/spannala/Projects/ai4science-studio/studio/frontend && \
   CHOKIDAR_USEPOLLING=true exec npm run dev -- --host 0.0.0.0 --port 5376' \
  > /home/spannala/Projects/ai4science-studio/studio/frontend/frontend_restart.log 2>&1 &
sleep 5; tail -8 /home/spannala/Projects/ai4science-studio/studio/frontend/frontend_restart.log
```

## Health checks

```bash
# Both ports listening on the compute node
srun --jobid=17479 --overlap bash -c "ss -ltn | grep -E '5376|8376'"

# Backend responds (405 on / is fine — server is alive)
srun --jobid=17479 --overlap bash -c \
  "curl -s -o /dev/null -w 'backend %{http_code}\n' http://127.0.0.1:8376/"

# From the laptop, after the tunnel is up:
#   curl -s -o /dev/null -w '%{http_code}\n' http://localhost:5376/
```

## Common failure modes (already diagnosed — don't re-investigate)

- **Blank screen after edits / stale interface:** vite dep re-optimization served
  a stale chunk hash. Hard-refresh (Ctrl+Shift+R). If persistent, restart vite.
- **HMR not applying edits:** NFS breaks the inotify watcher. Restart vite with
  `CHOKIDAR_USEPOLLING=true` (already in the launch command above).
- **Backend dies when stopped via `srun bash -c`:** one-shot srun steps tear down
  the process group. Always launch as a backgrounded persistent `srun --overlap`
  step (as above), never inline.
- **`scancel --jobid` / `scancel -n hg_backend` errors:** those flags aren't
  supported here. Kill the login-node wrappers with
  `pkill -f "job-name=hg_backend"` instead.
- **Backend edits not taking effect:** uvicorn runs without `--reload`. Restart it.

## If the allocation (job 17479) is gone

Everything above assumes job `17479` on `lux-mi355x-a6`. If `squeue -u $USER`
shows it's gone, request a fresh allocation and substitute the new job id in
every `--jobid=` above:

```bash
salloc --partition=lux --nodes=1 --time=24:00:00   # note the new JOBID and node
```

If the new node is not `lux-mi355x-a6`, update the SSH-tunnel target check and
confirm both services land on that same node (they will, since they attach to the
same `--jobid`).
