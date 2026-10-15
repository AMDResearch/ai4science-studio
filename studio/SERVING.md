# AI4Science Studio: Remote Serving Runbook

Tested procedure for running the Studio on a SLURM GPU cluster (validated on
AMD Instinct MI355X nodes) and reaching it from a laptop. Site-specific values
(shared directory, partition, account, host names) are never hardcoded; they
come from the environment. Keep notes that only apply to your own deployment in
`studio/SERVING.local.md` (gitignored).

> **The public tunnel is optional and off by default.** A default bring-up is
> reached over the login-to-compute SSH forward described below. Only start a
> Cloudflare tunnel when you explicitly want a public URL.

## Site configuration

Copy [`.env.example`](.env.example) to `studio/.env` (gitignored) and fill it in.
The backend loads `studio/.env` at startup (variables already set in the
environment win), so the same checkout works on any cluster. For hand-submitted
batch scripts, export the same values in your shell first:

```bash
set -a; . studio/.env; set +a
```

| Variable | Used by | Purpose |
|----------|---------|---------|
| `AI4S_SHARED_DIR` | backend (live mode), telemetry and study scripts | Site shared directory holding container images, overlays, weights, and run output. Required for live runs; demo (replay) mode needs nothing. |
| `AI4S_SLURM_PARTITION` | backend | Default partition for live runs (the Run step can pick another). |
| `AI4S_SLURM_ACCOUNT` | backend | SLURM account for live runs; leave empty to use your default account. |
| `SBATCH_PARTITION`, `SBATCH_ACCOUNT` | `sbatch` | Standard SLURM variables. They override the `YOUR_PARTITION_HERE` / `YOUR_ACCOUNT_HERE` placeholders in the committed `#SBATCH` headers. |
| `AI4S_SIF` | backend | Container image (default `$AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif`). |
| `PERF_TOOLS_DIR` | backend, `telemetry/` | Omnistat venv and VictoriaMetrics binary (default `$AI4S_SHARED_DIR/perf-tools`). |
| `HG_BASE`, `HG_DATA_DIR` | backend, `telemetry/` | HydraGNN work tree (default `$AI4S_SHARED_DIR/models/HydraGNN`) and Alexandria dataset directory (default `$HG_BASE/weights`). |
| `ORBIT2_BASE`, `ORBIT2_ROOT`, `ORBIT2_HF_CACHE`, `ORBIT2_SR_DIR` | backend, `backend/tools/orbit2_sr_snapshot/` | ORBIT-2 model tree, upstream code checkout, pretrained checkpoint cache (default `~/.cache/huggingface/orbit2`), and the DC downscaling study directory. |
| `GPMOL_BASE` | backend | GP-MoLFormer work directory (default `$AI4S_SHARED_DIR/models/GP-MoLFormer`). |
| `AI4S_LOCAL_SCRATCH` | demo recorders | Node-local scratch for the staged Chromium (default `$TMPDIR/$USER` or `/tmp/$USER`). NFS homes cannot mmap Chromium's ICU data. |
| `STUDIO_HOST` | `demo/record_demo.slurm` | Host (usually an internal IP) running the services the recorder connects to. |
| `STUDIO_LOGIN_HOST`, `STUDIO_PUBLIC_URL` | `launch-local.sh` | Only used to print the laptop tunnel command and the public URL. |

A live launch with a missing required value fails immediately with a message
naming the variable to set; it never submits a half-configured job.

## Topology

```
laptop browser              login node                  compute node (<compute-node>)
  localhost:5376 ── ssh -L ──► <login-node> ── SLURM ──► vite   :5376  (frontend)
                              (holder allocation)        uvicorn:8376 (backend)
                                                         vite proxies /api → 127.0.0.1:8376
```

- **Persistent allocation:** a holder job (for example
  [`scripts/launch/studio-hold-4d.sbatch`](scripts/launch/studio-hold-4d.sbatch))
  keeps a node. All services attach to it with `srun --jobid=<jobid> --overlap`.
  If the job is gone, everything below fails; request a new allocation first
  (see "If the allocation is gone").
- The login node **can** reach the compute node's `0.0.0.0` ports directly, so
  the laptop tunnels to the login node and traffic flows to the compute node.
- Compute nodes on air-gapped clusters have no DNS or internet. Run
  `pip install` / `npm install` on the login node before launching.

## Ports (fixed; do not change)

| Service  | Port  | Bind        | SLURM step name |
|----------|-------|-------------|-----------------|
| Frontend (vite) | 5376 | 0.0.0.0 | `hg_frontend` |
| Backend (uvicorn) | 8376 | 0.0.0.0 | `hg_backend` |

The frontend proxies `/api` to `http://127.0.0.1:8376` (see
[frontend/vite.config.js](frontend/vite.config.js)). Both processes run on the
**same** compute node so `127.0.0.1` works.

> [`launch-local.sh`](launch-local.sh) uses the same ports (backend `8376` with
> `--reload`, frontend `5376`), so the vite `/api` proxy target and any tunnel
> config work unchanged whether the stack runs on the login node or via `srun`
> on a compute node. Frontend JSX is hot-reloaded by vite; bump the `?v=` rev in
> `index.html` only to bust a stale browser or proxy cache.

## Public URL (optional Cloudflare tunnel)

To publish the Studio, run a Cloudflare named tunnel on the **login node**
(compute nodes may have no outbound DNS) with its ingress pointing at the
compute-node frontend by hostname, for example
`service: http://<compute-node>:5376`. When the Studio moves to a new node,
update that ingress and restart `cloudflared`:

```bash
# 1. Edit the ingress in your cloudflared config.yml:
#      service: http://<compute-node>:5376
# 2. Restart cloudflared on the login node:
pkill -f 'cloudflared tunnel'; sleep 2
nohup cloudflared tunnel --protocol http2 --config <config.yml> run <tunnel-name> \
  > cloudflared_restart.log 2>&1 &
# 3. Verify: curl -s -o /dev/null -w '%{http_code}\n' "$STUDIO_PUBLIC_URL/api/health"   # 200
```

`--protocol http2` avoids QUIC, which some site firewalls block.

## SSH tunnel (from the laptop)

The laptop can usually reach ONLY the login node, never the compute nodes. So the
path is two hops:

```
laptop  --ssh -L 5376:localhost:5376-->  login node  --ssh -L 127.0.0.1:5376:localhost:5376-->  compute node
```

**Hop 1 (persistent, on the LOGIN node): a login-to-compute forward** so that the
login node's own `127.0.0.1:5376` maps to the compute node's frontend.
`launch-local.sh` starts this automatically; to (re)create it manually:

```bash
# Find the node holding your allocation: squeue -j <jobid> -h -o '%N'
nohup ssh -N -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -L 127.0.0.1:5376:localhost:5376 <compute-node> \
  > studio/logs/headnode_tunnel.log 2>&1 &
# verify: curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5376/   # 200
```

**Hop 2 (from the LAPTOP):** tunnel to the login node's `localhost:5376` (not the
compute hostname, which the laptop cannot resolve or reach):

```bash
ssh -N -L 5376:localhost:5376 <login-node>
```

Then open **http://localhost:5376** in the laptop browser. `/api` calls are
proxied by vite to the compute-node backend (:8376) automatically.

**If the Studio moves nodes:** update BOTH the login-to-compute forward (kill the
old `ssh ... <old-node>` and recreate it against the new node) AND the tunnel
ingress, if you run one. Find the node with `squeue -j <jobid> -h -o '%N'`.

## Start / restart the backend (uvicorn, port 8376)

Run these from the **login node**, from the repository root. This launch does
not use `--reload`, so any edit to `backend/*.py` requires a restart. Site
settings come from `studio/.env`.

```bash
STUDIO="$PWD/studio"

# 1. Kill the old step (login-node srun wrappers; scancel cannot target the step)
pkill -f "job-name=hg_backend"; sleep 3

# 2. Confirm port 8376 is free on the compute node
srun --jobid=<jobid> --overlap bash -c "ss -ltn | grep -c 8376"   # expect 0 (or exit 1)

# 3. Relaunch (note the cd: uvicorn uses relative paths)
nohup srun --jobid=<jobid> --overlap --job-name=hg_backend bash -c \
  "cd $STUDIO/backend && \
   HG_USE_PBC_EDGES=1 exec .venv/bin/python3 .venv/bin/uvicorn main:app \
   --host 0.0.0.0 --port 8376" \
  > "$STUDIO/backend/logs/backend_restart.log" 2>&1 &

# 4. Confirm startup
sleep 4; tail -6 "$STUDIO/backend/logs/backend_restart.log"
# expect: "Uvicorn running on http://0.0.0.0:8376"
```

## Start / restart the frontend (vite, port 5376)

The frontend has HMR, so edits to `frontend/src/**` are picked up live and you
rarely need to restart it. Restart only if HMR stops applying (NFS can break the
file watcher; `CHOKIDAR_USEPOLLING=true` mitigates it) or after changing
`vite.config.js` or installing dependencies.

```bash
STUDIO="$PWD/studio"
pkill -f "job-name=hg_frontend"; sleep 2
nohup srun --jobid=<jobid> --overlap --job-name=hg_frontend bash -c \
  "cd $STUDIO/frontend && \
   CHOKIDAR_USEPOLLING=true exec npm run dev -- --host 0.0.0.0 --port 5376" \
  > "$STUDIO/frontend/logs/frontend_restart.log" 2>&1 &
sleep 5; tail -8 "$STUDIO/frontend/logs/frontend_restart.log"
```

## Health checks

```bash
# Both ports listening on the compute node
srun --jobid=<jobid> --overlap bash -c "ss -ltn | grep -E '5376|8376'"

# Backend responds
srun --jobid=<jobid> --overlap bash -c \
  "curl -s -o /dev/null -w 'backend %{http_code}\n' http://127.0.0.1:8376/api/health"

# From the laptop, after the tunnel is up:
#   curl -s -o /dev/null -w '%{http_code}\n' http://localhost:5376/
```

## Common failure modes (already diagnosed)

- **Blank screen after edits / stale interface:** vite dependency re-optimization
  served a stale chunk hash. Hard-refresh (Ctrl+Shift+R). If persistent, restart
  vite.
- **HMR not applying edits:** NFS breaks the inotify watcher. Restart vite with
  `CHOKIDAR_USEPOLLING=true` (already in the launch command above).
- **Backend dies when stopped via `srun bash -c`:** one-shot srun steps tear down
  the process group. Always launch as a backgrounded persistent
  `srun --overlap` step (as above), never inline.
- **`scancel` cannot target the step on some sites:** kill the login-node
  wrappers with `pkill -f "job-name=hg_backend"` instead.
- **Backend edits not taking effect:** uvicorn runs without `--reload`. Restart it.
- **Live launch returns HTTP 400 naming a variable:** that site setting is
  missing; add it to `studio/.env` and restart the backend.

## If the allocation is gone

If `squeue -u $USER` shows the holder job is gone, request a fresh allocation
and substitute the new job id in every `--jobid=` above:

```bash
SBATCH_PARTITION=<partition> SBATCH_ACCOUNT=<account> \
  sbatch studio/scripts/launch/studio-hold-4d.sbatch   # note the new JOBID
squeue -j <jobid> -h -o '%N'                          # and its node
```

Then update the SSH forward (and tunnel ingress, if any) to the new node. Both
services land on that same node because they attach to the same `--jobid`.
