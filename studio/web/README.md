# AI4Science Studio — React web GUI

The React (Vite) single-page app for AI4Science Studio. It is a thin frontend
over the FastAPI backend in [`studio/api`](../api), which in turn wraps the same
reusable `studio/core` package the Streamlit reference GUI and the `/run-*`
slash commands use. It reads the same `models.yaml` / `model.yaml` metadata and
`.cluster-config.yaml`, and submits the same tracked `sbatch` scripts unmodified.

## Why the split (build here, run on the cluster)

The React build is a **pure local static-bundle step** (no cluster needed), but
the app must **run on the login node** because that is where SLURM
(`sbatch`/`squeue`) lives — so the built `dist/` is shipped up via `scp`.

---

## Approach A — build locally, ship `dist/`, run on the login node (RECOMMENDED)

This is the normal workflow. The login node has **no npm**; your laptop does.
Build the static bundle on your laptop, copy it up, and let the backend serve it.

Placeholders below: `<user>@<login-node>` (e.g. `alice@rad-vultr-login`) and
`<repo-path>` (the checkout of this repo on the login node).

```bash
# 1) On your LOCAL laptop (has npm) — produces studio/web/dist/
cd studio/web
npm install
npm run build

# 2) Copy the build up to the login node
scp -r studio/web/dist/ <user>@<login-node>:<repo-path>/studio/web/dist/

# 3) On the LOGIN NODE — uvicorn serves the copied dist/ at 127.0.0.1:8600
./studio_web.sh

# 4) From your LOCAL laptop — tunnel in, then open http://localhost:8600
ssh -L 8600:localhost:8600 <user>@<login-node>
```

`studio/web/dist/` is git-ignored, so the build never gets committed — re-run
steps 1–2 whenever the frontend changes. Override the port with
`STUDIO_WEB_PORT=xxxx ./studio_web.sh` (tunnel the same port).

---

## Approach B — Vite dev server (for active UI development only)

Use this only while iterating on the UI. It runs the hot-reloading Vite dev
server on your **laptop** and proxies `/api` requests through the SSH tunnel to
the FastAPI backend running on the login node.

```bash
# On the LOGIN NODE — start just the backend (serves the JSON API at :8600)
./studio_web.sh                 # dist/ need not exist for the API to serve

# On your LOCAL laptop — forward the backend port...
ssh -L 8600:localhost:8600 <user>@<login-node>

# ...then run the dev server (Vite proxies /api -> http://localhost:8600)
cd studio/web
npm install
npm run dev                     # open the URL Vite prints (default :5173)
```

The Vite config proxies `/api` to `http://localhost:8600`, so with the tunnel up
the dev server talks to the real backend (and thus the real SLURM scheduler) on
the login node. This gives hot reload but requires npm + a running tunnel; for
everyday use prefer **Approach A**.

---

## What it does

- **Catalog** — browse all models with domain / task / license filters.
- **Model detail** — view a model's metadata and recipes.
- **Cluster setup** — auto-discover your cluster and save `.cluster-config.yaml`
  (the GUI equivalent of `/init-cluster`).
- **Run** — render a form from the recipe's env vars + scaling knobs, preview the
  exact `sbatch` command (dry-run), then submit.
- **Jobs** — track submitted jobs, poll `squeue`, tail logs, cancel / re-run.
- **Add model** — scaffold a new model folder from `_template/`.

## Notes

- The backend binds to `localhost` only; your SSH login is the auth (see the
  root [`studio/README.md`](../README.md) "Why there is no public URL").
- The frontend is presentation only — every blocker (disclaimer ack, required
  fields, Apptainer-only guard, partition/account) is enforced server-side in
  `studio/api`, mirroring the Streamlit pages in `studio/pages/*.py`.
