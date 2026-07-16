# AI4Science Studio — web GUI

A simple point-and-click GUI for browsing, configuring, running, and monitoring
the models in this repo. It is the no-agent alternative to the `/init-cluster` +
`/run-*` slash commands: it reads the same `models.yaml` / `model.yaml` metadata
and `.cluster-config.yaml`, and submits the same tracked `sbatch` scripts unmodified.

## Running it

The app is meant to run **on your cluster login node** (where you already have
SLURM access). It binds to `localhost` only — reach it from your laptop with an
SSH tunnel.

```bash
# on the login node
cd /path/to/ai4science-studio
./studio.sh                 # first run creates studio/.venv and installs deps

# from your laptop (separate terminal)
ssh -L 8501:localhost:8501 $USER@<login-node>
# then open http://localhost:8501
```

Override the port with `STUDIO_PORT=8600 ./studio.sh`.

## What it does

- **Catalog** — browse all models with domain / task / license filters.
- **Model detail** — view a model's metadata and recipes.
- **Cluster setup** — auto-discover your cluster and save `.cluster-config.yaml`
  (the GUI equivalent of `/init-cluster`).
- **Run** — render a form from the recipe's env vars + scaling knobs, preview the
  exact `sbatch` command (dry-run), then submit. Apptainer runtime, inference +
  training recipes.
- **Jobs** — track submitted jobs, poll `squeue`, tail logs, cancel / re-run.
- **Add model** — scaffold a new model folder from `_template/`.

## Design notes

- Pure Python; depends only on `streamlit` + `pyyaml` (no ML libraries).
- Never edits tracked files to run a job — partition/account/scaling are passed
  as `sbatch` CLI flags; model params as exported environment variables.
- Works off-cluster in **dry-run** mode (no SLURM required) for demos.
