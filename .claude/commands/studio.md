# Launch the Studio web GUI

Start the AI4Science Studio web GUI on this login node so the user can browse
models, set up their cluster, run jobs, and monitor them from a browser.

## Steps

1. From the repo root, run the launcher in the background:
   ```bash
   ./studio.sh
   ```
   (Override the port with `STUDIO_PORT=xxxx ./studio.sh` if 8501 is taken.)

2. The launcher creates a self-contained venv on first run (`studio/.venv`,
   just `streamlit` + `pyyaml`) and starts Streamlit bound to `127.0.0.1`.

3. Tell the user how to reach it from their laptop:
   ```bash
   ssh -L 8501:localhost:8501 $USER@<this-login-node>
   ```
   then open `http://localhost:8501`.

The GUI reads the same `models.yaml` / `model.yaml` metadata and `.cluster-config.yaml`
the agent commands use, and submits the same tracked sbatch scripts — it does not fork
run logic. It is the point-and-click alternative to `/init-cluster` + `/run-*`.

$ARGUMENTS
