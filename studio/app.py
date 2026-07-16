"""AI4Science Studio — web GUI entry point.

Runs on a cluster login node; browse models, set up the cluster, run jobs,
monitor them, and scaffold new models. Reads repo metadata and submits the
repo's own sbatch scripts unmodified.
"""
from __future__ import annotations

import socket

import streamlit as st

st.set_page_config(
    page_title="AI4Science Studio",
    page_icon="\U0001F9EA",  # test tube
    layout="wide",
)


def _home() -> None:
    st.title("AI4Science Studio")
    st.caption(
        "A simple GUI for running open AI-for-science models on your own cluster."
    )
    st.markdown(
        f"You are on login node **`{socket.gethostname()}`**. "
        "Jobs you launch here run as **you**, via your cluster's SLURM scheduler."
    )

    from core import catalog, cluster_config

    col1, col2, col3 = st.columns(3)
    try:
        models = catalog.load_catalog()
        col1.metric("Models available", len(models))
    except Exception as exc:  # noqa: BLE001
        col1.metric("Models available", "?")
        st.warning(f"Could not read models.yaml: {exc}")

    cfg_path = cluster_config.active_config_path()
    if cfg_path:
        col2.metric("Cluster config", "found")
        col2.caption(str(cfg_path))
    else:
        col2.metric("Cluster config", "not set")
        col2.caption("Run **Cluster setup** first.")

    import shutil

    col3.metric("SLURM", "yes" if shutil.which("sbatch") else "no")
    if not shutil.which("sbatch"):
        col3.caption("Off-cluster: dry-run only.")

    st.divider()
    st.subheader("Getting started")
    st.markdown(
        "1. **Cluster setup** — auto-discover and save your cluster config (one time).\n"
        "2. **Catalog** — browse models and pick a recipe.\n"
        "3. **Run** — fill the form, dry-run to preview, then submit.\n"
        "4. **Jobs** — watch status and tail logs.\n"
        "5. **Add model** — scaffold a new model into the repo."
    )


pages = [
    st.Page(_home, title="Home", icon="\U0001F3E0", default=True),
    st.Page("pages/1_catalog.py", title="Catalog", icon="\U0001F4DA"),
    st.Page("pages/2_model_detail.py", title="Model detail", icon="\U0001F50E"),
    st.Page("pages/3_cluster_setup.py", title="Cluster setup", icon="\U0001F5A5"),
    st.Page("pages/4_run.py", title="Run", icon="\U0001F680"),
    st.Page("pages/5_jobs.py", title="Jobs", icon="\U0001F4CB"),
    st.Page("pages/6_add_model.py", title="Add model", icon="\U00002795"),
]

st.navigation(pages).run()
