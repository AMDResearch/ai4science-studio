"""Cluster setup — auto-discover and save .cluster-config.yaml (GUI /init-cluster)."""
from __future__ import annotations

import streamlit as st

from core import cluster_config, probes

st.title("Cluster setup")
st.caption(
    "Discover this cluster's settings and save them to `.cluster-config.yaml`. "
    "This is the GUI equivalent of the `/init-cluster` agent command."
)

existing = cluster_config.load_config()
active = cluster_config.active_config_path()
if active:
    st.success(f"Current config: `{active}`")
    with st.expander("Show current config"):
        st.json(existing)
else:
    st.warning("No cluster config found yet. Run discovery below.")

st.divider()

if st.button("🔍 Discover cluster settings", type="primary"):
    with st.spinner("Probing GPUs, SLURM, storage, network..."):
        st.session_state["discovery"] = probes.discover()

disc: probes.Discovery | None = st.session_state.get("discovery")
if not disc:
    st.stop()

st.subheader("Confirm settings")

# Seed from discovery, falling back to existing config values.
ex_slurm = existing.get("slurm", {})
ex_paths = existing.get("paths", {})
ex_cont = existing.get("containers", {})

col1, col2 = st.columns(2)
with col1:
    st.markdown("**GPU**")
    st.text_input("Architecture", value=disc.gpu_arch or "", key="cs_arch",
                  help=probes.arch_label(disc.gpu_arch) if disc.gpu_arch else "")
    st.number_input("GPUs per node", value=int(disc.gpu_count or 8),
                    min_value=1, max_value=64, key="cs_gpu_count")
    st.number_input("VRAM per GPU (GB)", value=int(disc.vram_gb or 0),
                    min_value=0, key="cs_vram")

    st.markdown("**SLURM**")
    part_opts = disc.partitions or ([ex_slurm.get("partition")] if ex_slurm.get("partition") else [])
    partition = st.selectbox(
        "Partition", options=part_opts or ["<none>"],
        index=0,
    )
    partition = st.text_input("Partition (override)", value=partition if partition != "<none>" else ex_slurm.get("partition", ""))
    acct_opts = disc.accounts or ([ex_slurm.get("account")] if ex_slurm.get("account") else [])
    account = st.selectbox("Account", options=acct_opts or ["<none>"], index=0)
    account = st.text_input("Account (override)", value=account if account != "<none>" else ex_slurm.get("account", ""))

with col2:
    st.markdown("**Container runtime**")
    rt_opts = disc.runtimes or ["apptainer"]
    if "apptainer" in rt_opts:
        rt_default = rt_opts.index("apptainer")
    else:
        rt_default = 0
    runtime = st.selectbox("Runtime", options=rt_opts, index=rt_default)
    if runtime != "apptainer":
        st.warning("v1 of Studio runs Apptainer only. Docker support is planned.")

    st.markdown("**Storage**")
    scratch_opts = disc.scratch_dirs or ([ex_paths.get("scratch")] if ex_paths.get("scratch") else [])
    scratch = st.selectbox("Scratch / shared root", options=scratch_opts or ["<none>"], index=0)
    scratch = st.text_input("Scratch (override)", value=scratch if scratch != "<none>" else ex_paths.get("scratch", ""),
                            help="Exported as AI4S_SHARED_DIR. Layout: <scratch>/images/ and <scratch>/models/<Model>/")

    st.markdown("**Network**")
    st.write(f"Internet: {'✅' if disc.internet else '❌'}")
    st.write(f"Interfaces: {', '.join(disc.net_ifaces) or '—'}")
    st.write(f"IB HCAs: {', '.join(disc.ib_hcas) or '—'}")

if disc.sifs:
    with st.expander(f"Discovered SIF images ({len(disc.sifs)})"):
        for s in disc.sifs:
            st.code(s, language=None)

st.divider()
location = st.radio(
    "Save location",
    options=[cluster_config.REPO_CONFIG, cluster_config.USER_CONFIG],
    format_func=lambda x: {
        cluster_config.REPO_CONFIG: f"Repo root ({cluster_config.repo_config_path()})",
        cluster_config.USER_CONFIG: f"User home ({cluster_config.user_config_path()})",
    }[x],
)

if st.button("💾 Save config", type="primary"):
    # Rebuild a Discovery-like object using the confirmed fields.
    disc.gpu_arch = st.session_state["cs_arch"]
    disc.gpu_count = int(st.session_state["cs_gpu_count"])
    disc.vram_gb = int(st.session_state["cs_vram"])
    cfg = probes.to_config(disc, partition=partition, account=account,
                           scratch=scratch, runtime=runtime)
    path = cluster_config.save_config(cfg, location=location)
    st.success(f"Saved to `{path}` (gitignored). Extra blocks preserved.")
    st.json(cfg)
