"""Catalog — browse and filter all models from models.yaml."""
from __future__ import annotations

import streamlit as st

from core import catalog

st.title("Model catalog")

try:
    models = catalog.load_catalog()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load catalog: {exc}")
    st.stop()

# --- filters ----------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    query = st.text_input("Search", placeholder="name, task, hf id...").strip().lower()
    sel_domains = st.multiselect("Domain", catalog.domains())
    sel_tasks = st.multiselect("Task", catalog.all_tasks())
    sel_licenses = st.multiselect("License", catalog.all_licenses())


def _matches(m: catalog.Model) -> bool:
    if sel_domains and m.domain not in sel_domains:
        return False
    if sel_tasks and not (set(sel_tasks) & set(m.tasks_available)):
        return False
    if sel_licenses and m.license not in sel_licenses:
        return False
    if query:
        hay = " ".join(
            [m.name, m.slug, m.task, m.hf_id, m.domain, " ".join(m.tasks_available)]
        ).lower()
        if query not in hay:
            return False
    return True


filtered = [m for m in models if _matches(m)]
st.caption(f"{len(filtered)} of {len(models)} models")

# --- card grid --------------------------------------------------------------
COLS = 3
rows = (len(filtered) + COLS - 1) // COLS
for r in range(rows):
    cols = st.columns(COLS)
    for c in range(COLS):
        idx = r * COLS + c
        if idx >= len(filtered):
            continue
        m = filtered[idx]
        with cols[c].container(border=True):
            st.subheader(m.name)
            st.caption(f"{m.domain}  |  {m.license or 'license: n/a'}")
            st.write(m.task or "_no description_")
            if m.tasks_available:
                st.write(
                    " ".join(f"`{t}`" for t in m.tasks_available)
                )
            if m.disclaimer:
                st.warning("Usage disclaimer applies", icon="⚠️")
            if st.button("View details", key=f"view_{m.slug}", width="stretch"):
                st.session_state["selected_model"] = m.slug
                st.switch_page("pages/2_model_detail.py")
