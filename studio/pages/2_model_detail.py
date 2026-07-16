"""Model detail — render one model.yaml and its recipes."""
from __future__ import annotations

import streamlit as st

from core import catalog

st.title("Model detail")

models = catalog.load_catalog()
slug = st.session_state.get("selected_model")

# Allow direct navigation with a picker if nothing selected yet.
slugs = [m.slug for m in models]
if slug not in slugs:
    slug = st.selectbox("Select a model", slugs)
else:
    slug = st.selectbox("Select a model", slugs, index=slugs.index(slug))
st.session_state["selected_model"] = slug

m = catalog.get_model(slug)
if not m:
    st.stop()

st.header(m.name)
st.caption(f"{m.domain}  |  slug: `{m.slug}`")

from core import validation

_schema_errs = validation.validate_model_yaml(m.raw)
if _schema_errs:
    with st.expander(f"⚠️ model.yaml schema notes ({len(_schema_errs)})"):
        for e in _schema_errs:
            st.caption(e)

if m.disclaimer:
    st.warning(m.disclaimer, icon="⚠️")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Task:** {m.task or '—'}")
    st.markdown(f"**License:** {m.license or '—'}")
    if m.hf_id and m.hf_id != "N/A":
        st.markdown(f"**Hugging Face:** [`{m.hf_id}`](https://huggingface.co/{m.hf_id})")
    else:
        st.markdown(f"**Weights:** {m.weight_source or 'see model card'}")
with col2:
    if m.upstream_code:
        st.markdown(f"**Code:** [{m.upstream_code}]({m.upstream_code})")
    if m.paper:
        st.markdown(f"**Paper:** [{m.paper}]({m.paper})")
    if m.validated_hardware:
        st.markdown(f"**Validated hardware:** {', '.join(m.validated_hardware)}")
    if m.vram_gb:
        st.markdown(f"**VRAM:** {m.vram_gb} GB")

if m.container_image:
    st.markdown("**Container image(s):**")
    for img in m.container_image:
        st.code(img, language=None)

if m.data_source:
    st.markdown(f"**Data source:** {m.data_source}")

if m.model_variants:
    st.markdown("**Variants:**")
    for v in m.model_variants:
        name = v.get("name", "?")
        cond = v.get("conditioning", "")
        st.markdown(f"- `{name}` — {cond}")

st.divider()
st.subheader("Recipes")

if not m.recipes:
    st.info("No recipes defined for this model.")

for r in m.recipes:
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{r.task}** — {r.description or ''}")
            if r.recipe_path:
                st.caption(r.recipe_path)
        with right:
            if r.runnable:
                if st.button("Run", key=f"run_{m.slug}_{r.task}", width="stretch"):
                    st.session_state["run_model"] = m.slug
                    st.session_state["run_task"] = r.task
                    st.switch_page("pages/4_run.py")
            else:
                st.button(
                    "Agent-only",
                    key=f"na_{m.slug}_{r.task}",
                    disabled=True,
                    help="No single sbatch script; run via the CLI/agent.",
                    width="stretch",
                )
