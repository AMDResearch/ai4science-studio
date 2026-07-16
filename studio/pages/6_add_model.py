"""Add model — scaffold a new model folder and register it in models.yaml."""
from __future__ import annotations

import streamlit as st

from core import scaffold

st.title("Add a model")
st.caption(
    "Scaffold a new model into the repo from `_template/`. This writes files "
    "locally; review with `git diff` and open a PR yourself."
)

hf_id = st.text_input("Hugging Face id", placeholder="org/model  (or N/A)")
default_slug = scaffold.slug_from_hf(hf_id) if hf_id else ""
col1, col2 = st.columns(2)
with col1:
    slug = st.text_input("Slug (folder name)", value=default_slug,
                         help="org/model → org__model, or a public name.")
    domain = st.selectbox("Domain", scaffold.DOMAINS)
    license_ = st.text_input("License", placeholder="Apache-2.0 / MIT / link")
with col2:
    task = st.text_input("Task", placeholder="one-line description")
    upstream = st.text_input("Upstream code URL", placeholder="https://github.com/...")
    paper = st.text_input("Paper URL", placeholder="https://arxiv.org/abs/...")

container = st.text_input(
    "Container image (optional)",
    placeholder="rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0",
)

nm = scaffold.NewModel(
    slug=slug, domain=domain, hf_id=hf_id, license=license_, task=task,
    upstream_code=upstream, paper=paper, container_image=container,
)

errs = scaffold.validate(nm) if slug else ["fill in the fields above"]
if errs:
    st.info("Before creating: " + "; ".join(errs))

if st.button("Create model", type="primary", disabled=bool(errs)):
    try:
        dest = scaffold.create(nm)
        st.success(f"Created `{dest}` and registered in models.yaml.")
        st.markdown(
            "**Next steps:**\n"
            "1. Fill in `model.yaml` (recipes, env_vars, validated_hardware).\n"
            "2. Add recipe docs under `recipes/` and scripts under `examples/`.\n"
            "3. Add an `ACKNOWLEDGEMENTS.md` entry.\n"
            "4. Review with `git diff` and open a PR."
        )
        st.cache_data.clear()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not create model: {exc}")
