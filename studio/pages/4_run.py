"""Run — configure a recipe, preview the sbatch command, and submit."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import streamlit as st

from core import catalog, cluster_config, form_builder, jobs, submission
from core import sbatch_header as sbh

st.title("Run a model")

models = catalog.load_catalog()
slugs = [m.slug for m in models]

# Selection (seeded from Model detail page).
pre_model = st.session_state.get("run_model")
c1, c2 = st.columns(2)
with c1:
    slug = st.selectbox(
        "Model", slugs,
        index=slugs.index(pre_model) if pre_model in slugs else 0,
    )
model = catalog.get_model(slug)
runnable = [r for r in model.recipes if r.runnable]
if not runnable:
    st.warning("This model has no runnable recipes (agent-only).")
    st.stop()
tasks = [r.task for r in runnable]
with c2:
    pre_task = st.session_state.get("run_task")
    task = st.selectbox(
        "Recipe", tasks,
        index=tasks.index(pre_task) if pre_task in tasks else 0,
    )
recipe = model.recipe_for(task)

cfg = cluster_config.load_config()
rt = cluster_config.runtime(cfg)

# --- Apptainer-only guard ---------------------------------------------------
if rt and rt not in ("apptainer", "singularity"):
    st.error(
        f"Cluster runtime is `{rt}`. Studio v1 runs **Apptainer** only. "
        "Update the runtime on the Cluster setup page or use the CLI/agent."
    )
    st.stop()

# --- disclaimer gate --------------------------------------------------------
ack = True
if model.disclaimer:
    st.warning(model.disclaimer, icon="⚠️")
    ack = st.checkbox("I acknowledge the above and accept responsibility for this run.")

# --- ORBIT-2 data-mode branch (metadata-light special case) -----------------
env_overrides: dict[str, str] = {}
hidden: set[str] = set()
if slug == "ORBIT-2":
    mode = st.radio("Data mode", ["Synthetic (smoke test)", "Real data"],
                    horizontal=True)
    if mode.startswith("Synthetic"):
        env_overrides["ORBIT2_USE_SYNTHETIC"] = "1"
        hidden.update({"ORBIT2_CONFIG", "ORBIT2_CHECKPOINT"})
        st.caption("Synthetic mode generates a tiny dataset and auto-downloads a checkpoint from HF.")
    else:
        hidden.add("ORBIT2_USE_SYNTHETIC")

st.divider()
st.subheader("Scaling")

header = sbh.parse(model.abs_path / recipe.sbatch_script)
sc1, sc2, sc3, sc4 = st.columns(4)
nodes = sc1.number_input("Nodes", min_value=1, value=header.nodes)
ntpn = sc2.number_input("Tasks/node", min_value=1, value=header.ntasks_per_node)
gpus = sc3.number_input("GPUs/node", min_value=1, value=header.gpus)
walltime = sc4.text_input("Walltime", value=header.time or "00:30:00")

st.subheader("SLURM")
sl = cluster_config.slurm_defaults(cfg)
p1, p2, p3 = st.columns(3)
partition = p1.text_input("Partition", value=sl["partition"])
account = p2.text_input("Account", value=sl["account"])
qos = p3.text_input("QOS", value=sl["qos"])

# --- env var form -----------------------------------------------------------
st.subheader("Parameters")
scratch = cluster_config.scratch_dir(cfg)
assets = cluster_config.assets(cfg)

if scratch:
    env_overrides.setdefault("AI4S_SHARED_DIR", scratch)
    st.caption(f"`AI4S_SHARED_DIR` = `{scratch}` (from cluster config)")

fields = [f for f in form_builder.build_fields(model.env_vars) if f.name not in hidden]

values: dict[str, str] = {}
preflight_warnings: list[str] = []
for f in fields:
    label = f"{f.name}{' *' if f.required else ''}"
    if f.widget == "checkbox":
        checked = st.checkbox(label, value=(f.default == "1"), help=f.description)
        values[f.name] = "1" if checked else "0"
        default_val = f.default
    elif f.widget == "select":
        idx = f.options.index(f.default) if f.default in f.options else 0
        values[f.name] = st.selectbox(label, f.options, index=idx, help=f.description)
        default_val = f.default
    elif f.widget == "number":
        raw = st.text_input(label, value=f.default, help=f.description)
        values[f.name] = raw
        default_val = f.default
    else:
        val = st.text_input(label, value=f.default, help=f.description)
        values[f.name] = val
        default_val = f.default
        # pre-flight: required path that doesn't exist
        if f.is_path and val and not Path(val).expanduser().exists():
            preflight_warnings.append(f"`{f.name}` path does not exist: `{val}`")

    # Only emit if changed from default OR required-and-nonempty. Never emit "".
    v = values[f.name]
    if f.required and not v:
        pass  # will block submit below
    elif v and v != default_val:
        env_overrides[f.name] = v
    elif f.required and v:
        env_overrides[f.name] = v

# overlay-empty warning
for f in fields:
    if "OVERLAY" in f.name.upper() and not values.get(f.name):
        st.info(f"`{f.name}` empty → dependencies install at job start (~15 min).")

missing_required = [f.name for f in fields
                    if f.required and not values.get(f.name)]

for w in preflight_warnings:
    st.warning(w, icon="⚠️")

# --- build + preview + submit ----------------------------------------------
spec = submission.SubmitSpec(
    partition=partition, account=account, qos=qos,
    nodes=int(nodes), ntasks_per_node=int(ntpn), gpus=int(gpus),
    gpu_flag=header.gpu_flag, time=walltime, env=env_overrides,
)
built = submission.build_command(model, recipe, spec)

st.divider()
st.subheader("Preview (dry-run)")
st.code(submission.render_preview(built), language="bash")

can = submission.can_submit()
blockers = []
if not ack:
    blockers.append("acknowledge the disclaimer")
if missing_required:
    blockers.append("fill required fields: " + ", ".join(missing_required))
if not can:
    blockers.append("sbatch not available on this host (dry-run only)")
if not partition or not account:
    blockers.append("set SLURM partition and account")

if blockers:
    st.info("To submit: " + "; ".join(blockers) + ".")

submit_clicked = st.button(
    "🚀 Submit job", type="primary",
    disabled=bool(blockers) or st.session_state.get("_submitting", False),
)

if submit_clicked:
    st.session_state["_submitting"] = True
    with st.spinner("Submitting..."):
        result = submission.submit(built)
    st.session_state["_submitting"] = False
    if result.ok:
        st.success(f"Submitted batch job {result.jobid}")
        rec = {
            "jobid": result.jobid,
            "model": slug,
            "recipe": task,
            "runtime": rt or "apptainer",
            "submit_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cwd": built.cwd,
            "script": built.script_path,
            "outfile": result.outfile,
            "params": env_overrides,
            "overrides": {
                "nodes": int(nodes), "ntasks_per_node": int(ntpn),
                "gpus": int(gpus), "partition": partition, "account": account,
            },
            "last_state": "PENDING",
        }
        if model.disclaimer:
            rec["disclaimer_ack"] = {
                "hash": hashlib.sha256(model.disclaimer.encode()).hexdigest()[:16],
                "ts": rec["submit_time"],
            }
        jobs.add_job(rec)
        st.page_link("pages/5_jobs.py", label="→ View in Jobs", icon="📋")
    else:
        st.error("Submission failed.")
        if result.stderr:
            st.code(result.stderr)
