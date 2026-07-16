"""Jobs — monitor submitted jobs, poll SLURM state, tail logs."""
from __future__ import annotations

import streamlit as st

from core import jobs

st.title("Jobs")

records = jobs.list_jobs()
if not records:
    st.info("No jobs submitted yet. Launch one from the **Run** page.")
    st.stop()

top = st.columns([1, 1, 4])
if top[0].button("🔄 Refresh"):
    st.rerun()
auto = top[1].toggle("Auto-refresh", value=False)

states = jobs.poll_states()

_STATE_ICON = {
    "PENDING": "🟡", "RUNNING": "🟢", "COMPLETING": "🟢",
    "COMPLETED": "✅", "GONE": "✅", "FAILED": "🔴", "OUT_OF_MEMORY": "🔴",
    "NODE_FAIL": "🔴", "CANCELLED": "⚪", "TIMEOUT": "🔴", "UNKNOWN": "❔",
}

# Human-readable one-liner for a terminal state.
_STATE_HELP = {
    "COMPLETED": "Finished successfully.",
    "FAILED": "The job exited with an error — check the log below.",
    "OUT_OF_MEMORY": "Killed: out of memory.",
    "TIMEOUT": "Killed: exceeded the walltime limit.",
    "CANCELLED": "Cancelled.",
    "NODE_FAIL": "A compute node failed.",
    "RUNNING": "Currently running.",
    "PENDING": "Queued, waiting for resources.",
}

for rec in records:
    jid = rec.get("jobid", "?")
    state = states.get(jid, rec.get("last_state", "UNKNOWN"))
    icon = _STATE_ICON.get(state, "❔")
    code = rec.get("exit_code", "")
    code_str = f"  (exit {code})" if code and state != "COMPLETED" else ""
    title = f"{icon} {jid} — {rec.get('model')} / {rec.get('recipe')} — {state}{code_str}"
    with st.expander(title, expanded=(state in ("RUNNING", "PENDING"))):
        help_line = _STATE_HELP.get(state)
        if help_line:
            (st.success if state == "COMPLETED" else
             st.error if _STATE_ICON.get(state) == "🔴" else st.info)(help_line)

        meta = st.columns(4)
        meta[0].metric("Nodes", rec.get("overrides", {}).get("nodes", "?"))
        meta[1].metric("GPUs/node", rec.get("overrides", {}).get("gpus", "?"))
        meta[2].write(f"**Submitted:** {rec.get('submit_time', '?')}")
        meta[3].write(f"**Partition:** {rec.get('overrides', {}).get('partition', '?')}")

        outfile = rec.get("outfile", "")

        # Headline results (PSNR/SSIM/loss/throughput) parsed from the log.
        metrics = jobs.extract_metrics(outfile) if outfile else {}
        if metrics:
            st.markdown("**Results**")
            mcols = st.columns(len(metrics))
            for col, (label, val) in zip(mcols, metrics.items()):
                col.metric(label, val)
            if rec.get("model") == "ORBIT-2" and rec.get("params", {}).get(
                    "ORBIT2_USE_SYNTHETIC") == "1":
                st.caption(
                    "Synthetic-data smoke test — these metrics confirm the pipeline "
                    "runs end-to-end, not model accuracy. Use real ERA5/PRISM data "
                    "for meaningful PSNR/SSIM."
                )

        if rec.get("params"):
            with st.popover("Parameters"):
                st.json(rec["params"])

        st.caption(f"Log: `{outfile or '(none)'}`")
        log = jobs.tail_log(outfile) if outfile else ""
        if log:
            with st.expander("Full log", expanded=not metrics):
                st.code(log, language=None)
        elif state in ("PENDING",):
            st.caption("Waiting for job to start...")
        else:
            st.caption("No log output yet.")

        actions = st.columns(3)
        if state in ("PENDING", "RUNNING", "COMPLETING"):
            if actions[0].button("🛑 Cancel", key=f"cancel_{jid}"):
                if jobs.cancel(jid):
                    st.success(f"Cancelled {jid}")
                    st.rerun()
                else:
                    st.error("scancel unavailable")
        if actions[1].button("🔁 Re-run", key=f"rerun_{jid}"):
            st.session_state["run_model"] = rec.get("model")
            st.session_state["run_task"] = rec.get("recipe")
            st.switch_page("pages/4_run.py")
        if actions[2].button("🗑 Remove", key=f"rm_{jid}"):
            jobs.remove_job(jid)
            st.rerun()

if auto:
    import time

    time.sleep(10)
    st.rerun()
