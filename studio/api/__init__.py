"""FastAPI backend for AI4Science Studio (React alternative to the Streamlit UI).

This package is a thin JSON wrapper around the UI-agnostic logic in
``studio/core/``. It never reimplements sbatch/squeue/probe/suggestion logic —
it imports and reuses ``core`` verbatim, exactly like the Streamlit pages do.
"""
