from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import streamlit as st

DB = Path(os.environ.get("SQLITE_PATH", "results/eval.db"))


def load_runs() -> list[dict]:
    if not DB.exists():
        return []
    conn = sqlite3.connect(DB)
    try:
        rows = conn.execute(
            "SELECT feature, model, commit, aggregate_score, baseline_score, passed, created_at "
            "FROM runs ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "feature": r[0],
            "model": r[1],
            "commit": r[2],
            "aggregate_score": r[3],
            "baseline_score": r[4],
            "passed": bool(r[5]),
            "created_at": r[6],
        }
        for r in rows
    ]


st.title("LLM Regression Dashboard")
runs = load_runs()
if not runs:
    st.info("No runs recorded yet.")
else:
    st.dataframe(runs)
