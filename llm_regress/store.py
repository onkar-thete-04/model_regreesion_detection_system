from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .evaluator import CaseResult
from .regress import Verdict


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_run(
        self,
        feature: str,
        model: str,
        commit: str,
        verdict: Verdict,
        results: list[CaseResult],
    ) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT,
                    model TEXT,
                    commit TEXT,
                    aggregate_score REAL,
                    baseline_score REAL,
                    passed INTEGER,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS case_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    case_id TEXT,
                    case_name TEXT,
                    dimension TEXT,
                    score REAL
                );
                """
            )
            created_at = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                "INSERT INTO runs (feature, model, commit, aggregate_score, baseline_score, passed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (feature, model, commit, verdict.aggregate_score, verdict.baseline_score, int(verdict.passed), created_at),
            )
            run_id = cur.lastrowid
            for r in results:
                for s in r.scores:
                    conn.execute(
                        "INSERT INTO case_scores (run_id, case_id, case_name, dimension, score) VALUES (?, ?, ?, ?, ?)",
                        (run_id, r.case_id, r.case_name, s.dimension, s.score),
                    )
            conn.commit()
        finally:
            conn.close()
