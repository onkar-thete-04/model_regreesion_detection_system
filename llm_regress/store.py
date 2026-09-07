from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .diff import RunSummary
from .scoring import CaseResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature TEXT,
    model TEXT,
    "commit" TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS case_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    case_id TEXT,
    predicted_category TEXT,
    expected_category TEXT,
    category_match INTEGER,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
);
CREATE TABLE IF NOT EXISTS case_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    case_id TEXT,
    dimension TEXT,
    score REAL,
    reason TEXT
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.executescript(_SCHEMA)
        return conn

    def record_results(self, feature: str, model: str, commit: str, results: list[CaseResult]) -> int:
        conn = self._connect()
        try:
            created_at = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                "INSERT INTO runs (feature, model, \"commit\", created_at) VALUES (?, ?, ?, ?)",
                (feature, model, commit, created_at),
            )
            run_id = cursor.lastrowid
            for result in results:
                predicted = result.actual.category.value if result.actual is not None else None
                token_usage = result.token_usage
                conn.execute(
                    "INSERT INTO case_runs "
                    "(run_id, case_id, predicted_category, expected_category, category_match, latency_ms, prompt_tokens, completion_tokens) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        result.case_id,
                        predicted,
                        result.expected.category.value,
                        int(result.category_match),
                        result.latency_ms,
                        token_usage.prompt_tokens if token_usage else None,
                        token_usage.completion_tokens if token_usage else None,
                    ),
                )
                for score in result.scores:
                    conn.execute(
                        "INSERT INTO case_scores (run_id, case_id, dimension, score, reason) VALUES (?, ?, ?, ?, ?)",
                        (run_id, result.case_id, score.dimension, score.score, score.reason),
                    )
            conn.commit()
            return run_id
        finally:
            conn.close()

    def load_previous_run(self, feature: str, model: str) -> RunSummary | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM runs WHERE feature = ? AND model = ? ORDER BY id DESC LIMIT 1",
                (feature, model),
            ).fetchone()
            if row is None:
                return None
            run_id = row[0]
            summary = RunSummary(run_id=run_id)
            category_totals: dict[str, int] = {}
            category_passed: dict[str, int] = {}
            for case_id, matched, expected_category in conn.execute(
                "SELECT case_id, category_match, expected_category FROM case_runs WHERE run_id = ?",
                (run_id,),
            ).fetchall():
                summary.total += 1
                if matched:
                    summary.passed += 1
                summary.case_pass[case_id] = bool(matched)
                if expected_category is not None:
                    category_totals[expected_category] = category_totals.get(expected_category, 0) + 1
                    if matched:
                        category_passed[expected_category] = category_passed.get(expected_category, 0) + 1
            for category, total in category_totals.items():
                summary.per_category_accuracy[category] = category_passed.get(category, 0) / total
            return summary
        finally:
            conn.close()
