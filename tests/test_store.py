import sqlite3

from llm_regress.diff import RunSummary
from llm_regress.judge import DimensionScore
from llm_regress.scoring import CaseResult
from llm_regress.store import Store
from llm_regress.types import Category, ClassifierOutput, ExpectedOutput, TokenUsage


def make_result(case_id, category, matched):
    return CaseResult(
        case_id=case_id,
        input="email",
        expected=ExpectedOutput(category=category, summary="ref"),
        actual=ClassifierOutput(category=category, summary="s") if matched else ClassifierOutput(category=Category.GENERAL, summary="s"),
        category_match=matched,
        scores=[DimensionScore(dimension="tone", score=4.0, reason="r")],
        latency_ms=12.5,
        token_usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def test_load_previous_run_empty_db(tmp_path):
    store = Store(tmp_path / "eval.db")
    assert store.load_previous_run("email_classifier", "model-a") is None


def test_record_and_load_previous_run(tmp_path):
    store = Store(tmp_path / "eval.db")
    store.record_results(
        "email_classifier",
        "model-a",
        "abc123",
        [
            make_result("c-1", Category.BILLING, True),
            make_result("c-2", Category.BILLING, False),
            make_result("c-3", Category.TECHNICAL, True),
        ],
    )

    summary = store.load_previous_run("email_classifier", "model-a")

    assert isinstance(summary, RunSummary)
    assert summary.total == 3
    assert summary.passed == 2
    assert summary.case_pass == {"c-1": True, "c-2": False, "c-3": True}
    assert summary.per_category_accuracy["billing"] == 0.5
    assert summary.per_category_accuracy["technical"] == 1.0


def test_load_previous_run_picks_most_recent(tmp_path):
    store = Store(tmp_path / "eval.db")
    store.record_results("email_classifier", "model-a", "one", [make_result("c-1", Category.BILLING, True)])
    store.record_results("email_classifier", "model-a", "two", [make_result("c-1", Category.BILLING, False)])

    summary = store.load_previous_run("email_classifier", "model-a")

    assert summary.case_pass["c-1"] is False
