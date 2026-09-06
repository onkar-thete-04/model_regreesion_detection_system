import asyncio

from llm_regress.judge import DimensionScore
from llm_regress.scoring import CaseResult, Scorer, category_matches
from llm_regress.types import Category, ClassifierOutput, ExpectedOutput, GoldenCase, RawResult, TokenUsage


def make_case(case_id, text, category=Category.BILLING, category_any=None, summary="ref"):
    return GoldenCase(
        id=case_id,
        input=text,
        expected=ExpectedOutput(category=category, category_any=category_any, summary=summary),
        expected_difficulty=1,
        notes="n",
    )


def _raw(case_id, category, text="email"):
    return RawResult(
        case_id=case_id,
        input=text,
        output=ClassifierOutput(category=category, summary="actual"),
        latency_ms=10.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class FakeJudge:
    def __init__(self, scores=None):
        self.scores = scores or []

    async def grade(self, case_input, actual_summary, expected_summary):
        return self.scores


def _run(coro):
    return asyncio.run(coro)


def test_category_matches_exact():
    expected = ExpectedOutput(category=Category.BILLING, summary="ref")
    assert category_matches(ClassifierOutput(category=Category.BILLING, summary="s"), expected) is True
    assert category_matches(ClassifierOutput(category=Category.TECHNICAL, summary="s"), expected) is False


def test_category_matches_category_any():
    expected = ExpectedOutput(category=Category.BILLING, category_any=[Category.BILLING, Category.ACCOUNT], summary="ref")
    assert category_matches(ClassifierOutput(category=Category.ACCOUNT, summary="s"), expected) is True


def test_category_matches_none_actual():
    assert category_matches(None, ExpectedOutput(category=Category.BILLING, summary="ref")) is False


def test_score_many_applies_judge_and_match():
    case = make_case("c-1", "email", category=Category.BILLING, summary="ref")
    judge = FakeJudge([DimensionScore(dimension="tone", score=4.0, reason="r")])
    scorer = Scorer(judge)

    results = _run(scorer.score_many([_raw("c-1", Category.BILLING)], [case]))

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, CaseResult)
    assert r.category_match is True
    assert r.scores[0].dimension == "tone"
    assert r.average == 4.0


def test_score_many_error_case_skips_judge():
    case = make_case("c-1", "email", category=Category.BILLING)
    judge = FakeJudge([DimensionScore(dimension="tone", score=1.0, reason="r")])
    raw = RawResult(case_id="c-1", input="email", error="boom")

    results = _run(Scorer(judge).score_many([raw], [case]))

    assert results[0].category_match is False
    assert results[0].scores == []
