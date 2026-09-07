import asyncio
import json

from llm_regress.judge import Judge

from fakes import FakeAsyncOpenAI, make_response


JUDGE_JSON = json.dumps(
    {
        "tone": {"score": 4, "reason": "neutral"},
        "relevance": {"score": 5, "reason": "on point"},
        "grounding": {"score": 5, "reason": "no hallucination"},
        "summary_relevance": {"score": 4, "reason": "close to reference"},
    }
)


def _run(coro):
    return asyncio.run(coro)


def test_grade_parses_all_dimensions():
    client = FakeAsyncOpenAI(lambda **kwargs: make_response(JUDGE_JSON))
    judge = Judge(model="judge-model", base_url="https://example/v1", client=client)

    scores = _run(judge.grade("email", "actual summary", "reference summary"))

    dims = {s.dimension: s.score for s in scores}
    assert dims == {"tone": 4.0, "relevance": 5.0, "grounding": 5.0, "summary_relevance": 4.0}
    assert len(client.calls) == 1


def test_grade_omits_summary_relevance_when_no_reference():
    payload = {"tone": {"score": 3, "reason": "ok"}, "relevance": {"score": 2, "reason": "weak"}, "grounding": {"score": 4, "reason": "grounded"}}
    client = FakeAsyncOpenAI(lambda **kwargs: make_response(json.dumps(payload)))
    judge = Judge(model="m", base_url="https://example/v1", client=client)

    scores = _run(judge.grade("email", "actual", None))

    dims = {s.dimension for s in scores}
    assert dims == {"tone", "relevance", "grounding"}


def test_grade_returns_empty_on_bad_json():
    client = FakeAsyncOpenAI(lambda **kwargs: make_response("not json"))
    judge = Judge(model="m", base_url="https://example/v1", client=client)

    scores = _run(judge.grade("email", "actual", None))

    assert scores == []


def test_grade_returns_empty_on_non_object_json():
    client = FakeAsyncOpenAI(lambda **kwargs: make_response(json.dumps([1, 2, 3])))
    judge = Judge(model="m", base_url="https://example/v1", client=client)

    scores = _run(judge.grade("email", "actual", None))

    assert scores == []
