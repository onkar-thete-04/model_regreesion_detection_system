import asyncio
import json

from llm_regress.runner import Runner
from llm_regress.types import Category, ExpectedOutput, GoldenCase, PromptConfig

from fakes import FakeAsyncOpenAI, make_response


def make_case(case_id, text, category=Category.BILLING):
    return GoldenCase(
        id=case_id,
        input=text,
        expected=ExpectedOutput(category=category, summary="ref"),
        expected_difficulty=1,
        notes="n",
    )


def make_prompt():
    return PromptConfig(
        version_id="v1",
        timestamp="2026-09-06T00:00:00Z",
        system_prompt="Classify the email.",
        few_shot_examples=[],
    )


def _run(coro):
    return asyncio.run(coro)


def test_run_collects_output_latency_and_tokens():
    def respond(**kwargs):
        return make_response(
            json.dumps({"category": "billing", "summary": "a billing issue"}),
            prompt_tokens=20,
            completion_tokens=4,
        )

    client = FakeAsyncOpenAI(respond)
    runner = Runner(make_prompt(), "m", "https://example/v1", client=client, max_concurrency=2)
    cases = [make_case("c-1", "I was overcharged."), make_case("c-2", "Refund please.")]

    results = _run(runner.run(cases))

    assert len(results) == 2
    for r in results:
        assert r.error is None
        assert r.output.category == Category.BILLING
        assert r.latency_ms >= 0
        assert r.token_usage.total_tokens == 24


def test_run_isolates_a_failing_case():
    call_count = {"n": 0}

    def respond(**kwargs):
        call_count["n"] += 1
        if "always-fail" in kwargs["messages"][-1]["content"]:
            raise RuntimeError("boom")
        return make_response(json.dumps({"category": "account", "summary": "s"}))

    client = FakeAsyncOpenAI(respond)
    runner = Runner(make_prompt(), "m", "https://example/v1", client=client, max_retries=2)
    cases = [make_case("c-ok", "fine"), make_case("c-bad", "always-fail")]

    results = _run(runner.run(cases))

    by_id = {r.case_id: r for r in results}
    assert by_id["c-ok"].error is None
    assert by_id["c-bad"].error is not None
    assert by_id["c-bad"].output is None


def test_run_retries_then_succeeds():
    attempts = {"n": 0}

    def respond(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return make_response(json.dumps({"category": "general", "summary": "s"}))

    client = FakeAsyncOpenAI(respond)
    runner = Runner(make_prompt(), "m", "https://example/v1", client=client, max_retries=2)

    results = _run(runner.run([make_case("c-1", "hi")]))

    assert results[0].error is None
    assert attempts["n"] == 3
