from llm_regress.types import Category, ClassifierOutput, RawResult, TokenUsage


def test_token_usage_model():
    usage = TokenUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20)
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 20


def test_raw_result_defaults():
    result = RawResult(case_id="c-abc", input="email text")
    assert result.case_id == "c-abc"
    assert result.output is None
    assert result.latency_ms == 0.0
    assert result.token_usage is None
    assert result.error is None


def test_raw_result_full():
    result = RawResult(
        case_id="c-1",
        input="email",
        output=ClassifierOutput(category=Category.BILLING, summary="a summary"),
        latency_ms=123.4,
        token_usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )
    assert result.output.category == Category.BILLING
    assert result.latency_ms == 123.4
    assert result.token_usage.total_tokens == 7
