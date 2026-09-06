# Phase 3 Evaluation Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the evaluation engine — async batched test runner, multi-dimensional scoring (category match + LLM judge + latency + tokens), and run-over-run diff backed by SQLite.

**Architecture:** Layered, one responsibility per file. `runner.py` executes golden cases concurrently via `AsyncOpenAI` and returns raw outputs. `scoring.py` turns raw outputs into scored `CaseResult`s using a direct-OpenAI `judge.py` (no DeepEval). `diff.py` compares a current run against the previous run loaded from `store.py`. `evaluator.py` composes runner + scorer.

**Tech Stack:** Python 3.11+, openai (AsyncOpenAI), pydantic v2, pytest. No new runtime deps; no pytest-asyncio (async tested via `asyncio.run`).

## Global Constraints

- Python 3.11+; only `openai`, `pydantic`, `pyyaml` (already installed) — do NOT add `deepeval`, `pytest-asyncio`, or any other dependency.
- All LLM calls go through injectable `client` arguments; tests inject fakes, never hit the network.
- Per-case isolation: one failing case must not crash a run; record `error` on the `RawResult`/`CaseResult`.
- Regression thresholds `absolute_floor: 3.5`, `max_drop: 0.5` in `config.yaml` are NOT changed this phase.
- Windows/PowerShell. Run tests: `.venv\Scripts\python.exe -m pytest -q`.
- Commit after each task with the message shown in Step 5.
- Keep `average` property on `CaseResult` (used by `regress.py` in a later phase).

---

### Task 1: TokenUsage + RawResult models

**Files:**
- Modify: `llm_regress/types.py` (append new models)
- Test: `tests/test_eval_types.py` (create)

**Interfaces:**
- Produces: `TokenUsage` (pydantic, ints `prompt_tokens`, `completion_tokens`, `total_tokens`), `RawResult` (pydantic: `case_id: str`, `input: str`, `output: ClassifierOutput | None`, `latency_ms: float`, `token_usage: TokenUsage | None`, `error: str | None`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_types.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_types.py -v`
Expected: FAIL — `ImportError: cannot import name 'RawResult'` (or `'TokenUsage'`).

- [ ] **Step 3: Write minimal implementation**

Append to `llm_regress/types.py`:

```python
class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RawResult(BaseModel):
    case_id: str
    input: str
    output: ClassifierOutput | None = None
    latency_ms: float = 0.0
    token_usage: TokenUsage | None = None
    error: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_types.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/types.py tests/test_eval_types.py
git commit -m "feat: add TokenUsage and RawResult models"
```

---

### Task 2: Judge (direct OpenAI, async)

**Files:**
- Modify: `llm_regress/judge.py` (rewrite; remove DeepEval import)
- Test: `tests/fakes.py` (create), `tests/test_judge.py` (create)

**Interfaces:**
- Produces: `Judge(model, base_url, api_key=None, *, client=None, timeout_seconds=60.0)` with `async grade(case_input: str, actual_summary: str, expected_summary: str | None) -> list[DimensionScore]`.
- Keeps existing `DimensionScore(dimension, score, reason)` dataclass.
- Produces `tests/fakes.py` helper: `FakeAsyncOpenAI(respond)` (records `.calls`, async `chat.completions.create`) and `make_response(content, ...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/fakes.py`:

```python
from types import SimpleNamespace


class FakeAsyncOpenAI:
    """Minimal AsyncOpenAI stand-in with a recording chat.completions.create."""

    def __init__(self, respond):
        self._respond = respond
        self.calls = []

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                return self._outer._respond(**kwargs)

        self.chat = SimpleNamespace(completions=_Completions(self))


def make_response(content, prompt_tokens=10, completion_tokens=5):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)
```

Create `tests/test_judge.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_judge.py -v`
Expected: FAIL — `judge.grade` raises `NotImplementedError` (current stub), or import error for `fakes`.

- [ ] **Step 3: Write minimal implementation**

Replace `llm_regress/judge.py` entirely:

```python
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class DimensionScore:
    dimension: str
    score: float
    reason: str


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of email-classifier summaries. Rate the ACTUAL summary of a customer support email on each dimension using a 1-5 integer scale.

- tone: the summary uses a neutral, professional support tone. 5 = perfect professional tone, 1 = rude or unprofessional.
- relevance: the summary captures the email's core request or issue. 5 = captures it precisely, 1 = misses or misstates it.
- grounding: every claim in the summary is supported by the email text. 5 = fully grounded, 1 = hallucinates or fabricates.
- summary_relevance: the summary matches the reference summary's meaning. Rate this ONLY if a reference summary is provided.

Respond with ONLY a JSON object of this shape:
{"tone": {"score": <int 1-5>, "reason": "<short>"}, "relevance": {"score": <int 1-5>, "reason": "<short>"}, "grounding": {"score": <int 1-5>, "reason": "<short>"}, "summary_relevance": {"score": <int 1-5>, "reason": "<short>"}}
If no reference summary is provided, omit the summary_relevance key entirely."""

_JUDGE_DIMENSIONS = ["tone", "relevance", "grounding", "summary_relevance"]


class Judge:
    def __init__(self, model: str, base_url: str, api_key: str | None = None, *, client=None, timeout_seconds: float = 60.0):
        self.model = model
        self._client = client or AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    async def grade(self, case_input: str, actual_summary: str, expected_summary: str | None) -> list[DimensionScore]:
        user_payload = {"email": case_input, "actual_summary": actual_summary}
        if expected_summary is not None:
            user_payload["reference_summary"] = expected_summary

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        content = response.choices[0].message.content
        try:
            raw = json.loads(content)
        except (TypeError, ValueError):
            return []

        scores: list[DimensionScore] = []
        for dim in _JUDGE_DIMENSIONS:
            entry = raw.get(dim)
            if entry is None:
                continue
            try:
                scores.append(DimensionScore(dimension=dim, score=float(entry["score"]), reason=str(entry.get("reason", ""))))
            except (KeyError, TypeError, ValueError):
                continue
        return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_judge.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/judge.py tests/fakes.py tests/test_judge.py
git commit -m "feat: direct OpenAI judge with rubric scoring"
```

---

### Task 3: Runner (async batched execution)

**Files:**
- Create: `llm_regress/runner.py`
- Test: `tests/test_runner.py` (create)

**Interfaces:**
- Consumes: `TokenUsage`, `RawResult`, `GoldenCase`, `PromptConfig` (types), `_build_messages` from `llm_regress/features/email_classifier.py`, `FakeAsyncOpenAI`/`make_response` from `tests/fakes.py`.
- Produces: `Runner(prompt, model, base_url, api_key=None, *, max_concurrency=8, timeout_seconds=60.0, max_retries=2, client=None)` with `async run(cases: list[GoldenCase]) -> list[RawResult]` and `run_sync(cases) -> list[RawResult]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_regress.runner'`.

- [ ] **Step 3: Write minimal implementation**

Create `llm_regress/runner.py`:

```python
from __future__ import annotations

import asyncio
from time import perf_counter

from openai import AsyncOpenAI

from .features.email_classifier import _build_messages
from .types import ClassifierOutput, GoldenCase, PromptConfig, RawResult, TokenUsage


class Runner:
    def __init__(
        self,
        prompt: PromptConfig,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        max_concurrency: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client=None,
    ):
        self.prompt = prompt
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self._client = client or AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    async def run(self, cases: list[GoldenCase]) -> list[RawResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._run_one(semaphore, case) for case in cases]
        return await asyncio.gather(*tasks)

    def run_sync(self, cases: list[GoldenCase]) -> list[RawResult]:
        return asyncio.run(self.run(cases))

    async def _run_one(self, semaphore: asyncio.Semaphore, case: GoldenCase) -> RawResult:
        async with semaphore:
            last_error: Exception | None = None
            for _ in range(self.max_retries + 1):
                try:
                    return await self._attempt(case)
                except Exception as exc:  # noqa: BLE001 - per-case isolation
                    last_error = exc
            return RawResult(case_id=case.id, input=case.input, error=f"{type(last_error).__name__}: {last_error}")

    async def _attempt(self, case: GoldenCase) -> RawResult:
        messages = _build_messages(case.input, self.prompt)
        start = perf_counter()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        latency_ms = (perf_counter() - start) * 1000.0
        content = response.choices[0].message.content
        output = ClassifierOutput.model_validate_json(content)

        usage = response.usage
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )

        return RawResult(
            case_id=case.id,
            input=case.input,
            output=output,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_runner.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/runner.py tests/test_runner.py
git commit -m "feat: async batched test runner"
```

---

### Task 4: Scorer + CaseResult

**Files:**
- Create: `llm_regress/scoring.py`
- Test: `tests/test_scoring.py` (create)

**Interfaces:**
- Consumes: `DimensionScore`, `Judge` (judge), `RawResult`, `GoldenCase`, `ExpectedOutput`, `ClassifierOutput`, `TokenUsage` (types), `tests/fakes.py`.
- Produces: `CaseResult` dataclass (`case_id`, `input`, `expected`, `actual: ClassifierOutput | None`, `category_match: bool`, `scores: list[DimensionScore]`, `latency_ms: float`, `token_usage: TokenUsage | None`, `error: str | None`, property `average`). `category_matches(actual, expected) -> bool`. `Scorer(judge, *, max_concurrency=8)` with `async score_many(raws: list[RawResult], cases: list[GoldenCase]) -> list[CaseResult]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_regress.scoring'`.

- [ ] **Step 3: Write minimal implementation**

Create `llm_regress/scoring.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .judge import DimensionScore, Judge
from .types import ClassifierOutput, ExpectedOutput, GoldenCase, RawResult, TokenUsage


@dataclass
class CaseResult:
    case_id: str
    input: str
    expected: ExpectedOutput
    actual: ClassifierOutput | None
    category_match: bool
    scores: list[DimensionScore] = field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: TokenUsage | None = None
    error: str | None = None

    @property
    def average(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


def category_matches(actual: ClassifierOutput | None, expected: ExpectedOutput) -> bool:
    if actual is None:
        return False
    if actual.category == expected.category:
        return True
    if expected.category_any and actual.category in expected.category_any:
        return True
    return False


class Scorer:
    def __init__(self, judge: Judge, *, max_concurrency: int = 8):
        self.judge = judge
        self.max_concurrency = max_concurrency

    async def score_many(self, raws: list[RawResult], cases: list[GoldenCase]) -> list[CaseResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._score_one(semaphore, raw, case) for raw, case in zip(raws, cases)]
        return await asyncio.gather(*tasks)

    async def _score_one(self, semaphore: asyncio.Semaphore, raw: RawResult, case: GoldenCase) -> CaseResult:
        matched = category_matches(raw.output, case.expected)
        scores: list[DimensionScore] = []
        if raw.output is not None and raw.error is None:
            async with semaphore:
                scores = await self.judge.grade(case.input, raw.output.summary, case.expected.summary)
        return CaseResult(
            case_id=case.id,
            input=case.input,
            expected=case.expected,
            actual=raw.output,
            category_match=matched,
            scores=scores,
            latency_ms=raw.latency_ms,
            token_usage=raw.token_usage,
            error=raw.error,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_scoring.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/scoring.py tests/test_scoring.py
git commit -m "feat: case scoring with category match and judge"
```

---

### Task 5: Diff logic

**Files:**
- Create: `llm_regress/diff.py`
- Test: `tests/test_diff.py` (create)

**Interfaces:**
- Consumes: `CaseResult` (scoring).
- Produces: `RunSummary` dataclass (`run_id: int`, `total: int`, `passed: int`, `per_category_accuracy: dict[str, float]`, `case_pass: dict[str, bool]`, property `pass_rate`). `DiffReport` dataclass (`previous: RunSummary | None`, `current: RunSummary`, `overall_pass_rate_delta: float | None`, `per_category_accuracy_delta: dict[str, float]`, `regressions: list[str]`, `improvements: list[str]`). `summarize(run_id: int, results: list[CaseResult]) -> RunSummary`. `diff(current: list[CaseResult], previous: RunSummary | None) -> DiffReport`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diff.py`:

```python
from llm_regress.diff import diff, summarize
from llm_regress.scoring import CaseResult
from llm_regress.types import Category, ClassifierOutput, ExpectedOutput


def make_result(case_id, category, matched):
    return CaseResult(
        case_id=case_id,
        input="email",
        expected=ExpectedOutput(category=category, summary="ref"),
        actual=ClassifierOutput(category=Category.BILLING, summary="s"),
        category_match=matched,
    )


def test_summarize():
    results = [
        make_result("c-1", Category.BILLING, True),
        make_result("c-2", Category.BILLING, False),
        make_result("c-3", Category.TECHNICAL, True),
    ]
    summary = summarize(7, results)

    assert summary.run_id == 7
    assert summary.total == 3
    assert summary.passed == 2
    assert summary.pass_rate == 2 / 3
    assert summary.per_category_accuracy["billing"] == 0.5
    assert summary.per_category_accuracy["technical"] == 1.0
    assert summary.case_pass == {"c-1": True, "c-2": False, "c-3": True}


def test_diff_no_previous():
    results = [make_result("c-1", Category.BILLING, True)]
    report = diff(results, None)

    assert report.previous is None
    assert report.overall_pass_rate_delta is None
    assert report.per_category_accuracy_delta == {}
    assert report.regressions == []
    assert report.improvements == []


def test_diff_flips_and_deltas():
    previous = summarize(
        1,
        [
            make_result("c-1", Category.BILLING, True),
            make_result("c-2", Category.BILLING, False),
            make_result("c-3", Category.BILLING, True),
        ],
    )
    current = [
        make_result("c-1", Category.BILLING, False),  # regression
        make_result("c-2", Category.BILLING, True),   # improvement
        make_result("c-3", Category.BILLING, True),
    ]

    report = diff(current, previous)

    assert report.regressions == ["c-1"]
    assert report.improvements == ["c-2"]
    assert report.current.pass_rate == 2 / 3
    assert report.previous.pass_rate == 2 / 3
    assert report.overall_pass_rate_delta == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_diff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_regress.diff'`.

- [ ] **Step 3: Write minimal implementation**

Create `llm_regress/diff.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from .scoring import CaseResult


@dataclass
class RunSummary:
    run_id: int
    total: int = 0
    passed: int = 0
    per_category_accuracy: dict[str, float] = field(default_factory=dict)
    case_pass: dict[str, bool] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class DiffReport:
    previous: RunSummary | None
    current: RunSummary
    overall_pass_rate_delta: float | None
    per_category_accuracy_delta: dict[str, float]
    regressions: list[str]
    improvements: list[str]


def summarize(run_id: int, results: list[CaseResult]) -> RunSummary:
    summary = RunSummary(run_id=run_id, total=len(results))
    category_totals: dict[str, int] = {}
    category_passed: dict[str, int] = {}
    for result in results:
        category = result.expected.category.value
        category_totals[category] = category_totals.get(category, 0) + 1
        if result.category_match:
            summary.passed += 1
            category_passed[category] = category_passed.get(category, 0) + 1
        summary.case_pass[result.case_id] = result.category_match
    for category, total in category_totals.items():
        summary.per_category_accuracy[category] = category_passed.get(category, 0) / total
    return summary


def diff(current: list[CaseResult], previous: RunSummary | None) -> DiffReport:
    current_summary = summarize(0, current)
    if previous is None:
        return DiffReport(
            previous=None,
            current=current_summary,
            overall_pass_rate_delta=None,
            per_category_accuracy_delta={},
            regressions=[],
            improvements=[],
        )

    overall_delta = current_summary.pass_rate - previous.pass_rate
    categories = set(current_summary.per_category_accuracy) | set(previous.per_category_accuracy)
    category_delta = {
        category: current_summary.per_category_accuracy.get(category, 0.0)
        - previous.per_category_accuracy.get(category, 0.0)
        for category in categories
    }

    regressions = [
        case_id
        for case_id, was_passing in previous.case_pass.items()
        if was_passing and case_id in current_summary.case_pass and not current_summary.case_pass[case_id]
    ]
    improvements = [
        case_id
        for case_id, was_passing in previous.case_pass.items()
        if not was_passing and case_id in current_summary.case_pass and current_summary.case_pass[case_id]
    ]

    return DiffReport(
        previous=previous,
        current=current_summary,
        overall_pass_rate_delta=overall_delta,
        per_category_accuracy_delta=category_delta,
        regressions=regressions,
        improvements=improvements,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_diff.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/diff.py tests/test_diff.py
git commit -m "feat: run-over-run diff logic"
```

---

### Task 6: Store (SQLite schema + previous-run load)

**Files:**
- Modify: `llm_regress/store.py` (rewrite)
- Test: `tests/test_store.py` (create)

**Interfaces:**
- Consumes: `CaseResult` (scoring), `RunSummary` (diff).
- Produces: `Store(path)` with `record_results(feature: str, model: str, commit: str, results: list[CaseResult]) -> int` and `load_previous_run(feature: str, model: str) -> RunSummary | None`.
- Replaces old `Store.record_run(...)` (no callers in-scope; nothing else imports it).

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_store.py -v`
Expected: FAIL — `AttributeError`/`TypeError` (old `record_run` exists but `record_results` does not).

- [ ] **Step 3: Write minimal implementation**

Replace `llm_regress/store.py` entirely:

```python
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
    commit TEXT,
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
                "INSERT INTO runs (feature, model, commit, created_at) VALUES (?, ?, ?, ?)",
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_store.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/store.py tests/test_store.py
git commit -m "feat: SQLite run storage with previous-run loader"
```

---

### Task 7: Config — RunnerConfig + updated config.yaml

**Files:**
- Modify: `llm_regress/config.py` (add `RunnerConfig`, wire into `Config` + `load`)
- Modify: `config.yaml` (feature name, dimensions, runner block)
- Test: `tests/test_config.py` (create)

**Interfaces:**
- Produces: `RunnerConfig(max_concurrency: int = 8, timeout_seconds: float = 60.0, max_retries: int = 2)`. `Config` gains `runner: RunnerConfig` field, populated by `Config.load`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from llm_regress.config import Config, RunnerConfig


def test_load_config_reads_runner_and_dimensions():
    config = Config.load("config.yaml")

    assert config.feature["name"] == "email_classifier"
    assert isinstance(config.runner, RunnerConfig)
    assert config.runner.max_concurrency == 8
    assert config.runner.timeout_seconds == 60
    assert config.runner.max_retries == 2
    assert config.dimensions == ["tone", "relevance", "grounding", "summary_relevance"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `Config` has no `runner` attribute (or feature name mismatch).

- [ ] **Step 3: Write minimal implementation**

In `llm_regress/config.py`, add:

```python
@dataclass
class RunnerConfig:
    max_concurrency: int = 8
    timeout_seconds: float = 60.0
    max_retries: int = 2
```

Add `runner: RunnerConfig` to the `Config` dataclass (after `dimensions`), and in `Config.load` add:

```python
runner = RunnerConfig(**raw.get("eval", {}).get("runner", {}))
```

and pass `runner=runner` into the `cls(...)` call.

Update `config.yaml`:

```yaml
feature:
  name: email_classifier
  description: Classifies customer support emails into category + one-sentence summary

model_under_test:
  provider: nvidia_nim
  base_url: https://integrate.api.nvidia.com/v1
  model: meta/llama-3.1-8b-instruct

judge:
  provider: nvidia_nim
  base_url: https://integrate.api.nvidia.com/v1
  model: nvidia/llama-3.1-nemotron-70b-instruct

eval:
  dimensions:
    - tone
    - relevance
    - grounding
    - summary_relevance
  runner:
    max_concurrency: 8
    timeout_seconds: 60
    max_retries: 2

regression:
  absolute_floor: 3.5
  max_drop: 0.5

storage:
  sqlite_path: results/eval.db
  report_dir: results/reports

slack:
  webhook_url_env: SLACK_WEBHOOK_URL
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/config.py config.yaml tests/test_config.py
git commit -m "feat: runner config and email_classifier wiring"
```

---

### Task 8: Evaluator orchestrator + import fixes

**Files:**
- Modify: `llm_regress/evaluator.py` (rewrite: remove old `CaseResult`/stub `Evaluator`; add orchestrator)
- Modify: `llm_regress/regress.py` (import `CaseResult` from `scoring`)
- Modify: `llm_regress/report.py` (import `CaseResult` from `scoring`)
- Test: `tests/test_evaluator.py` (create)

**Interfaces:**
- Consumes: `Config` (config), `Runner` (runner), `Scorer` + `CaseResult` (scoring), `Judge` (judge), `GoldenDataset` + `PromptConfig` (types).
- Produces: `Evaluator(config, *, runner=None, scorer=None)` with `async run_async(dataset: GoldenDataset, prompt: PromptConfig) -> list[CaseResult]` and `run(dataset, prompt) -> list[CaseResult]`. Optional `runner`/`scorer` injection for tests.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluator.py`:

```python
import asyncio

from llm_regress.config import Config, ModelConfig, RegressionConfig, RunnerConfig, StorageConfig
from llm_regress.evaluator import Evaluator
from llm_regress.types import Category, ExpectedOutput, GoldenCase, GoldenDataset, PromptConfig


class FakeRunner:
    def __init__(self, results):
        self.results = results
        self.seen_prompt = None

    async def run(self, cases):
        self.seen_prompt = "prompt-used"
        return self.results


class FakeScorer:
    async def score_many(self, raws, cases):
        return raws


def make_config():
    return Config(
        feature={"name": "email_classifier"},
        model_under_test=ModelConfig(provider="nvidia_nim", base_url="https://x/v1", model="m"),
        judge=ModelConfig(provider="nvidia_nim", base_url="https://x/v1", model="j"),
        dimensions=["tone"],
        runner=RunnerConfig(),
        regression=RegressionConfig(),
        storage=StorageConfig(),
    )


def make_dataset():
    case = GoldenCase(
        id="c-1",
        input="email",
        expected=ExpectedOutput(category=Category.BILLING, summary="ref"),
        expected_difficulty=1,
        notes="n",
    )
    return GoldenDataset(version=1, feature="email_classifier", version_date="2026-09-06", cases=[case])


def make_prompt():
    return PromptConfig(version_id="v1", timestamp="2026-09-06T00:00:00Z", system_prompt="s")


def test_run_composes_runner_and_scorer():
    raw = {"marker": "raw"}
    runner = FakeRunner([raw])
    scorer = FakeScorer()
    evaluator = Evaluator(make_config(), runner=runner, scorer=scorer)

    results = asyncio.run(evaluator.run_async(make_dataset(), make_prompt()))

    assert results == [raw]
    assert runner.seen_prompt is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluator.py -v`
Expected: FAIL — `Evaluator.__init__` does not accept `runner=`/`scorer=`; `run_async` missing.

- [ ] **Step 3: Write minimal implementation**

Replace `llm_regress/evaluator.py` entirely:

```python
from __future__ import annotations

import asyncio
import os

from .config import Config
from .judge import Judge
from .runner import Runner
from .scoring import CaseResult, Scorer
from .types import GoldenDataset, PromptConfig


class Evaluator:
    def __init__(self, config: Config, *, runner: Runner | None = None, scorer: Scorer | None = None):
        self.config = config
        self._runner = runner
        self._scorer = scorer

    def _make_runner(self, prompt: PromptConfig) -> Runner:
        if self._runner is not None:
            return self._runner
        model_under_test = self.config.model_under_test
        return Runner(
            prompt,
            model_under_test.model,
            model_under_test.base_url,
            api_key=os.getenv(model_under_test.api_key_env),
            max_concurrency=self.config.runner.max_concurrency,
            timeout_seconds=self.config.runner.timeout_seconds,
            max_retries=self.config.runner.max_retries,
        )

    def _make_scorer(self) -> Scorer:
        if self._scorer is not None:
            return self._scorer
        judge = self.config.judge
        return Scorer(
            Judge(judge.model, judge.base_url, api_key=os.getenv(judge.api_key_env)),
            max_concurrency=self.config.runner.max_concurrency,
        )

    async def run_async(self, dataset: GoldenDataset, prompt: PromptConfig) -> list[CaseResult]:
        raws = await self._make_runner(prompt).run(dataset.cases)
        return await self._make_scorer().score_many(raws, dataset.cases)

    def run(self, dataset: GoldenDataset, prompt: PromptConfig) -> list[CaseResult]:
        return asyncio.run(self.run_async(dataset, prompt))
```

In `llm_regress/regress.py`, change:

```python
from .evaluator import CaseResult
```

to:

```python
from .scoring import CaseResult
```

In `llm_regress/report.py`, change:

```python
from .evaluator import CaseResult
```

to:

```python
from .scoring import CaseResult
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluator.py -v`
Expected: PASS (1 passed).

Then run the full suite:

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all tests PASS (existing 13 + new = all green).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/evaluator.py llm_regress/regress.py llm_regress/report.py tests/test_evaluator.py
git commit -m "feat: evaluator orchestration + CaseResult import fixes"
```

---

## Self-Review Notes

- Spec coverage: 3.1 → Task 3; 3.2 → Tasks 2 + 4; 3.3 → Tasks 5 + 6; config changes → Task 7; orchestrator → Task 8; models → Task 1.
- `CaseResult` moved from `evaluator.py` to `scoring.py`; importers (`regress.py`, `report.py`, `store.py`) updated (store in Task 6, others in Task 8).
- `Judge` no longer imports `deepeval` (removed in Task 2); package stays importable with only installed deps.
- No placeholders, no new dependencies, per-case error isolation honored.
