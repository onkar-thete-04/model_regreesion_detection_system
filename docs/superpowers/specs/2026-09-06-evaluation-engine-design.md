# Phase 3 — Evaluation Engine Design

Date: 2026-09-06
Feature: customer support email classifier (category + one-sentence summary)

## Goal

Build the evaluation engine: run every golden case through the LLM feature with
async batching, score each case across multiple dimensions (category match,
rubric judge scores, latency, token usage), and diff the current run against the
previous run stored in SQLite.

## Scope

Engine-only — pure functions/classes plus unit tests. CLI wiring
(`baseline`/`run`/`report`), regression verdict, reporting, and Slack land in
later phases.

In scope:

- 3.1 Test runner — async batched execution → raw outputs.
- 3.2 Multi-dimensional scoring — category match, LLM-as-judge rubric, latency,
  token usage.
- 3.3 Comparison logic — diff current vs previous run from SQLite.

Out of scope:

- CLI commands, `RegressionChecker` verdict integration, JSON/HTML reports,
  Slack notification.

## Data Models (`llm_regress/types.py` additions)

```python
class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class RawResult(BaseModel):          # runner output
    case_id: str
    input: str
    output: ClassifierOutput | None  # None on error
    latency_ms: float
    token_usage: TokenUsage | None
    error: str | None = None
```

`CaseResult` (reworked in `llm_regress/evaluator.py`):

- `case_id: str`
- `input: str`
- `expected: ExpectedOutput`
- `actual: ClassifierOutput | None`
- `category_match: bool`
- `scores: list[DimensionScore]` — judge dimensions only
- `latency_ms: float`
- `token_usage: TokenUsage | None`
- `error: str | None`
- `average` property — mean of non-None judge scores (kept for later
  `RegressionChecker` use).

## Dimensions (agreed)

| dimension          | type              | role                                    |
|--------------------|-------------------|-----------------------------------------|
| category_match     | binary (0/1)      | pass/fail basis for 3.3 diff            |
| tone               | judge 1-5         | aggregate quality score                 |
| relevance          | judge 1-5         | aggregate quality score                 |
| grounding          | judge 1-5         | aggregate quality score                 |
| summary_relevance  | judge 1-5         | aggregate quality score                 |
| latency_ms         | measured          | stored + reported only                  |
| token_usage        | measured          | stored + reported only                  |

The 1-5 aggregate = mean of the four judge scores present per case. Latency and
token usage never enter the aggregate or pass/fail.

## 3.1 Runner (`llm_regress/runner.py`)

- `Runner(prompt, model, base_url, api_key=None, *, max_concurrency=8,
  timeout_seconds=60, max_retries=2, client=None)`
- `async run(cases: list[GoldenCase]) -> list[RawResult]` using `AsyncOpenAI`,
  `asyncio.Semaphore(max_concurrency)`, `asyncio.gather`.
- Reuses `_build_messages()` from `features/email_classifier.py` (no duplication).
- Per case: `temperature=0`, `response_format={"type": "json_object"}`,
  `perf_counter()` latency, capture `response.usage`.
- Error handling: retry up to `max_retries` on transient failure; on final
  failure return `RawResult(error=..., output=None)` — one bad case never crashes
  the run.
- Sync wrapper `run_sync()` via `asyncio.run`.

## 3.2 Scoring (`llm_regress/scoring.py` + `llm_regress/judge.py`)

`Judge(model, base_url, api_key=None, *, client=None, timeout_seconds=60)`:

- `async grade(case_input, actual_summary, expected_summary: str | None)
  -> list[DimensionScore]`
- One LLM call per case returning JSON for four rubric dimensions, each
  `{score: 1-5, reason}`.
- `summary_relevance` omitted when `expected_summary` is None (adversarial cases).

Rubric definitions (in the judge system prompt):

- **tone** — summary written in a neutral, professional support tone.
- **relevance** — summary captures the email's core request/issue.
- **grounding** — every claim in the summary is supported by the email text (no
  hallucination).
- **summary_relevance** — generated summary matches the reference summary's
  meaning (only when reference exists).

`Scorer(judge)`:

- `category_match` deterministic: `actual.category == expected.category` OR
  `actual.category in (expected.category_any or [])`.
- Runner error → `category_match=False`, no judge scores.
- `async score_many(raws, cases) -> list[CaseResult]` — judge calls batched with
  the same semaphore pattern.

## 3.3 Diff (`llm_regress/diff.py` + `llm_regress/store.py`)

```python
class RunSummary:
    run_id: int
    total: int
    passed: int
    pass_rate: float
    per_category_accuracy: dict[str, float]  # bucketed by expected.category
    case_pass: dict[str, bool]               # case_id -> category_match

class DiffReport:
    previous: RunSummary | None      # None = no previous run
    current: RunSummary
    overall_pass_rate_delta: float | None
    per_category_accuracy_delta: dict[str, float]
    regressions: list[str]           # case_ids pass -> fail
    improvements: list[str]          # case_ids fail -> pass

def diff(current: list[CaseResult], previous: RunSummary | None) -> DiffReport
```

`diff` is pure. Flips computed from `previous.case_pass` vs current
`category_match`.

**Store changes (`llm_regress/store.py`)**:

- Schema: `runs` + `case_runs` (case-level: `predicted_category`,
  `expected_category`, `category_match`, `latency_ms`, `prompt_tokens`,
  `completion_tokens`) + `case_scores` (`dimension`, `score`, `reason`).
- `record_results(feature, model, commit, results) -> int` (run_id).
- `load_previous_run(feature, model) -> RunSummary | None` — most recent prior
  run for feature+model, reconstructed with `case_pass` map.

## Config Changes

`config.yaml`:

- `feature.name` → `email_classifier`, update description.
- `eval.dimensions` → `[tone, relevance, grounding, summary_relevance]`
  (judge dims; `category_match`/`latency_ms`/`token_usage` always computed).
- Add `eval.runner: {max_concurrency: 8, timeout_seconds: 60, max_retries: 2}`.
- Drop stale `eval.framework: deepeval`.

`config.py`:

- Add `RunnerConfig` dataclass; `Config` gains `runner`; `load()` parses it.

## Error Handling

- Per-case isolation: one bad case records `error`, others unaffected.
- Judge parse failure → empty scores with reason logged; `category_match` still
  deterministic.
- No previous run → diff deltas `None`, empty flips.

## Testing (offline, fake AsyncOpenAI clients)

- `runner`: all cases executed, latency ≥ 0, token usage captured, error case
  isolated, retry works.
- `scoring`: category match + `category_any`; judge scores parsed;
  `summary_relevance` skipped on `expected.summary=None`; error → no judge scores.
- `diff`: flips, per-category deltas, no-previous behavior.
- `store`: record/load round-trip, most-recent selection, empty DB → `None`.
