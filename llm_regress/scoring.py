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
                try:
                    scores = await self.judge.grade(case.input, raw.output.summary, case.expected.summary)
                except Exception:
                    scores = []
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
