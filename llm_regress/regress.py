from __future__ import annotations

from dataclasses import dataclass

from .baseline import Baseline
from .config import RegressionConfig
from .evaluator import CaseResult


@dataclass
class Verdict:
    passed: bool
    aggregate_score: float
    absolute_floor: float
    max_drop: float
    baseline_score: float | None
    drop: float | None
    reasons: list[str]


class RegressionChecker:
    def __init__(self, config: RegressionConfig):
        self.config = config

    def check(self, results: list[CaseResult], baseline: Baseline | None) -> Verdict:
        if not results:
            return Verdict(
                passed=False,
                aggregate_score=0.0,
                absolute_floor=self.config.absolute_floor,
                max_drop=self.config.max_drop,
                baseline_score=None,
                drop=None,
                reasons=["no results"],
            )
        aggregate = sum(r.average for r in results) / len(results)
        reasons: list[str] = []
        drop: float | None = None
        baseline_score: float | None = None
        if baseline is not None and baseline.averages:
            baseline_score = sum(baseline.averages.values()) / len(baseline.averages)
            drop = baseline_score - aggregate
            if drop > self.config.max_drop:
                reasons.append(f"score dropped {drop:.2f} from baseline (max {self.config.max_drop})")
        if aggregate < self.config.absolute_floor:
            reasons.append(f"aggregate {aggregate:.2f} below floor {self.config.absolute_floor}")
        return Verdict(
            passed=not reasons,
            aggregate_score=aggregate,
            absolute_floor=self.config.absolute_floor,
            max_drop=self.config.max_drop,
            baseline_score=baseline_score,
            drop=drop,
            reasons=reasons,
        )
