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
