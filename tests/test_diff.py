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
