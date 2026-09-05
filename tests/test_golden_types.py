from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_regress.types import Category, ExpectedOutput, GoldenCase


def test_expected_output_single_category():
    out = ExpectedOutput(category="billing", summary="Refund request.")
    assert out.category == Category.BILLING
    assert out.category_any is None
    assert out.summary == "Refund request."


def test_expected_output_category_any():
    out = ExpectedOutput(category="technical", category_any=["technical", "general"])
    assert out.category_any == [Category.TECHNICAL, Category.GENERAL]


def test_golden_case_requires_notes():
    with pytest.raises(ValidationError):
        GoldenCase(
            id="c-abc", input="hi",
            expected=ExpectedOutput(category="general"),
            expected_difficulty=2,
        )
