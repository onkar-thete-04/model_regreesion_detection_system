from __future__ import annotations

from pathlib import Path

from llm_regress.dataset import GoldenDataset
from llm_regress.types import Category

GOLDEN = Path(__file__).resolve().parents[1] / "datasets" / "golden.json"


def test_load_golden_dataset():
    ds = GoldenDataset.load(GOLDEN)
    assert ds.version == 1
    assert ds.feature == "email_classifier"
    assert ds.version_date
    assert len(ds.cases) == 75


def test_case_ids_unique_and_valid():
    ds = GoldenDataset.load(GOLDEN)
    ids = [c.id for c in ds.cases]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("c-") and len(i) == 12 for i in ids)


def test_all_categories_valid_and_difficulty_in_range():
    ds = GoldenDataset.load(GOLDEN)
    valid = {c.value for c in Category}
    for case in ds.cases:
        assert case.expected.category.value in valid
        assert 1 <= case.expected_difficulty <= 5
        assert case.notes.strip()


def test_category_any_superset_of_category():
    ds = GoldenDataset.load(GOLDEN)
    for case in ds.cases:
        if case.expected.category_any:
            assert case.expected.category in case.expected.category_any
