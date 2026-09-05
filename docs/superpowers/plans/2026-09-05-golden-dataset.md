# Golden Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author a 75-case human-verified golden dataset for the email classifier and wire it into typed, versioned schema + loader.

**Architecture:** Extend the existing `types.py` (Pydantic) and `dataset.py` (loader) in place to the phase-2 schema, then hand-write 75 cases into `datasets/golden.json`. Loader tests validate structural invariants.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest. Pure-data work — no model calls.

## Global Constraints

- No LLM generation of cases — every case hand-authored.
- 75 cases exactly.
- Categories restricted to: `billing`, `technical`, `account`, `general`.
- `expected_difficulty` integer 1-5 inclusive.
- Every case has `notes`.
- `category_any`, when present, must contain `category`.
- `id` = `"c-" + sha256(category + "\x00" + input).hexdigest()[:10]`.
- Dataset `version` starts at 1; file `datasets/golden.json`.
- Run tests with: `.venv\Scripts\python.exe -m pytest -q` from repo root `D:\Model regreesion detection system`.

---

### Task 1: Pydantic models for golden schema

**Files:**
- Modify: `llm_regress/types.py` (append new models)

**Interfaces:**
- Consumes: existing `Category(str, Enum)` in same file.
- Produces: `ExpectedOutput`, `GoldenCase`, `GoldenDataset` — exact models below, used by Task 2 (loader) and Task 4 (tests).

- [ ] **Step 1: Write the failing test**

Create `tests/test_golden_types.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_regress.types import Category, ExpectedOutput, GoldenCase, GoldenDataset


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


def test_golden_dataset_roundtrip():
    ds = GoldenDataset(
        version=1, feature="email_classifier", version_date="2026-09-05",
        cases=[GoldenCase(
            id="c-abc", input="hi",
            expected=ExpectedOutput(category="general", summary="Greeting."),
            expected_difficulty=1, notes="trivial",
        )],
    )
    assert ds.version == 1
    assert len(ds.cases) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_golden_types.py -q`
Expected: FAIL — `ImportError` (models not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `llm_regress/types.py`:

```python
class ExpectedOutput(BaseModel):
    category: Category = Field(description="Single best-answer category")
    category_any: list[Category] | None = Field(
        default=None, description="Acceptable categories for ambiguous cases"
    )
    summary: str | None = Field(
        default=None, description="Ideal one-sentence summary (None for hard cases)"
    )


class GoldenCase(BaseModel):
    id: str = Field(description="Stable content-hash ID")
    input: str = Field(description="Customer email text")
    expected: ExpectedOutput
    expected_difficulty: int = Field(ge=1, le=5)
    notes: str = Field(description="Why this case matters")
```

Note: `GoldenDataset` is added in Task 3 after we confirm the file shape; for now only these two models plus the existing `Category`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_golden_types.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add llm_regress/types.py tests/test_golden_types.py
git commit -m "feat: add golden dataset pydantic models"
```

---

### Task 2: Loader for golden dataset

**Files:**
- Modify: `llm_regress/dataset.py`
- Test: `tests/test_dataset_loader.py`

**Interfaces:**
- Consumes: `GoldenCase`, `ExpectedOutput` from Task 1.
- Produces: `class GoldenDataset` (Pydantic, in `types.py`) with `version`, `feature`, `version_date`, `cases: list[GoldenCase]`, and `@classmethod load(path) -> GoldenDataset` in `dataset.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_loader.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset_loader.py -q`
Expected: FAIL — `ImportError: cannot import name 'GoldenDataset'`.

- [ ] **Step 3: Write minimal implementation**

Append to `llm_regress/types.py`:

```python
class GoldenDataset(BaseModel):
    version: int
    feature: str
    version_date: str
    cases: list[GoldenCase]
```

Rewrite `llm_regress/dataset.py` (replace existing content) to:

```python
from __future__ import annotations

import json
from pathlib import Path

from .types import GoldenDataset


class Dataset(GoldenDataset):
    @classmethod
    def load(cls, path: str | Path) -> "GoldenDataset":
        raw = json.loads(Path(path).read_text())
        return GoldenDataset(**raw)
```

- [ ] **Step 4: Run test to verify it passes** — will FAIL on `len == 75` until data exists. Confirm the `ImportError` is gone and only data-count failures remain:

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset_loader.py -q`
Expected: `test_load_golden_dataset` fails on 75-count; others may error on empty/missing data. This is the expected TDD red state before Task 3. Move on; do NOT commit the loader test as green yet.

- [ ] **Step 5: Commit loader code only (tests expected-red for count)**

```bash
git add llm_regress/dataset.py llm_regress/types.py
git commit -m "feat: golden dataset loader via pydantic"
```

(Leave the red count test uncommitted until Task 3 adds data, or commit tests as red — prefer committing loader code only and deferring test file to Task 3.)

---

### Task 3: Author the 75-case golden dataset

**Files:**
- Rewrite: `datasets/golden.json`

**Interfaces:**
- Consumes: schema from Task 1 (id format, expected fields, difficulty 1-5, notes).
- Produces: `datasets/golden.json` v1 with exactly 75 cases.

- [ ] **Step 1: Author cases**

Hand-write 75 cases directly into `datasets/golden.json` following this shape:

```json
{
  "version": 1,
  "feature": "email_classifier",
  "version_date": "2026-09-05",
  "cases": [
    {
      "id": "c-<hash>",
      "input": "<email text>",
      "expected": {
        "category": "billing",
        "summary": "<ideal one-sentence summary>"
      },
      "expected_difficulty": 2,
      "notes": "<why this case matters>"
    }
  ]
}
```

Distribution target (75 total):

| group | count |
|-------|-------|
| billing clean | 15 |
| technical clean | 15 |
| account clean | 15 |
| general clean | 12 |
| ambiguous (category_any) | 6 |
| extremely short | 6 |
| typos | 4 |
| mixed language | 2 |
| sarcastic | 2 |
| hard/adversarial (difficulty 5) | 2 |

Cross-cutting: edge-case groups overlap clean categories; each edge case carries
`expected_difficulty` >= 4 unless it is a trivial-short case. Every ambiguous case
sets `expected.category_any` (superset of `category`). Every hard/adversarial case
omits `summary` (sets `"summary": null`).

ID generation: compute `sha256(category + "\x00" + input).hexdigest()[:10]`, prefix `c-`.

- [ ] **Step 2: Generate IDs deterministically**

Use a one-off helper (run via python, not committed) to compute the 75 IDs and paste them into the JSON:

```python
import hashlib, json
from pathlib import Path
p = Path("datasets/golden.json")
data = json.loads(p.read_text())
for c in data["cases"]:
    key = c["expected"]["category"] + "\x00" + c["input"]
    c["id"] = "c-" + hashlib.sha256(key.encode()).hexdigest()[:10]
p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
```

- [ ] **Step 3: Verify against invariant tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset_loader.py -q`
Expected: PASS (4 tests) — 75 cases, unique IDs, valid categories, difficulty 1-5, notes present, category_any superset.

- [ ] **Step 4: Commit**

```bash
git add datasets/golden.json tests/test_dataset_loader.py
git commit -m "feat: author 75-case golden dataset v1"
```

---

### Task 4: Full suite green + push

**Files:** none new.

- [ ] **Step 1: Run full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all tests PASS (golden types 4 + loader 4 + prior phase-1 tests 5 = 13).

- [ ] **Step 2: Commit any loose ends and push**

```bash
git add -A
git commit -m "test: golden dataset invariants"  # only if needed
git push -u origin feat/phase2-golden-dataset
```

---

## Self-Review

- **Spec coverage:** 2.1 (75 hand-written cases) = Task 3; 2.2 (edge cases with `expected_difficulty`) = Task 3 distribution + `GoldenCase.expected_difficulty`; 2.3 (versioned JSON, stable ID, notes) = Task 3 schema + Task 1 models. Loader = Task 2. Tests = Tasks 1,2,4.
- **Placeholder scan:** none — all steps have concrete code or data.
- **Type consistency:** `GoldenCase`, `ExpectedOutput`, `GoldenDataset`, `Category` names consistent across Tasks 1-4; `load` defined on `Dataset(GoldenDataset)` classmethod returning `GoldenDataset`.
