# LLM Regression Detection System

CI/CD-style pipeline that continuously tests an LLM-powered feature against a
golden dataset whenever a prompt or model changes, detects quality regressions,
and alerts the team via Slack before bad outputs reach users.

## Feature under test

Customer support email classifier. Given an email, it returns:

- **category** — one of `billing`, `technical`, `account`, `general`
- **summary** — a one-sentence summary of the email

## Stack

- Python 3.11+, DeepEval (LLM-judge metrics)
- NVIDIA NIM (OpenAI-compatible) for model-under-test and judge
- Golden dataset: versioned JSON in repo, human-verified ground truth
- Storage: SQLite + JSON reports
- CI: GitHub Actions on every PR (Docker)
- Dashboard: Streamlit
- Alerts: Slack webhook

## Layout

```
llm_regress/            core pipeline package
  features/             feature under test (email_classifier.py)
  types.py              Pydantic contracts (PromptConfig, ClassifierOutput,
                        GoldenCase, GoldenDataset)
  prompts.py            prompt loader
  dataset.py            golden dataset loader
prompts/                versioned prompts (email_classifier/v1.yaml)
datasets/golden.json    75-case golden dataset (v1)
tests/                  unit tests
config.yaml             feature, models, thresholds
.github/workflows/      eval workflow
streamlit_app.py        results dashboard
```

## Golden dataset

`datasets/golden.json` is human-verified ground truth. Each case has:

- `id` — stable content hash (`c-` + sha256 of category + input)
- `input` — the customer email
- `expected.category` — correct category (`category_any` lists acceptable
  answers for ambiguous cases)
- `expected.summary` — ideal one-sentence summary (omitted for adversarial cases)
- `expected_difficulty` — 1 (trivial) to 5 (adversarial)
- `notes` — why this case matters

The dataset is versioned (`version`, `version_date`). Bump the version whenever
the eval bar changes.

## Run locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m llm_regress.cli --help
```

Run tests:

```
.venv\Scripts\python.exe -m pytest -q
```

## Regression rules

Aggregate score across all cases/dimensions. Fail if:

- aggregate < `absolute_floor` (default 3.5), or
- drop from baseline > `max_drop` (default 0.5)

## Status

- **Phase 1 — define feature**: `classify_email()` + versioned prompts (merged).
- **Phase 2 — golden dataset**: 75 hand-written cases + typed schema + loader.
- **Pending**: judge/evaluator, baseline, regression runner, CLI commands,
  storage, reporting, Slack integration (stubs exist).
