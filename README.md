# LLM Regression Detection System

CI/CD-style pipeline that continuously tests an LLM-powered feature against a
golden dataset whenever a prompt or model changes, detects quality regressions,
and alerts the team via Slack before bad outputs reach users.

## Stack

- Python 3.11+, DeepEval (LLM-judge metrics)
- NVIDIA NIM (OpenAI-compatible) for model-under-test and judge
- Golden dataset: JSON in repo, rubric-graded cases
- Storage: SQLite + JSON reports
- CI: GitHub Actions on every PR (Docker)
- Dashboard: Streamlit
- Alerts: Slack webhook

## Layout

```
llm_regress/       core pipeline package
datasets/          golden dataset JSON
config.yaml        feature, models, thresholds
.github/workflows/ eval workflow
streamlit_app.py   results dashboard
```

## Run locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m llm_regress.cli --help
```

## Regression rules

Aggregate score across all cases/dimensions. Fail if:

- aggregate < `absolute_floor` (default 3.5), or
- drop from baseline > `max_drop` (default 0.5)
