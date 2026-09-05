from __future__ import annotations

from pathlib import Path

from llm_regress.prompts import load_prompt
from llm_regress.types import Category

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "email_classifier" / "v1.yaml"


def test_load_prompt_fields():
    prompt = load_prompt(PROMPT_PATH)
    assert prompt.version_id == "email_classifier-v1"
    assert prompt.system_prompt
    assert len(prompt.few_shot_examples) == 4


def test_few_shot_categories_parsed():
    prompt = load_prompt(PROMPT_PATH)
    categories = [ex.output.category for ex in prompt.few_shot_examples]
    assert categories == [Category.BILLING, Category.TECHNICAL, Category.ACCOUNT, Category.GENERAL]
