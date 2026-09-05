from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_regress.features.email_classifier import classify_email
from llm_regress.prompts import load_prompt
from llm_regress.types import Category

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "email_classifier" / "v1.yaml"


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def test_classify_email_parses_output():
    fake = FakeClient('{"category": "billing", "summary": "Double charge refund."}')
    prompt = load_prompt(PROMPT_PATH)
    out = classify_email("I was charged twice.", prompt, model="test-model", client=fake)
    assert out.category == Category.BILLING
    assert out.summary == "Double charge refund."


def test_classify_email_injects_few_shot_examples():
    fake = FakeClient('{"category": "general", "summary": "Thanks."}')
    prompt = load_prompt(PROMPT_PATH)
    classify_email("thanks!", prompt, model="test-model", client=fake)
    messages = fake.chat.completions.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert any(m["role"] == "assistant" for m in messages)
    assert messages[-1] == {"role": "user", "content": "thanks!"}


def test_invalid_category_raises():
    fake = FakeClient('{"category": "unknown", "summary": "Nope."}')
    prompt = load_prompt(PROMPT_PATH)
    with pytest.raises(ValidationError):
        classify_email("hello", prompt, model="test-model", client=fake)
