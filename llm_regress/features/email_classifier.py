from __future__ import annotations

from openai import OpenAI

from ..types import ClassifierOutput, PromptConfig


def _build_messages(email: str, prompt: PromptConfig) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": prompt.system_prompt}]
    for example in prompt.few_shot_examples:
        messages.append({"role": "user", "content": example.input})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": email})
    return messages


def classify_email(
    email: str,
    prompt: PromptConfig,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
) -> ClassifierOutput:
    openai_client = client or OpenAI(base_url=base_url, api_key=api_key)
    response = openai_client.chat.completions.create(
        model=model,
        messages=_build_messages(email, prompt),
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    return ClassifierOutput.model_validate_json(content)
