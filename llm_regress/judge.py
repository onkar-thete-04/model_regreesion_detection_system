from __future__ import annotations

import json
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class DimensionScore:
    dimension: str
    score: float
    reason: str


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of email-classifier summaries. Rate the ACTUAL summary of a customer support email on each dimension using a 1-5 integer scale.

- tone: the summary uses a neutral, professional support tone. 5 = perfect professional tone, 1 = rude or unprofessional.
- relevance: the summary captures the email's core request or issue. 5 = captures it precisely, 1 = misses or misstates it.
- grounding: every claim in the summary is supported by the email text. 5 = fully grounded, 1 = hallucinates or fabricates.
- summary_relevance: the summary matches the reference summary's meaning. Rate this ONLY if a reference summary is provided.

Respond with ONLY a JSON object of this shape:
{"tone": {"score": <int 1-5>, "reason": "<short>"}, "relevance": {"score": <int 1-5>, "reason": "<short>"}, "grounding": {"score": <int 1-5>, "reason": "<short>"}, "summary_relevance": {"score": <int 1-5>, "reason": "<short>"}}
If no reference summary is provided, omit the summary_relevance key entirely."""

_JUDGE_DIMENSIONS = ["tone", "relevance", "grounding", "summary_relevance"]


class Judge:
    def __init__(self, model: str, base_url: str, api_key: str | None = None, *, client=None, timeout_seconds: float = 60.0):
        self.model = model
        self._client = client or AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    async def grade(self, case_input: str, actual_summary: str, expected_summary: str | None) -> list[DimensionScore]:
        user_payload = {"email": case_input, "actual_summary": actual_summary}
        if expected_summary is not None:
            user_payload["reference_summary"] = expected_summary

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        content = response.choices[0].message.content
        try:
            raw = json.loads(content)
        except (TypeError, ValueError):
            return []

        if not isinstance(raw, dict):
            return []

        scores: list[DimensionScore] = []
        for dim in _JUDGE_DIMENSIONS:
            entry = raw.get(dim)
            if entry is None:
                continue
            try:
                scores.append(DimensionScore(dimension=dim, score=float(entry["score"]), reason=str(entry.get("reason", ""))))
            except (KeyError, TypeError, ValueError):
                continue
        return scores
