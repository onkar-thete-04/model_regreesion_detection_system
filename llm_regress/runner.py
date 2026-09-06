from __future__ import annotations

import asyncio
from time import perf_counter

from openai import AsyncOpenAI

from .features.email_classifier import _build_messages
from .types import ClassifierOutput, GoldenCase, PromptConfig, RawResult, TokenUsage


class Runner:
    def __init__(
        self,
        prompt: PromptConfig,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        max_concurrency: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client=None,
    ):
        self.prompt = prompt
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self._client = client or AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    async def run(self, cases: list[GoldenCase]) -> list[RawResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._run_one(semaphore, case) for case in cases]
        return await asyncio.gather(*tasks)

    def run_sync(self, cases: list[GoldenCase]) -> list[RawResult]:
        return asyncio.run(self.run(cases))

    async def _run_one(self, semaphore: asyncio.Semaphore, case: GoldenCase) -> RawResult:
        async with semaphore:
            last_error: Exception | None = None
            for _ in range(self.max_retries + 1):
                try:
                    return await self._attempt(case)
                except Exception as exc:  # noqa: BLE001 - per-case isolation
                    last_error = exc
            return RawResult(case_id=case.id, input=case.input, error=f"{type(last_error).__name__}: {last_error}")

    async def _attempt(self, case: GoldenCase) -> RawResult:
        messages = _build_messages(case.input, self.prompt)
        start = perf_counter()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        latency_ms = (perf_counter() - start) * 1000.0
        content = response.choices[0].message.content
        output = ClassifierOutput.model_validate_json(content)

        usage = response.usage
        token_usage = None
        if usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )

        return RawResult(
            case_id=case.id,
            input=case.input,
            output=output,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
