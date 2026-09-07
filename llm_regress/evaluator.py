from __future__ import annotations

import asyncio
import os

from .config import Config
from .judge import Judge
from .runner import Runner
from .scoring import CaseResult, Scorer
from .types import GoldenDataset, PromptConfig


class Evaluator:
    def __init__(self, config: Config, *, runner: Runner | None = None, scorer: Scorer | None = None):
        self.config = config
        self._runner = runner
        self._scorer = scorer

    def _make_runner(self, prompt: PromptConfig) -> Runner:
        if self._runner is not None:
            return self._runner
        model_under_test = self.config.model_under_test
        return Runner(
            prompt,
            model_under_test.model,
            model_under_test.base_url,
            api_key=os.getenv(model_under_test.api_key_env),
            max_concurrency=self.config.runner.max_concurrency,
            timeout_seconds=self.config.runner.timeout_seconds,
            max_retries=self.config.runner.max_retries,
        )

    def _make_scorer(self) -> Scorer:
        if self._scorer is not None:
            return self._scorer
        judge = self.config.judge
        return Scorer(
            Judge(judge.model, judge.base_url, api_key=os.getenv(judge.api_key_env)),
            max_concurrency=self.config.runner.max_concurrency,
        )

    async def run_async(self, dataset: GoldenDataset, prompt: PromptConfig) -> list[CaseResult]:
        raws = await self._make_runner(prompt).run(dataset.cases)
        return await self._make_scorer().score_many(raws, dataset.cases)

    def run(self, dataset: GoldenDataset, prompt: PromptConfig) -> list[CaseResult]:
        return asyncio.run(self.run_async(dataset, prompt))
