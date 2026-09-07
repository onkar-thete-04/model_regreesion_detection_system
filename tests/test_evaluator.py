import asyncio

from llm_regress.config import Config, ModelConfig, RegressionConfig, RunnerConfig, StorageConfig
from llm_regress.evaluator import Evaluator
from llm_regress.types import Category, ExpectedOutput, GoldenCase, GoldenDataset, PromptConfig


class FakeRunner:
    def __init__(self, results):
        self.results = results
        self.seen_prompt = None

    async def run(self, cases):
        self.seen_prompt = "prompt-used"
        return self.results


class FakeScorer:
    async def score_many(self, raws, cases):
        return raws


def make_config():
    return Config(
        feature={"name": "email_classifier"},
        model_under_test=ModelConfig(provider="nvidia_nim", base_url="https://x/v1", model="m"),
        judge=ModelConfig(provider="nvidia_nim", base_url="https://x/v1", model="j"),
        dimensions=["tone"],
        runner=RunnerConfig(),
        regression=RegressionConfig(),
        storage=StorageConfig(),
    )


def make_dataset():
    case = GoldenCase(
        id="c-1",
        input="email",
        expected=ExpectedOutput(category=Category.BILLING, summary="ref"),
        expected_difficulty=1,
        notes="n",
    )
    return GoldenDataset(version=1, feature="email_classifier", version_date="2026-09-06", cases=[case])


def make_prompt():
    return PromptConfig(version_id="v1", timestamp="2026-09-06T00:00:00Z", system_prompt="s")


def test_run_composes_runner_and_scorer():
    raw = {"marker": "raw"}
    runner = FakeRunner([raw])
    scorer = FakeScorer()
    evaluator = Evaluator(make_config(), runner=runner, scorer=scorer)

    results = asyncio.run(evaluator.run_async(make_dataset(), make_prompt()))

    assert results == [raw]
    assert runner.seen_prompt is not None
