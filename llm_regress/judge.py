from __future__ import annotations

from dataclasses import dataclass

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


@dataclass
class DimensionScore:
    dimension: str
    score: float
    reason: str


class Judge:
    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env

    def grade(self, case_input: str, actual_output: str, dimensions: list[str], rubric: dict) -> list[DimensionScore]:
        raise NotImplementedError("judge implementation lands in a later phase")
