from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .dataset import Case, Dataset
from .judge import DimensionScore, Judge


@dataclass
class CaseResult:
    case_id: str
    case_name: str
    output: str
    scores: list[DimensionScore]

    @property
    def average(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


class Evaluator:
    def __init__(self, config: Config):
        self.config = config

    def run(self, dataset: Dataset) -> list[CaseResult]:
        raise NotImplementedError("evaluator implementation lands in a later phase")
