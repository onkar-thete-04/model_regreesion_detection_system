from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class RegressionConfig:
    absolute_floor: float = 3.5
    max_drop: float = 0.5


@dataclass
class StorageConfig:
    sqlite_path: str = "results/eval.db"
    report_dir: str = "results/reports"


@dataclass
class Config:
    feature: dict
    model_under_test: ModelConfig
    judge: ModelConfig
    dimensions: list[str]
    regression: RegressionConfig
    storage: StorageConfig
    slack_webhook_env: str = "SLACK_WEBHOOK_URL"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        model = ModelConfig(**raw["model_under_test"])
        judge = ModelConfig(**raw["judge"])
        regression = RegressionConfig(**raw.get("regression", {}))
        storage = StorageConfig(**raw.get("storage", {}))
        return cls(
            feature=raw.get("feature", {}),
            model_under_test=model,
            judge=judge,
            dimensions=raw.get("eval", {}).get("dimensions", ["tone", "relevance", "grounding"]),
            regression=regression,
            storage=storage,
            slack_webhook_env=raw.get("slack", {}).get("webhook_url_env", "SLACK_WEBHOOK_URL"),
        )
