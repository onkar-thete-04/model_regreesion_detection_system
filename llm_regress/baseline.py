from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Baseline:
    feature: str
    averages: dict[str, float]


class BaselineStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Baseline | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text())
        return Baseline(feature=raw.get("feature", ""), averages=raw.get("averages", {}))

    def save(self, baseline: Baseline) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "feature": baseline.feature,
                    "averages": baseline.averages,
                },
                indent=2,
            )
        )
