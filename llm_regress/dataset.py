from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Case:
    id: str
    name: str
    input: str
    rubric: dict


@dataclass
class Dataset:
    version: int
    feature: str
    cases: list[Case]

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        raw = json.loads(Path(path).read_text())
        cases = [
            Case(id=c["id"], name=c["name"], input=c["input"], rubric=c.get("rubric", {}))
            for c in raw.get("cases", [])
        ]
        return cls(version=raw.get("version", 1), feature=raw.get("feature", ""), cases=cases)
