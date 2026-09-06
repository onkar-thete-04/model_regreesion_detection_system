from __future__ import annotations

import json
from pathlib import Path

from .types import GoldenDataset


class Dataset(GoldenDataset):
    @classmethod
    def load(cls, path: str | Path) -> "GoldenDataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return GoldenDataset(**raw)


GoldenDataset = Dataset
