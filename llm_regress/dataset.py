from __future__ import annotations

import json
from pathlib import Path

from .types import GoldenDataset


class Dataset(GoldenDataset):
    @classmethod
    def load(cls, path: str | Path) -> "GoldenDataset":
        raw = json.loads(Path(path).read_text())
        return GoldenDataset(**raw)


GoldenDataset = Dataset
