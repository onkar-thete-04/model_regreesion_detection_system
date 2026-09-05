from __future__ import annotations

from pathlib import Path

import yaml

from .types import PromptConfig


def load_prompt(path: str | Path) -> PromptConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return PromptConfig.model_validate(raw)
