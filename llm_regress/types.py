from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassifierOutput(BaseModel):
    category: Category = Field(description="Support category for the email")
    summary: str = Field(description="One-sentence summary of the email")


class FewShotExample(BaseModel):
    input: str = Field(description="Example email text")
    output: ClassifierOutput = Field(description="Expected classification for the example")


class PromptConfig(BaseModel):
    version_id: str = Field(description="Unique prompt version identifier")
    timestamp: datetime = Field(description="When this prompt version was created")
    system_prompt: str = Field(description="System prompt that drives the classifier")
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
