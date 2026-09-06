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


class ExpectedOutput(BaseModel):
    category: Category = Field(description="Single best-answer category")
    category_any: list[Category] | None = Field(
        default=None, description="Acceptable categories for ambiguous cases"
    )
    summary: str | None = Field(
        default=None, description="Ideal one-sentence summary (None for hard cases)"
    )


class GoldenCase(BaseModel):
    id: str = Field(description="Stable content-hash ID")
    input: str = Field(description="Customer email text")
    expected: ExpectedOutput
    expected_difficulty: int = Field(ge=1, le=5)
    notes: str = Field(description="Why this case matters")


class GoldenDataset(BaseModel):
    version: int
    feature: str
    version_date: str
    cases: list[GoldenCase]


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RawResult(BaseModel):
    case_id: str
    input: str
    output: ClassifierOutput | None = None
    latency_ms: float = 0.0
    token_usage: TokenUsage | None = None
    error: str | None = None
