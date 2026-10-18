"""Typed contracts shared by every layer of the pipeline.

These models are the interface between the feature under test, the eval
engine, and the reporting layer. Changing one of them is a breaking change
for stored runs, so `SCHEMA_VERSION` is bumped alongside.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 2


class Category(enum.StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class Difficulty(enum.StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Severity(enum.StrEnum):
    """Whether a failed deterministic check fails the case (hard) or is only
    counted in error analysis (soft)."""

    HARD = "hard"
    SOFT = "soft"


class AssertionFailure(BaseModel):
    """One deterministic check that did not hold for a model output."""

    model_config = ConfigDict(extra="forbid")

    check: str
    severity: Severity
    detail: str = ""


class ClassificationOutput(BaseModel):
    """What the feature under test must return."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    summary: str = Field(min_length=1, max_length=400)

    @field_validator("summary")
    @classmethod
    def _single_sentence_ish(cls, value: str) -> str:
        return " ".join(value.split())


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """A versioned prompt. This is the artefact CI runs against."""

    model_config = ConfigDict(extra="forbid")

    version: str
    created_at: str
    description: str = ""
    system_prompt: str
    user_template: str = "{email}"
    few_shot: list[FewShotExample] = Field(default_factory=list)
    temperature: float = 0.0

    @field_validator("user_template")
    @classmethod
    def _must_carry_email(cls, value: str) -> str:
        if "{email}" not in value:
            raise ValueError("user_template must contain the {email} placeholder")
        return value


class GoldenCase(BaseModel):
    """One hand-labelled test case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    email: str
    expected_category: Category
    expected_summary: str
    expected_difficulty: Difficulty
    notes: str
    # Cases that a reasonable human could file under two headings. The scorer
    # accepts either, otherwise the eval punishes the model for our ambiguity.
    acceptable_categories: list[Category] = Field(default_factory=list)
    # Deterministic, case-specific constraints on the summary. `must_include`
    # is for identifiers an agent needs to act on; `must_not_include` is for
    # content that must never be copied into a ticket, such as card digits.
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)

    def accepts(self, category: Category) -> bool:
        return category == self.expected_category or category in self.acceptable_categories


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    created_at: str
    description: str = ""
    cases: list[GoldenCase]

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[GoldenCase]) -> list[GoldenCase]:
        seen: set[str] = set()
        for case in cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
        return cases


class CaseResult(BaseModel):
    """Per-case, multi-dimensional score."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_category: Category
    predicted_category: Category | None
    predicted_summary: str = ""
    category_match: bool
    judge_score: int | None = None
    judge_reason: str = ""
    passed: bool
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    difficulty: Difficulty = Difficulty.EASY
    assertion_failures: list[AssertionFailure] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunSummary(BaseModel):
    """Aggregate scorecard for one eval run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    prompt_version: str
    dataset_version: str
    model: str
    provider: str
    git_sha: str = ""
    total_cases: int
    passed_cases: int
    pass_rate: float
    category_accuracy: float
    mean_judge_score: float
    composite_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    error_count: int
    per_category_accuracy: dict[str, float] = Field(default_factory=dict)
    per_difficulty_pass_rate: dict[str, float] = Field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION


class Verdict(enum.StrEnum):
    PASS = "pass"
    WARN = "warn"
    CRITICAL = "critical"


class FlippedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    difficulty: Difficulty
    expected_category: Category
    old_category: Category | None
    new_category: Category | None
    old_summary: str
    new_summary: str
    old_judge_score: int | None
    new_judge_score: int | None


class Comparison(BaseModel):
    """The diff between the current run and its baseline."""

    model_config = ConfigDict(extra="forbid")

    current: RunSummary
    baseline: RunSummary | None
    pass_rate_delta: float = 0.0
    category_accuracy_delta: float = 0.0
    composite_delta: float = 0.0
    per_category_delta: dict[str, float] = Field(default_factory=dict)
    regressions: list[FlippedCase] = Field(default_factory=list)
    improvements: list[FlippedCase] = Field(default_factory=list)
    verdict: Verdict = Verdict.PASS
    reasons: list[str] = Field(default_factory=list)
    p_value: float | None = None
    significant: bool = False
    drift_warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
