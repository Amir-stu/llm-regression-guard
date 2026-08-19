from __future__ import annotations

from dataclasses import replace

import pytest

from regression_guard.config import Settings, Thresholds
from regression_guard.models import (
    CaseResult,
    Category,
    Difficulty,
    GoldenCase,
    GoldenDataset,
    PromptConfig,
    RunSummary,
)


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        provider="mock",
        db_path=tmp_path / "runs.db",
        report_dir=tmp_path / "reports",
        concurrency=4,
    )


@pytest.fixture
def prompt_config() -> PromptConfig:
    return PromptConfig(
        version="test-v1",
        created_at="2026-01-01T00:00:00Z",
        system_prompt=(
            "Triage support email. Write one sentence.\n"
            "- billing: invoice, refund, charge\n"
            "- technical: error, crash, broken\n"
            "- account: password, login, access\n"
            "- general: hello, thanks, roadmap\n"
        ),
    )


@pytest.fixture
def dataset() -> GoldenDataset:
    return GoldenDataset(
        dataset_version="test.v1",
        created_at="2026-01-01",
        cases=[
            GoldenCase(
                id="c1",
                email="Please refund the duplicate invoice charge.",
                expected_category=Category.BILLING,
                expected_summary="Customer wants a refund for a duplicate invoice charge.",
                expected_difficulty=Difficulty.EASY,
                notes="canonical billing",
            ),
            GoldenCase(
                id="c2",
                email="The export crashes with an error every time.",
                expected_category=Category.TECHNICAL,
                expected_summary="Export crashes with an error every time.",
                expected_difficulty=Difficulty.EASY,
                notes="canonical technical",
            ),
            GoldenCase(
                id="c3",
                email="I cannot login, my password reset never arrives.",
                expected_category=Category.ACCOUNT,
                expected_summary="Customer cannot login and password reset emails never arrive.",
                expected_difficulty=Difficulty.MEDIUM,
                notes="canonical account",
            ),
        ],
    )


def make_summary(**overrides) -> RunSummary:
    base = {
        "run_id": "run-a",
        "prompt_version": "v1",
        "dataset_version": "d1",
        "model": "gpt-4o-mini",
        "provider": "mock",
        "total_cases": 100,
        "passed_cases": 90,
        "pass_rate": 0.90,
        "category_accuracy": 0.92,
        "mean_judge_score": 4.2,
        "composite_score": 0.884,
        "p50_latency_ms": 100.0,
        "p95_latency_ms": 200.0,
        "total_tokens": 1000,
        "error_count": 0,
    }
    base.update(overrides)
    return RunSummary(**base)


def make_result(case_id: str, passed: bool, category: Category = Category.BILLING, **kw) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        expected_category=Category.BILLING,
        predicted_category=category,
        predicted_summary=kw.pop("summary", "a summary"),
        category_match=kw.pop("category_match", passed),
        judge_score=kw.pop("judge_score", 5 if passed else 2),
        passed=passed,
        **kw,
    )
