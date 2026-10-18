"""Async test runner: golden dataset in, per-case scores out."""

from __future__ import annotations

import asyncio
import json
import statistics
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .assertions import has_hard_failure, run_assertions
from .classifier import ClassificationError, classify_email
from .config import (
    CATEGORY_WEIGHT,
    DEFAULT_SUMMARY_PASS_THRESHOLD,
    JUDGE_SCALE_MAX,
    JUDGE_SCALE_MIN,
    SUMMARY_WEIGHT,
    Settings,
)
from .judge import JUDGE_CALL_FAILED, judge_summary
from .models import CaseResult, GoldenDataset, PromptConfig, RunSummary
from .providers import LLMProvider


def load_dataset(path: str | Path) -> GoldenDataset:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(data)


def normalise_judge(score: int | None) -> float:
    """Map the 1-5 judge scale onto 0-1."""
    if score is None:
        return 0.0
    span = JUDGE_SCALE_MAX - JUDGE_SCALE_MIN
    return (max(JUDGE_SCALE_MIN, min(JUDGE_SCALE_MAX, score)) - JUDGE_SCALE_MIN) / span


def composite(category_accuracy: float, mean_judge: float) -> float:
    """Weighted blend of the objective and subjective dimensions."""
    return CATEGORY_WEIGHT * category_accuracy + SUMMARY_WEIGHT * normalise_judge_mean(mean_judge)


def normalise_judge_mean(mean_judge: float) -> float:
    span = JUDGE_SCALE_MAX - JUDGE_SCALE_MIN
    return (max(JUDGE_SCALE_MIN, min(JUDGE_SCALE_MAX, mean_judge)) - JUDGE_SCALE_MIN) / span


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


async def _run_case(
    case,
    config: PromptConfig,
    provider: LLMProvider,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    summary_pass_threshold: int,
) -> CaseResult:
    async with semaphore:
        try:
            call = await classify_email(case.email, config, provider, settings.model)
        except (ClassificationError, Exception) as exc:  # noqa: BLE001 - one bad case must not kill the run
            return CaseResult(
                case_id=case.id,
                expected_category=case.expected_category,
                predicted_category=None,
                category_match=False,
                passed=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
                difficulty=case.expected_difficulty,
            )

        match = case.accepts(call.output.category)
        assertion_failures = run_assertions(case, call.output, call.raw)
        try:
            score, reason = await judge_summary(
                case.email,
                case.expected_summary,
                call.output.summary,
                provider,
                settings.judge_model,
            )
        except Exception as exc:  # noqa: BLE001 - judge failure is not a quality regression
            score, reason = 3, f"{JUDGE_CALL_FAILED}: {type(exc).__name__}"

        return CaseResult(
            case_id=case.id,
            expected_category=case.expected_category,
            predicted_category=call.output.category,
            predicted_summary=call.output.summary,
            category_match=match,
            judge_score=score,
            judge_reason=reason,
            passed=match and score >= summary_pass_threshold and not has_hard_failure(assertion_failures),
            latency_ms=call.latency_ms,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            difficulty=case.expected_difficulty,
            assertion_failures=assertion_failures,
        )


async def run_eval(
    config: PromptConfig,
    dataset: GoldenDataset,
    provider: LLMProvider,
    settings: Settings,
    *,
    git_sha: str = "",
    summary_pass_threshold: int = DEFAULT_SUMMARY_PASS_THRESHOLD,
    run_id: str | None = None,
) -> tuple[RunSummary, list[CaseResult]]:
    semaphore = asyncio.Semaphore(max(1, settings.concurrency))
    results = await asyncio.gather(
        *(
            _run_case(case, config, provider, settings, semaphore, summary_pass_threshold)
            for case in dataset.cases
        )
    )
    summary = summarise(
        results,
        config=config,
        dataset=dataset,
        settings=settings,
        git_sha=git_sha,
        run_id=run_id,
    )
    return summary, list(results)


def summarise(
    results: list[CaseResult],
    *,
    config: PromptConfig,
    dataset: GoldenDataset,
    settings: Settings,
    git_sha: str = "",
    run_id: str | None = None,
) -> RunSummary:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    matches = sum(1 for r in results if r.category_match)
    judged = [r.judge_score for r in results if r.judge_score is not None]
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]

    category_accuracy = matches / total if total else 0.0
    mean_judge = statistics.fmean(judged) if judged else 0.0

    by_category: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_category[result.expected_category.value].append(result.category_match)
        by_difficulty[result.difficulty.value].append(result.passed)

    return RunSummary(
        run_id=run_id or uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC).isoformat(),
        prompt_version=config.version,
        dataset_version=dataset.dataset_version,
        model=settings.model,
        provider=settings.provider,
        git_sha=git_sha,
        total_cases=total,
        passed_cases=passed,
        pass_rate=passed / total if total else 0.0,
        category_accuracy=category_accuracy,
        mean_judge_score=mean_judge,
        composite_score=composite(category_accuracy, mean_judge),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        total_tokens=sum(r.total_tokens for r in results),
        error_count=sum(1 for r in results if r.error),
        per_category_accuracy={k: sum(v) / len(v) for k, v in sorted(by_category.items()) if v},
        per_difficulty_pass_rate={k: sum(v) / len(v) for k, v in sorted(by_difficulty.items()) if v},
    )
