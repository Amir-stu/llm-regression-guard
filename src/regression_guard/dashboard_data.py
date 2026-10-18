"""Data shaping for the Streamlit dashboard.

Kept free of Streamlit and pandas so the logic is unit-tested with the rest of
the package and the core install stays at two dependencies. The dashboard
script turns these row lists into tables and charts and does nothing else.
"""

from __future__ import annotations

from typing import Any

from .error_analysis import ErrorAnalysis, analyse
from .models import CaseResult, RunSummary

STATUS_REGRESSED = "regressed"
STATUS_IMPROVED = "improved"
STATUS_STILL_FAILING = "still failing"
STATUS_STILL_PASSING = "still passing"
STATUS_NEW = "new case"

_STATUS_ORDER = {
    STATUS_REGRESSED: 0,
    STATUS_IMPROVED: 1,
    STATUS_STILL_FAILING: 2,
    STATUS_NEW: 3,
    STATUS_STILL_PASSING: 4,
}

Row = dict[str, Any]


def pass_rate_trend(runs: list[RunSummary]) -> list[Row]:
    """One row per run, oldest first, for the trend chart."""
    return [
        {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "prompt_version": run.prompt_version,
            "model": run.model,
            "pass_rate": run.pass_rate,
            "category_accuracy": run.category_accuracy,
            "composite_score": run.composite_score,
        }
        for run in sorted(runs, key=lambda run: run.created_at)
    ]


def failure_mode_trend(
    history: list[tuple[RunSummary, list[CaseResult]]],
    *,
    summary_pass_threshold: int,
) -> list[Row]:
    """Long-format failure-mode counts per run, oldest first."""
    ordered = sorted(history, key=lambda pair: pair[0].created_at)
    return [
        {
            "run_id": run.run_id,
            "prompt_version": run.prompt_version,
            "mode": bucket.mode.value,
            "count": bucket.count,
        }
        for run, results in ordered
        for bucket in analyse(results, summary_pass_threshold=summary_pass_threshold).buckets
    ]


def _status(current: CaseResult, baseline: CaseResult | None) -> str:
    if baseline is None:
        return STATUS_NEW
    if baseline.passed and not current.passed:
        return STATUS_REGRESSED
    if not baseline.passed and current.passed:
        return STATUS_IMPROVED
    return STATUS_STILL_PASSING if current.passed else STATUS_STILL_FAILING


def _category(result: CaseResult | None) -> str | None:
    if result is None:
        return None
    return result.predicted_category.value if result.predicted_category else "error"


def side_by_side(
    current: list[CaseResult],
    baseline: list[CaseResult],
    *,
    summary_pass_threshold: int,
) -> list[Row]:
    """Pair each current case with its baseline result, regressions first."""
    primary = analyse(current, summary_pass_threshold=summary_pass_threshold).primary_mode
    baseline_by_id = {result.case_id: result for result in baseline}
    rows = [
        {
            "case_id": result.case_id,
            "status": _status(result, baseline_by_id.get(result.case_id)),
            "expected": result.expected_category.value,
            "baseline_category": _category(baseline_by_id.get(result.case_id)),
            "current_category": _category(result),
            "baseline_judge": getattr(baseline_by_id.get(result.case_id), "judge_score", None),
            "current_judge": result.judge_score,
            "failure_mode": primary[result.case_id].value if result.case_id in primary else "",
            "baseline_summary": getattr(baseline_by_id.get(result.case_id), "predicted_summary", ""),
            "current_summary": result.predicted_summary,
        }
        for result in current
    ]
    return sorted(rows, key=lambda row: (_STATUS_ORDER[row["status"]], row["case_id"]))


def case_log_rows(results: list[CaseResult], analysis: ErrorAnalysis) -> list[Row]:
    """The raw per-case log, annotated with the primary failure mode."""
    return [
        {
            "case_id": result.case_id,
            "passed": result.passed,
            "primary_mode": analysis.primary_mode[result.case_id].value
            if result.case_id in analysis.primary_mode
            else "",
            "expected": result.expected_category.value,
            "predicted": _category(result),
            "judge_score": result.judge_score,
            "assertions": "; ".join(
                f"{a.check} ({a.severity.value}): {a.detail}" for a in result.assertion_failures
            ),
            "summary": result.predicted_summary,
            "error": result.error or "",
        }
        for result in results
    ]
