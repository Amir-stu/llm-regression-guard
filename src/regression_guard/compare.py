"""Run-to-run diffing, significance testing and verdict assignment.

This module is the reason the project exists. Everything upstream produces
numbers; this decides whether a change in those numbers should stop a merge.

Why McNemar and not a two-proportion z-test
-------------------------------------------
Both runs are scored on the *same* golden cases, so the two pass rates are
paired, not independent. A two-proportion test would ignore that pairing and
badly overstate the variance - it would let genuine 5-case regressions look
like noise. McNemar's exact test looks only at the cases that flipped, which
is precisely the quantity a reviewer cares about: 'of the cases that changed
behaviour, did they change in one direction more than a coin flip explains?'
"""

from __future__ import annotations

from math import comb

from .config import Thresholds
from .models import (
    CaseResult,
    Comparison,
    FlippedCase,
    GoldenCase,
    RunSummary,
    Verdict,
)


def mcnemar_exact_p(regressions: int, improvements: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs.

    Under the null (a change is equally likely to help or hurt), the number of
    regressions among ``n`` flipped cases is Binomial(n, 0.5).
    """
    n = regressions + improvements
    if n == 0:
        return 1.0
    k = min(regressions, improvements)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _flip(
    case_id: str,
    old: CaseResult,
    new: CaseResult,
    cases_by_id: dict[str, GoldenCase],
) -> FlippedCase:
    case = cases_by_id.get(case_id)
    return FlippedCase(
        case_id=case_id,
        difficulty=new.difficulty,
        expected_category=case.expected_category if case else new.expected_category,
        old_category=old.predicted_category,
        new_category=new.predicted_category,
        old_summary=old.predicted_summary,
        new_summary=new.predicted_summary,
        old_judge_score=old.judge_score,
        new_judge_score=new.judge_score,
    )


def compare_runs(
    current: RunSummary,
    current_results: list[CaseResult],
    baseline: RunSummary | None,
    baseline_results: list[CaseResult],
    thresholds: Thresholds,
    cases_by_id: dict[str, GoldenCase] | None = None,
) -> Comparison:
    cases_by_id = cases_by_id or {}
    comparison = Comparison(current=current, baseline=baseline)

    if baseline is None:
        comparison.verdict = Verdict.PASS
        comparison.reasons.append("No baseline run found. This run is now the baseline.")
        return comparison

    if baseline.dataset_version != current.dataset_version:
        comparison.reasons.append(
            f"Dataset version changed ({baseline.dataset_version} -> {current.dataset_version}). "
            "Deltas below mix prompt effects with dataset effects; treat them as indicative only."
        )

    comparison.pass_rate_delta = current.pass_rate - baseline.pass_rate
    comparison.category_accuracy_delta = current.category_accuracy - baseline.category_accuracy
    comparison.composite_delta = current.composite_score - baseline.composite_score
    comparison.per_category_delta = {
        category: current.per_category_accuracy.get(category, 0.0) - value
        for category, value in baseline.per_category_accuracy.items()
    }

    old_by_id = {r.case_id: r for r in baseline_results}
    for result in current_results:
        old = old_by_id.get(result.case_id)
        if old is None:
            continue  # New case: no pair, so it cannot be a flip.
        if old.passed and not result.passed:
            comparison.regressions.append(_flip(result.case_id, old, result, cases_by_id))
        elif not old.passed and result.passed:
            comparison.improvements.append(_flip(result.case_id, old, result, cases_by_id))

    comparison.p_value = mcnemar_exact_p(len(comparison.regressions), len(comparison.improvements))
    comparison.significant = comparison.p_value < thresholds.significance_alpha

    comparison.verdict, reasons = _decide(comparison, thresholds)
    comparison.reasons.extend(reasons)
    return comparison


def _decide(comparison: Comparison, thresholds: Thresholds) -> tuple[Verdict, list[str]]:
    """Map the diff onto pass / warn / critical.

    Two independent routes to CRITICAL:
      1. The pass-rate drop exceeds the critical threshold. Size alone is
         enough - an 8pp drop does not need a p-value to be worth blocking.
      2. The drop clears the (smaller) warn threshold *and* McNemar says the
         flip pattern is unlikely under chance. This catches a real but modest
         regression that a size-only rule would wave through.
    """
    delta = comparison.pass_rate_delta
    regressions = len(comparison.regressions)
    improvements = len(comparison.improvements)
    reasons: list[str] = []
    verdict = Verdict.PASS

    if delta <= -thresholds.critical_delta:
        verdict = Verdict.CRITICAL
        reasons.append(
            f"Pass rate fell {abs(delta):.1%}, at or beyond the critical threshold "
            f"of {thresholds.critical_delta:.0%}."
        )
    elif delta <= -thresholds.warn_delta:
        if comparison.significant:
            verdict = Verdict.CRITICAL
            reasons.append(
                f"Pass rate fell {abs(delta):.1%} and the flip pattern is statistically "
                f"significant (McNemar p={comparison.p_value:.3f} < "
                f"{thresholds.significance_alpha}), so this is signal, not noise."
            )
        else:
            verdict = Verdict.WARN
            reasons.append(
                f"Pass rate fell {abs(delta):.1%}, past the {thresholds.warn_delta:.0%} warning "
                f"threshold, but the flip pattern is within noise "
                f"(McNemar p={comparison.p_value:.3f}). Review before merging."
            )
    elif regressions and comparison.significant:
        verdict = Verdict.WARN
        reasons.append(
            f"Headline pass rate held ({delta:+.1%}) but {regressions} cases regressed against "
            f"{improvements} improvements, which is unlikely by chance "
            f"(McNemar p={comparison.p_value:.3f}). Behaviour shifted even though the "
            "aggregate did not."
        )
    elif regressions:
        reasons.append(
            f"{regressions} regressions and {improvements} improvements; net effect "
            f"{delta:+.1%} is within noise."
        )
    else:
        reasons.append(f"No regressions. Pass rate {delta:+.1%} versus baseline.")

    baseline = comparison.baseline
    if baseline is not None and comparison.current.error_count > baseline.error_count:
        reasons.append(
            f"Hard failures rose from {baseline.error_count} to "
            f"{comparison.current.error_count} (unparseable or errored responses)."
        )

    return verdict, reasons
