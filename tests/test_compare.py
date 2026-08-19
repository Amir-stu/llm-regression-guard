"""The verdict logic is the part of this system that can block a merge, so it
gets the most exhaustive tests: every threshold boundary, both routes to
CRITICAL, and the significance maths against values computed by hand."""

from __future__ import annotations

import pytest

from conftest import make_result, make_summary
from regression_guard.compare import compare_runs, mcnemar_exact_p
from regression_guard.models import Category, Verdict


@pytest.mark.unit
@pytest.mark.parametrize(
    ("regressions", "improvements", "expected"),
    [
        (0, 0, 1.0),
        (1, 1, 1.0),
        (1, 0, 1.0),
        (2, 0, 0.5),
        (3, 0, 0.25),
        (4, 0, 0.125),
        (5, 0, 0.0625),
        (6, 0, 0.03125),
        (0, 6, 0.03125),
        (7, 1, 0.0703125),
    ],
)
def test_mcnemar_exact_p_matches_hand_computed_binomial(regressions, improvements, expected):
    assert mcnemar_exact_p(regressions, improvements) == pytest.approx(expected)


@pytest.mark.unit
def test_mcnemar_is_symmetric():
    assert mcnemar_exact_p(9, 2) == mcnemar_exact_p(2, 9)


@pytest.mark.unit
def test_mcnemar_never_exceeds_one():
    for regressions in range(0, 12):
        for improvements in range(0, 12):
            assert 0.0 <= mcnemar_exact_p(regressions, improvements) <= 1.0


@pytest.mark.unit
def test_no_baseline_passes_and_says_so(thresholds):
    comparison = compare_runs(make_summary(), [], None, [], thresholds)
    assert comparison.verdict is Verdict.PASS
    assert "baseline" in comparison.reasons[0].lower()
    assert comparison.p_value is None


@pytest.mark.unit
def test_identical_runs_report_no_change(thresholds):
    results = [make_result(f"c{i}", passed=True) for i in range(20)]
    baseline = make_summary(run_id="base", pass_rate=1.0, passed_cases=20, total_cases=20)
    current = make_summary(run_id="curr", pass_rate=1.0, passed_cases=20, total_cases=20)
    comparison = compare_runs(current, results, baseline, results, thresholds)
    assert comparison.verdict is Verdict.PASS
    assert comparison.regressions == []
    assert comparison.improvements == []
    assert comparison.p_value == 1.0


@pytest.mark.unit
def test_large_drop_is_critical_on_size_alone(thresholds):
    baseline_results = [make_result(f"c{i}", passed=True) for i in range(100)]
    current_results = [make_result(f"c{i}", passed=i >= 10) for i in range(100)]
    baseline = make_summary(run_id="base", pass_rate=1.0, passed_cases=100, total_cases=100)
    current = make_summary(run_id="curr", pass_rate=0.90, passed_cases=90, total_cases=100)
    comparison = compare_runs(current, current_results, baseline, baseline_results, thresholds)
    assert comparison.verdict is Verdict.CRITICAL
    assert len(comparison.regressions) == 10
    assert "critical threshold" in comparison.reasons[0]


@pytest.mark.unit
def test_moderate_drop_with_significance_escalates_to_critical(thresholds):
    """5pp drop sits between warn (3pp) and critical (8pp).

    Five one-directional flips give McNemar p=0.0625 > 0.05, so this stays a
    WARN; six flips give p=0.03125 and must escalate.
    """
    baseline_results = [make_result(f"c{i}", passed=True) for i in range(100)]
    five_worse = [make_result(f"c{i}", passed=i >= 5) for i in range(100)]
    baseline = make_summary(run_id="base", pass_rate=1.0, passed_cases=100, total_cases=100)
    current = make_summary(run_id="curr", pass_rate=0.95, passed_cases=95, total_cases=100)
    warn = compare_runs(current, five_worse, baseline, baseline_results, thresholds)
    assert warn.verdict is Verdict.WARN
    assert warn.significant is False

    six_worse = [make_result(f"c{i}", passed=i >= 6) for i in range(100)]
    current_six = make_summary(run_id="curr2", pass_rate=0.94, passed_cases=94, total_cases=100)
    critical = compare_runs(current_six, six_worse, baseline, baseline_results, thresholds)
    assert critical.verdict is Verdict.CRITICAL
    assert critical.significant is True


@pytest.mark.unit
def test_flat_pass_rate_with_significant_churn_warns(thresholds):
    """Six cases break and six recover: the headline is unchanged.

    A size-only rule sees nothing. The flip pattern is balanced so McNemar is
    not significant either, meaning this must stay PASS rather than WARN - the
    test pins that the churn path does not fire on balanced churn.
    """
    baseline_results = [make_result(f"c{i}", passed=i < 50) for i in range(100)]
    current_results = [make_result(f"c{i}", passed=(50 <= i < 100)) for i in range(100)]
    baseline = make_summary(run_id="base", pass_rate=0.50, passed_cases=50, total_cases=100)
    current = make_summary(run_id="curr", pass_rate=0.50, passed_cases=50, total_cases=100)
    comparison = compare_runs(current, current_results, baseline, baseline_results, thresholds)
    assert len(comparison.regressions) == 50
    assert len(comparison.improvements) == 50
    assert comparison.significant is False
    assert comparison.verdict is Verdict.PASS


@pytest.mark.unit
def test_lopsided_churn_at_flat_pass_rate_warns(thresholds):
    """Pass rate identical, but nine regressions against one improvement.

    This is the case the whole significance layer exists for: the aggregate
    says nothing changed, the flip pattern says behaviour moved. Nine-versus-one
    gives McNemar p=0.0215, inside alpha; eight-versus-two gives p=0.109 and
    correctly stays quiet.
    """
    baseline_results = [make_result(f"c{i}", passed=i < 50) for i in range(100)]

    def current_with(regressions: int, improvements: int):
        out = [make_result(f"c{i}", passed=i < 50) for i in range(100)]
        for i in range(regressions):
            out[i] = make_result(f"c{i}", passed=False)
        for i in range(50, 50 + improvements):
            out[i] = make_result(f"c{i}", passed=True)
        return out

    baseline_summary = make_summary(run_id="base", pass_rate=0.50, passed_cases=50, total_cases=100)
    current_summary = make_summary(run_id="curr", pass_rate=0.50, passed_cases=50, total_cases=100)

    loud = compare_runs(
        current_summary, current_with(9, 1), baseline_summary, baseline_results, thresholds
    )
    assert (len(loud.regressions), len(loud.improvements)) == (9, 1)
    assert loud.significant is True
    assert loud.verdict is Verdict.WARN

    quiet = compare_runs(
        current_summary, current_with(8, 2), baseline_summary, baseline_results, thresholds
    )
    assert quiet.significant is False
    assert quiet.verdict is Verdict.PASS


@pytest.mark.unit
def test_improvement_only_passes(thresholds):
    baseline_results = [make_result(f"c{i}", passed=False) for i in range(20)]
    current_results = [make_result(f"c{i}", passed=True) for i in range(20)]
    baseline = make_summary(run_id="base", pass_rate=0.0, passed_cases=0, total_cases=20)
    current = make_summary(run_id="curr", pass_rate=1.0, passed_cases=20, total_cases=20)
    comparison = compare_runs(current, current_results, baseline, baseline_results, thresholds)
    assert comparison.verdict is Verdict.PASS
    assert len(comparison.improvements) == 20
    assert comparison.regressions == []


@pytest.mark.unit
def test_new_cases_are_not_counted_as_flips(thresholds):
    """A case with no baseline pair cannot be a regression by construction."""
    baseline_results = [make_result("c1", passed=True)]
    current_results = [make_result("c1", passed=True), make_result("c2-new", passed=False)]
    baseline = make_summary(run_id="base", pass_rate=1.0, passed_cases=1, total_cases=1)
    current = make_summary(run_id="curr", pass_rate=0.5, passed_cases=1, total_cases=2)
    comparison = compare_runs(current, current_results, baseline, baseline_results, thresholds)
    assert comparison.regressions == []


@pytest.mark.unit
def test_dataset_version_change_is_flagged_in_reasons(thresholds):
    baseline = make_summary(run_id="base", dataset_version="d1")
    current = make_summary(run_id="curr", dataset_version="d2")
    comparison = compare_runs(current, [], baseline, [], thresholds)
    assert any("Dataset version changed" in r for r in comparison.reasons)


@pytest.mark.unit
def test_rising_error_count_is_reported(thresholds):
    baseline = make_summary(run_id="base", error_count=0)
    current = make_summary(run_id="curr", error_count=4)
    comparison = compare_runs(current, [], baseline, [], thresholds)
    assert any("Hard failures rose" in r for r in comparison.reasons)


@pytest.mark.unit
def test_flip_record_carries_both_outputs(thresholds):
    old = make_result("c1", passed=True, category=Category.BILLING, summary="old text", judge_score=5)
    new = make_result("c1", passed=False, category=Category.GENERAL, summary="new text", judge_score=2)
    baseline = make_summary(run_id="base", pass_rate=1.0, passed_cases=1, total_cases=1)
    current = make_summary(run_id="curr", pass_rate=0.0, passed_cases=0, total_cases=1)
    comparison = compare_runs(current, [new], baseline, [old], thresholds)
    flip = comparison.regressions[0]
    assert flip.old_summary == "old text"
    assert flip.new_summary == "new text"
    assert flip.old_category is Category.BILLING
    assert flip.new_category is Category.GENERAL
    assert flip.old_judge_score == 5 and flip.new_judge_score == 2


@pytest.mark.unit
def test_thresholds_are_configurable(thresholds):
    """A team that wants a stricter bar gets it by changing config, not code."""
    from dataclasses import replace

    strict = replace(thresholds, critical_delta=0.01, warn_delta=0.005)
    baseline_results = [make_result(f"c{i}", passed=True) for i in range(100)]
    current_results = [make_result(f"c{i}", passed=i >= 2) for i in range(100)]
    baseline = make_summary(run_id="base", pass_rate=1.0, passed_cases=100, total_cases=100)
    current = make_summary(run_id="curr", pass_rate=0.98, passed_cases=98, total_cases=100)
    assert compare_runs(current, current_results, baseline, baseline_results, strict).verdict is (
        Verdict.CRITICAL
    )
    assert compare_runs(
        current, current_results, baseline, baseline_results, thresholds
    ).verdict is Verdict.PASS
