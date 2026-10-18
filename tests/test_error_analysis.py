"""Failure taxonomy: bucketing failed cases into named, rankable failure modes."""

from __future__ import annotations

import pytest

from conftest import make_result
from regression_guard.assertions import (
    CHECK_NUMBERS_GROUNDED,
    CHECK_SINGLE_SENTENCE,
    CHECK_STRICT_JSON,
    Severity,
)
from regression_guard.error_analysis import FailureMode, analyse, render_markdown, render_text
from regression_guard.judge import JUDGE_CALL_FAILED
from regression_guard.models import AssertionFailure, Category


def hard(check: str, detail: str = "detail") -> AssertionFailure:
    return AssertionFailure(check=check, severity=Severity.HARD, detail=detail)


def soft(check: str, detail: str = "detail") -> AssertionFailure:
    return AssertionFailure(check=check, severity=Severity.SOFT, detail=detail)


def bucket(analysis, mode: FailureMode):
    return next(b for b in analysis.buckets if b.mode is mode)


@pytest.fixture
def mixed_results():
    return [
        make_result("ok-1", passed=True),
        make_result("ok-2", passed=True, assertion_failures=[soft(CHECK_STRICT_JSON)]),
        make_result("mis-1", passed=False, category=Category.TECHNICAL, judge_score=4),
        make_result("mis-2", passed=False, category=Category.TECHNICAL, judge_score=2),
        make_result("mis-3", passed=False, category=Category.ACCOUNT, judge_score=4),
        make_result("sum-1", passed=False, category_match=True, judge_score=2),
        make_result(
            "hal-1",
            passed=False,
            category_match=True,
            judge_score=5,
            assertion_failures=[hard(CHECK_NUMBERS_GROUNDED, "'2026' not in email")],
        ),
        make_result(
            "err-1",
            passed=False,
            category=None,
            category_match=False,
            judge_score=None,
            error="ClassificationError: no JSON object in response",
        ),
    ]


@pytest.mark.unit
def test_analyse_counts_failed_cases_not_failure_labels(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert analysis.total_cases == 8
    assert analysis.failed_cases == 6


@pytest.mark.unit
def test_buckets_are_ranked_by_count_descending(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    counts = [b.count for b in analysis.buckets]
    assert counts == sorted(counts, reverse=True)
    assert analysis.buckets[0].mode is FailureMode.MISROUTED
    assert bucket(analysis, FailureMode.MISROUTED).count == 3


@pytest.mark.unit
def test_a_case_can_land_in_more_than_one_bucket(mixed_results):
    """mis-2 is both misrouted and off-target. Shares are multi-label and
    therefore sum to more than 100%; the primary mode is single-label."""
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert "mis-2" in bucket(analysis, FailureMode.MISROUTED).case_ids
    assert "mis-2" in bucket(analysis, FailureMode.SUMMARY_OFF_TARGET).case_ids
    assert analysis.primary_mode["mis-2"] is FailureMode.MISROUTED


@pytest.mark.unit
def test_share_is_relative_to_failed_cases(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert bucket(analysis, FailureMode.MISROUTED).share_of_failures == pytest.approx(3 / 6)


@pytest.mark.unit
def test_output_contract_errors_get_their_own_bucket_and_no_misroute(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert bucket(analysis, FailureMode.OUTPUT_CONTRACT).case_ids == ["err-1"]
    assert "err-1" not in bucket(analysis, FailureMode.MISROUTED).case_ids
    assert analysis.primary_mode["err-1"] is FailureMode.OUTPUT_CONTRACT


@pytest.mark.unit
def test_hard_assertions_map_to_named_failure_modes(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    hallucinated = bucket(analysis, FailureMode.HALLUCINATED_NUMBER)
    assert hallucinated.case_ids == ["hal-1"]
    assert analysis.primary_mode["hal-1"] is FailureMode.HALLUCINATED_NUMBER


@pytest.mark.unit
def test_misroutes_are_broken_down_by_confusion_pair(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert analysis.misroutes == {"billing -> technical": 2, "billing -> account": 1}


@pytest.mark.unit
def test_soft_findings_are_reported_separately_and_include_passing_cases(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert [b.mode for b in analysis.soft_findings] == [FailureMode.NON_STRICT_JSON]
    assert analysis.soft_findings[0].case_ids == ["ok-2"]
    assert FailureMode.NON_STRICT_JSON not in {b.mode for b in analysis.buckets}


@pytest.mark.unit
def test_failed_judge_calls_are_flagged_as_infrastructure_not_quality():
    results = [
        make_result("j-1", passed=True, judge_score=3, judge_reason=f"{JUDGE_CALL_FAILED}: TimeoutError"),
    ]
    analysis = analyse(results, summary_pass_threshold=3)
    assert analysis.judge_unavailable == ["j-1"]
    assert analysis.failed_cases == 0


@pytest.mark.unit
def test_baseline_counts_produce_per_mode_deltas(mixed_results):
    baseline = [
        make_result("mis-1", passed=False, category=Category.TECHNICAL, judge_score=4),
        make_result("sum-1", passed=False, category_match=True, judge_score=2),
        make_result("sum-2", passed=False, category_match=True, judge_score=1),
    ]
    analysis = analyse(mixed_results, summary_pass_threshold=3, baseline_results=baseline)
    misrouted = bucket(analysis, FailureMode.MISROUTED)
    off_target = bucket(analysis, FailureMode.SUMMARY_OFF_TARGET)
    assert (misrouted.baseline_count, misrouted.delta) == (1, 2)
    assert (off_target.baseline_count, off_target.delta) == (2, 0)


@pytest.mark.unit
def test_a_mode_that_disappeared_since_baseline_is_still_listed():
    baseline = [make_result("sum-1", passed=False, category_match=True, judge_score=1)]
    analysis = analyse([make_result("ok", passed=True)], summary_pass_threshold=3, baseline_results=baseline)
    gone = bucket(analysis, FailureMode.SUMMARY_OFF_TARGET)
    assert (gone.count, gone.baseline_count, gone.delta) == (0, 1, -1)


@pytest.mark.unit
def test_without_baseline_deltas_are_none(mixed_results):
    analysis = analyse(mixed_results, summary_pass_threshold=3)
    assert all(b.delta is None and b.baseline_count is None for b in analysis.buckets)


@pytest.mark.unit
def test_all_passing_run_has_no_buckets():
    analysis = analyse([make_result("ok", passed=True)], summary_pass_threshold=3)
    assert analysis.failed_cases == 0
    assert analysis.buckets == []
    assert "No failed cases" in render_text(analysis)


@pytest.mark.unit
def test_examples_carry_the_assertion_detail_for_triage():
    results = [
        make_result(
            "fmt-1",
            passed=False,
            category_match=True,
            judge_score=4,
            assertion_failures=[hard(CHECK_SINGLE_SENTENCE, "2 sentences")],
        )
    ]
    analysis = analyse(results, summary_pass_threshold=3)
    example = bucket(analysis, FailureMode.FORMAT_VIOLATION).examples[0]
    assert example.case_id == "fmt-1"
    assert "2 sentences" in example.detail


@pytest.mark.unit
def test_markdown_render_lists_modes_and_deltas(mixed_results):
    baseline = [make_result("mis-1", passed=False, category=Category.TECHNICAL, judge_score=4)]
    analysis = analyse(mixed_results, summary_pass_threshold=3, baseline_results=baseline)
    markdown = render_markdown(analysis)
    assert "| Failure mode |" in markdown
    assert "misrouted" in markdown
    assert "+2" in markdown
    assert "billing -> technical" in markdown


@pytest.mark.unit
def test_text_render_mentions_the_top_mode(mixed_results):
    text = render_text(analyse(mixed_results, summary_pass_threshold=3))
    assert "misrouted" in text
    assert "6/8 cases failed" in text
