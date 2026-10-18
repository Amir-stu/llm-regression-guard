"""Data shaping behind the Streamlit dashboard. Pure functions, no Streamlit."""

from __future__ import annotations

import pytest

from conftest import make_result, make_summary
from regression_guard.assertions import CHECK_NUMBERS_GROUNDED, Severity
from regression_guard.dashboard_data import (
    STATUS_IMPROVED,
    STATUS_NEW,
    STATUS_REGRESSED,
    STATUS_STILL_FAILING,
    STATUS_STILL_PASSING,
    case_log_rows,
    failure_mode_trend,
    pass_rate_trend,
    side_by_side,
)
from regression_guard.error_analysis import FailureMode, analyse
from regression_guard.models import AssertionFailure, Category


@pytest.mark.unit
def test_pass_rate_trend_is_oldest_first():
    runs = [
        make_summary(run_id="new", created_at="2026-10-02T00:00:00+00:00", pass_rate=0.8),
        make_summary(run_id="old", created_at="2026-10-01T00:00:00+00:00", pass_rate=0.7),
    ]
    rows = pass_rate_trend(runs)
    assert [r["run_id"] for r in rows] == ["old", "new"]
    assert rows[0]["pass_rate"] == 0.7
    assert {"prompt_version", "composite_score", "created_at"} <= rows[0].keys()


@pytest.mark.unit
def test_pass_rate_trend_of_no_runs_is_empty():
    assert pass_rate_trend([]) == []


@pytest.mark.unit
def test_failure_mode_trend_has_one_row_per_run_and_mode():
    first = make_summary(run_id="r1", created_at="2026-10-01T00:00:00+00:00")
    second = make_summary(run_id="r2", created_at="2026-10-02T00:00:00+00:00")
    history = [
        (second, [make_result("a", passed=False, category=Category.TECHNICAL, judge_score=4)]),
        (first, [make_result("a", passed=True)]),
    ]
    rows = failure_mode_trend(history, summary_pass_threshold=3)
    assert rows == [{"run_id": "r2", "prompt_version": "v1", "mode": "misrouted", "count": 1}]


@pytest.mark.unit
def test_side_by_side_labels_every_transition():
    baseline = [
        make_result("reg", passed=True),
        make_result("imp", passed=False, category=Category.TECHNICAL),
        make_result("bad", passed=False, category=Category.TECHNICAL),
        make_result("good", passed=True),
    ]
    current = [
        make_result("reg", passed=False, category=Category.ACCOUNT, summary="new summary"),
        make_result("imp", passed=True),
        make_result("bad", passed=False, category=Category.TECHNICAL),
        make_result("good", passed=True),
        make_result("fresh", passed=True),
    ]
    rows = {r["case_id"]: r for r in side_by_side(current, baseline, summary_pass_threshold=3)}
    assert rows["reg"]["status"] == STATUS_REGRESSED
    assert rows["imp"]["status"] == STATUS_IMPROVED
    assert rows["bad"]["status"] == STATUS_STILL_FAILING
    assert rows["good"]["status"] == STATUS_STILL_PASSING
    assert rows["fresh"]["status"] == STATUS_NEW
    assert rows["reg"]["current_summary"] == "new summary"
    assert rows["reg"]["baseline_category"] == "billing"
    assert rows["reg"]["current_category"] == "account"
    assert rows["fresh"]["baseline_category"] is None


@pytest.mark.unit
def test_side_by_side_puts_regressions_first():
    baseline = [make_result("a", passed=True), make_result("b", passed=True)]
    current = [make_result("a", passed=True), make_result("b", passed=False, category=Category.ACCOUNT)]
    rows = side_by_side(current, baseline, summary_pass_threshold=3)
    assert rows[0]["case_id"] == "b"


@pytest.mark.unit
def test_case_log_rows_carry_primary_mode_and_assertion_detail():
    results = [
        make_result("ok", passed=True),
        make_result(
            "hal",
            passed=False,
            category_match=True,
            judge_score=5,
            assertion_failures=[
                AssertionFailure(
                    check=CHECK_NUMBERS_GROUNDED, severity=Severity.HARD, detail="'7' not in email"
                )
            ],
        ),
    ]
    analysis = analyse(results, summary_pass_threshold=3)
    rows = {r["case_id"]: r for r in case_log_rows(results, analysis)}
    assert rows["ok"]["primary_mode"] == ""
    assert rows["hal"]["primary_mode"] == FailureMode.HALLUCINATED_NUMBER.value
    assert "'7' not in email" in rows["hal"]["assertions"]
    assert rows["hal"]["passed"] is False
