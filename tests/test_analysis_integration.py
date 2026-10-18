"""Assertions and error analysis wired through runner, store, report and CLI."""

from __future__ import annotations

import json
import sqlite3

import pytest

from conftest import make_result, make_summary
from regression_guard.assertions import CHECK_REQUIRED_DETAIL, Severity
from regression_guard.cli import EXIT_ERROR, EXIT_PASS, main
from regression_guard.compare import compare_runs
from regression_guard.error_analysis import analyse
from regression_guard.models import AssertionFailure, Category, Difficulty, GoldenCase, GoldenDataset
from regression_guard.providers import MockProvider
from regression_guard.report import render_report
from regression_guard.runner import run_eval
from regression_guard.store import RunStore


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RG_PROVIDER", "mock")
    monkeypatch.setenv("RG_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("RG_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    return tmp_path


@pytest.mark.integration
async def test_hard_assertion_failure_fails_an_otherwise_correct_case(prompt_config, settings):
    """Right category and a good judge score are not enough when the summary
    drops an identifier the case requires."""
    case = GoldenCase(
        id="req-1",
        email="Please refund the duplicate invoice charge. Reference INV-77.",
        expected_category=Category.BILLING,
        expected_summary="Customer wants a refund for a duplicate invoice charge.",
        expected_difficulty=Difficulty.EASY,
        notes="requires the reference id",
        must_include=["INV-77"],
    )
    dataset = GoldenDataset(dataset_version="t", created_at="t", cases=[case])
    _, results = await run_eval(prompt_config, dataset, MockProvider(), settings)
    result = results[0]
    assert result.category_match is True
    assert result.judge_score is not None and result.judge_score >= 3
    assert [a.check for a in result.assertion_failures] == [CHECK_REQUIRED_DETAIL]
    assert result.passed is False


@pytest.mark.integration
async def test_clean_cases_carry_no_assertion_failures(prompt_config, dataset, settings):
    _, results = await run_eval(prompt_config, dataset, MockProvider(), settings)
    assert all(not r.assertion_failures for r in results)


@pytest.mark.integration
def test_store_roundtrips_assertion_failures(tmp_path):
    store = RunStore(tmp_path / "db.sqlite")
    failure = AssertionFailure(check=CHECK_REQUIRED_DETAIL, severity=Severity.HARD, detail="missing: X")
    store.save_run(make_summary(run_id="r1"), [make_result("c1", passed=False, assertion_failures=[failure])])
    loaded = store.get_case_results("r1")[0]
    assert loaded.assertion_failures == [failure]


@pytest.mark.integration
def test_store_migrates_a_database_written_before_assertions_existed(tmp_path):
    db_path = tmp_path / "old.sqlite"
    RunStore(db_path).save_run(make_summary(run_id="old"), [make_result("c1", passed=True)])
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE case_results DROP COLUMN assertion_failures")
    store = RunStore(db_path)
    assert store.get_case_results("old")[0].assertion_failures == []
    store.save_run(make_summary(run_id="new"), [make_result("c1", passed=True)])
    assert store.get_case_results("new")[0].case_id == "c1"


@pytest.mark.unit
def test_report_renders_the_failure_mode_section(thresholds):
    current = [make_result("c1", passed=False, category=Category.TECHNICAL, judge_score=4)]
    comparison = compare_runs(make_summary(run_id="curr"), current, None, [], thresholds)
    html = render_report(comparison, analysis=analyse(current, summary_pass_threshold=3))
    assert "Failure modes" in html
    assert "misrouted" in html
    assert "billing -&gt; technical" in html


@pytest.mark.unit
def test_report_escapes_assertion_details(thresholds):
    failure = AssertionFailure(check=CHECK_REQUIRED_DETAIL, severity=Severity.HARD, detail="<img src=x>")
    current = [
        make_result("c1", passed=False, category_match=True, judge_score=4, assertion_failures=[failure])
    ]
    comparison = compare_runs(make_summary(run_id="curr"), current, None, [], thresholds)
    html = render_report(comparison, analysis=analyse(current, summary_pass_threshold=3))
    assert "<img src=x>" not in html
    assert "&lt;img src=x&gt;" in html


@pytest.mark.integration
def test_run_prints_error_analysis_and_writes_it_to_artifacts(cli_env, capsys):
    json_path = cli_env / "out" / "comparison.json"
    md_path = cli_env / "out" / "summary.md"
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    main(
        [
            "run",
            "--prompt",
            "v3-terse",
            "--no-slack",
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(md_path),
        ]
    )
    assert "error analysis:" in capsys.readouterr().out
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["error_analysis"]["failed_cases"] > 0
    assert payload["error_analysis"]["buckets"][0]["delta"] is not None
    assert "#### Error analysis" in md_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_analyze_defaults_to_the_latest_run(cli_env, capsys):
    main(["run", "--prompt", "v3-terse", "--no-slack"])
    capsys.readouterr()
    assert main(["analyze"]) == EXIT_PASS
    out = capsys.readouterr().out
    assert "error analysis:" in out
    assert "v3-terse" in out


@pytest.mark.integration
def test_analyze_writes_markdown_and_json_against_an_explicit_baseline(cli_env, capsys):
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    main(["run", "--prompt", "v3-terse", "--no-slack"])
    runs = RunStore(cli_env / "runs.db").recent_runs()
    current, baseline = runs[0].run_id, runs[1].run_id
    md_path = cli_env / "analysis.md"
    json_path = cli_env / "analysis.json"
    code = main(
        [
            "analyze",
            "--run-id",
            current,
            "--baseline",
            baseline,
            "--markdown-out",
            str(md_path),
            "--json-out",
            str(json_path),
        ]
    )
    assert code == EXIT_PASS
    assert "vs baseline" in md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["buckets"][0]["baseline_count"] is not None


@pytest.mark.integration
def test_analyze_on_an_empty_database_says_so(cli_env, capsys):
    assert main(["analyze"]) == EXIT_PASS
    assert "no runs recorded yet" in capsys.readouterr().out


@pytest.mark.integration
def test_analyze_rejects_an_unknown_run_id(cli_env, capsys):
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    assert main(["analyze", "--run-id", "nope"]) == EXIT_ERROR
    assert "no run with id" in capsys.readouterr().err


@pytest.mark.integration
def test_analyze_rejects_an_unknown_baseline(cli_env, capsys):
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    assert main(["analyze", "--baseline", "nope"]) == EXIT_ERROR
    assert "no run with id" in capsys.readouterr().err
