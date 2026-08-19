"""CLI behaviour, especially the exit codes CI depends on.

The exit code is the contract with GitHub Actions: get it wrong and the
pipeline either blocks everything or blocks nothing.
"""

from __future__ import annotations

import json

import pytest

from regression_guard.cli import EXIT_CRITICAL, EXIT_ERROR, EXIT_PASS, EXIT_WARN, main


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RG_PROVIDER", "mock")
    monkeypatch.setenv("RG_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("RG_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    return tmp_path


@pytest.mark.integration
def test_validate_accepts_the_shipped_dataset_and_prompts(cli_env, capsys):
    assert main(["validate"]) == EXIT_PASS
    assert "all good" in capsys.readouterr().out


@pytest.mark.integration
def test_validate_rejects_a_broken_dataset(cli_env, monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"dataset_version": "x", "created_at": "y", "cases": []}', encoding="utf-8")
    monkeypatch.setenv("RG_DATASET", str(bad))
    assert main(["validate"]) == EXIT_ERROR


@pytest.mark.integration
def test_first_run_establishes_a_baseline_and_passes(cli_env):
    assert main(["run", "--prompt", "v1-baseline", "--no-slack"]) == EXIT_PASS
    assert (cli_env / "reports" / "index.html").exists()


@pytest.mark.integration
def test_terse_prompt_exits_critical_and_blocks_the_merge(cli_env):
    assert main(["run", "--prompt", "v2-tightened", "--no-slack"]) == EXIT_PASS
    assert main(["run", "--prompt", "v3-terse", "--no-slack"]) == EXIT_CRITICAL


@pytest.mark.integration
def test_rerunning_a_failing_prompt_stays_red(cli_env):
    """Retrying a red build must not turn it green by making the bad run its
    own baseline."""
    main(["run", "--prompt", "v2-tightened", "--no-slack"])
    first = main(["run", "--prompt", "v3-terse", "--no-slack"])
    second = main(["run", "--prompt", "v3-terse", "--no-slack"])
    assert first == second == EXIT_CRITICAL


@pytest.mark.integration
def test_improvement_run_passes_against_a_worse_baseline(cli_env):
    main(["run", "--prompt", "v3-terse", "--no-slack"])
    assert main(["run", "--prompt", "v2-tightened", "--no-slack"]) == EXIT_PASS


@pytest.mark.integration
def test_run_writes_json_and_markdown_artifacts(cli_env):
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    json_path = cli_env / "out" / "comparison.json"
    md_path = cli_env / "out" / "summary.md"
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
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "critical"
    assert payload["regressions"]

    markdown = md_path.read_text(encoding="utf-8")
    assert "Prompt eval" in markdown
    assert "| Metric | Baseline | Current | Delta |" in markdown
    assert "Regressed cases" in markdown


@pytest.mark.integration
def test_explicit_baseline_run_id_is_honoured(cli_env, capsys):
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    capsys.readouterr()
    assert main(["run", "--prompt", "v2-tightened", "--no-slack", "--baseline", "nope"]) == EXIT_ERROR
    assert "no run with id" in capsys.readouterr().err


@pytest.mark.integration
def test_fail_on_warn_is_opt_in(cli_env, monkeypatch):
    """A team that wants warnings to block gets it with a flag, not a fork."""
    monkeypatch.setenv("RG_WARN_DELTA", "0.01")
    monkeypatch.setenv("RG_CRITICAL_DELTA", "0.90")
    monkeypatch.setenv("RG_SIGNIFICANCE_ALPHA", "0.0000001")
    main(["run", "--prompt", "v2-tightened", "--no-slack"])
    lenient = main(["run", "--prompt", "v3-terse", "--no-slack"])
    assert lenient == EXIT_PASS

    monkeypatch.setenv("RG_DB_PATH", str(cli_env / "runs2.db"))
    main(["run", "--prompt", "v2-tightened", "--no-slack"])
    strict = main(["run", "--prompt", "v3-terse", "--no-slack", "--fail-on-warn"])
    assert strict == EXIT_WARN


@pytest.mark.integration
def test_history_lists_runs_newest_first(cli_env, capsys):
    main(["run", "--prompt", "v1-baseline", "--no-slack"])
    main(["run", "--prompt", "v2-tightened", "--no-slack"])
    capsys.readouterr()
    assert main(["history", "--limit", "5"]) == EXIT_PASS
    out = capsys.readouterr().out
    assert out.index("v2-tightened") < out.index("v1-baseline")


@pytest.mark.integration
def test_history_on_an_empty_database(cli_env, capsys):
    assert main(["history"]) == EXIT_PASS
    assert "no runs recorded yet" in capsys.readouterr().out


@pytest.mark.integration
def test_openai_provider_without_key_or_package_fails_cleanly(cli_env, monkeypatch, capsys):
    """Selecting the real provider with nothing configured must exit 3 with an
    actionable message, not raise a traceback at the user. Which of the two
    causes fires depends on whether the optional `openai` extra is installed,
    so both are accepted - what matters is that the message points somewhere."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["run", "--provider", "openai", "--no-slack"]) == EXIT_ERROR
    stderr = capsys.readouterr().err
    assert "OPENAI_API_KEY" in stderr or "openai package is not installed" in stderr
    assert "RG_PROVIDER=mock" in stderr


@pytest.mark.integration
def test_unknown_provider_is_rejected(cli_env, capsys):
    assert main(["run", "--provider", "mock", "--no-slack"]) == EXIT_PASS
    from regression_guard.providers import build_provider

    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("anthropic-but-typoed")


@pytest.mark.integration
def test_unknown_prompt_raises_with_available_options(cli_env):
    with pytest.raises(FileNotFoundError, match="Available"):
        main(["run", "--prompt", "v99-nope", "--no-slack"])


@pytest.mark.integration
def test_slack_is_attempted_when_a_webhook_is_configured(cli_env, monkeypatch):
    sent: list[str] = []

    def fake_send(url, comparison, report_url=None, drift=None, timeout=10.0):
        sent.append(url)
        return True

    monkeypatch.setattr("regression_guard.cli.send_alert", fake_send)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/abc")
    monkeypatch.setenv("RG_REPORT_BASE_URL", "https://reports.test/evals/")
    assert main(["run", "--prompt", "v1-baseline"]) == EXIT_PASS
    assert sent == ["https://hooks.slack.test/abc"]
