"""Runner, classifier, judge, store, drift, report and Slack payload."""

from __future__ import annotations

import json

import pytest

from conftest import make_result, make_summary
from regression_guard.classifier import (
    ClassificationError,
    build_messages,
    classify_email,
    parse_output,
)
from regression_guard.compare import compare_runs
from regression_guard.config import Thresholds
from regression_guard.drift import detect_drift
from regression_guard.judge import parse_judge_response
from regression_guard.models import Category, Verdict
from regression_guard.prompts import load_prompt, resolve_prompt
from regression_guard.providers import MockProvider
from regression_guard.report import render_report
from regression_guard.runner import composite, normalise_judge, run_eval
from regression_guard.slack import build_payload, send_alert
from regression_guard.store import RunStore

# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_output_accepts_clean_json():
    out = parse_output('{"category": "billing", "summary": "Wants a refund."}')
    assert out.category is Category.BILLING
    assert out.summary == "Wants a refund."


@pytest.mark.unit
def test_parse_output_salvages_json_wrapped_in_prose():
    """Real models pad JSON with prose. Failing on that would make the eval
    measure formatting luck instead of classification quality."""
    raw = 'Sure!\n```json\n{"category": "technical", "summary": "It crashes."}\n```\nHope that helps.'
    assert parse_output(raw).category is Category.TECHNICAL


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "no json here at all",
        '{"category": "invoices", "summary": "x"}',
        '{"category": "billing"}',
        '{"category": "billing", "summary": ""}',
        '["billing", "x"]',
    ],
)
def test_parse_output_rejects_broken_shapes(raw):
    with pytest.raises(ClassificationError):
        parse_output(raw)


@pytest.mark.unit
def test_build_messages_interleaves_few_shot_pairs(prompt_config):
    from regression_guard.models import FewShotExample

    prompt_config.few_shot = [
        FewShotExample(email="a", category=Category.BILLING, summary="s."),
        FewShotExample(email="b", category=Category.TECHNICAL, summary="t."),
    ]
    messages = build_messages(prompt_config, "real email")
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"].strip() == "real email"
    assert json.loads(messages[1]["content"])["category"] == "billing"


@pytest.mark.unit
async def test_classify_email_returns_usage_and_latency(prompt_config):
    call = await classify_email(
        "Please refund the duplicate invoice charge.", prompt_config, MockProvider(), "test-model"
    )
    assert call.output.category is Category.BILLING
    assert call.latency_ms >= 0
    assert call.prompt_tokens > 0 and call.completion_tokens > 0


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"score": 5, "reason": "good"}', 5),
        ('{"score": 1, "reason": ""}', 1),
        ('here you go: {"score": 4}', 4),
        ("score: 2", 2),
    ],
)
def test_parse_judge_response_extracts_score(raw, expected):
    assert parse_judge_response(raw)[0] == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", "I refuse", '{"score": 9}', '{"verdict": "good"}'])
def test_unparseable_judge_lands_neutral_not_failing(raw):
    """A broken judge is an infrastructure problem. It must not be able to
    fail the build on its own, so it scores mid-scale and stays visible."""
    score, reason = parse_judge_response(raw)
    assert score == 3
    assert "unparseable" in reason


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_extracts_cue_table_from_prompt():
    table = MockProvider.extract_keyword_table(
        "- billing: invoice, refund\n- technical: crash\nunrelated line\n"
    )
    assert "invoice" in table["billing"]
    assert "crash" in table["technical"]


@pytest.mark.unit
def test_mock_falls_back_to_defaults_for_a_cueless_prompt():
    table = MockProvider.extract_keyword_table("Classify the email. Reply with JSON.")
    assert table == MockProvider.extract_keyword_table("")


@pytest.mark.unit
async def test_mock_is_deterministic(prompt_config):
    provider = MockProvider()
    first = await classify_email("Refund my invoice please.", prompt_config, provider, "m")
    second = await classify_email("Refund my invoice please.", prompt_config, provider, "m")
    assert first.output == second.output


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("score", "expected"), [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0), (None, 0.0)]
)
def test_normalise_judge_maps_scale_to_unit_interval(score, expected):
    assert normalise_judge(score) == pytest.approx(expected)


@pytest.mark.unit
def test_composite_weights_category_above_summary():
    """Category is objective, the judge is not, so category must dominate."""
    category_only = composite(1.0, 1.0)
    summary_only = composite(0.0, 5.0)
    assert category_only > summary_only
    assert composite(1.0, 5.0) == pytest.approx(1.0)
    assert composite(0.0, 1.0) == pytest.approx(0.0)


@pytest.mark.integration
async def test_run_eval_scores_every_case(prompt_config, dataset, settings):
    summary, results = await run_eval(prompt_config, dataset, MockProvider(), settings)
    assert len(results) == len(dataset.cases)
    assert summary.total_cases == 3
    assert 0.0 <= summary.pass_rate <= 1.0
    assert summary.per_category_accuracy
    assert summary.error_count == 0


@pytest.mark.integration
async def test_one_broken_case_does_not_kill_the_run(prompt_config, dataset, settings):
    class ExplodingProvider(MockProvider):
        async def complete(self, system, messages, **kwargs):
            if "crashes" in messages[-1]["content"]:
                raise RuntimeError("boom")
            return await super().complete(system, messages, **kwargs)

    summary, results = await run_eval(prompt_config, dataset, ExplodingProvider(), settings)
    assert summary.error_count == 1
    assert len(results) == 3
    failed = next(r for r in results if r.error)
    assert failed.passed is False
    assert "boom" in failed.error


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_store_roundtrips_a_run(tmp_path):
    store = RunStore(tmp_path / "db.sqlite")
    summary = make_summary(run_id="r1", per_category_accuracy={"billing": 0.9})
    results = [make_result("c1", passed=True), make_result("c2", passed=False)]
    store.save_run(summary, results)

    loaded = store.get_run("r1")
    assert loaded is not None
    assert loaded.per_category_accuracy == {"billing": 0.9}
    cases = store.get_case_results("r1")
    assert {c.case_id for c in cases} == {"c1", "c2"}
    assert cases[0].passed is True and cases[1].passed is False


@pytest.mark.integration
def test_baseline_is_the_last_run_of_a_different_config(tmp_path):
    """Re-running a failing prompt must not let it become its own baseline,
    otherwise pressing retry turns a red build green."""
    store = RunStore(tmp_path / "db.sqlite")
    store.save_run(make_summary(run_id="v1", prompt_version="v1", created_at="2026-01-01T00:00:00Z"), [])
    store.save_run(make_summary(run_id="v2a", prompt_version="v2", created_at="2026-01-02T00:00:00Z"), [])
    store.save_run(make_summary(run_id="v2b", prompt_version="v2", created_at="2026-01-03T00:00:00Z"), [])

    baseline = store.latest_baseline(prompt_version="v2", model="gpt-4o-mini")
    assert baseline is not None and baseline.run_id == "v1"


@pytest.mark.integration
def test_baseline_falls_back_when_only_one_config_exists(tmp_path):
    store = RunStore(tmp_path / "db.sqlite")
    assert store.latest_baseline(prompt_version="v1", model="m") is None
    store.save_run(make_summary(run_id="only", prompt_version="v1"), [])
    baseline = store.latest_baseline(prompt_version="v1", model="gpt-4o-mini")
    assert baseline is not None and baseline.run_id == "only"


@pytest.mark.integration
def test_recent_runs_orders_newest_first_and_filters(tmp_path):
    store = RunStore(tmp_path / "db.sqlite")
    for i in range(5):
        store.save_run(
            make_summary(
                run_id=f"r{i}",
                prompt_version="a" if i % 2 else "b",
                created_at=f"2026-01-0{i + 1}T00:00:00Z",
            ),
            [],
        )
    assert [r.run_id for r in store.recent_runs(limit=3)] == ["r4", "r3", "r2"]
    assert all(r.prompt_version == "a" for r in store.recent_runs(prompt_version="a"))


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_drift_stays_quiet_without_enough_history(thresholds):
    runs = [make_summary(run_id=f"r{i}", composite_score=0.5) for i in range(3)]
    report = detect_drift(runs, thresholds)
    assert report.triggered is False
    assert report.message is None


@pytest.mark.unit
def test_drift_fires_when_rolling_mean_falls_below_floor(thresholds):
    runs = [make_summary(run_id=f"r{i}", composite_score=0.80) for i in range(8)]
    report = detect_drift(runs, thresholds)
    assert report.below_floor is True
    assert "below the" in report.message


@pytest.mark.unit
def test_drift_stays_quiet_on_a_healthy_flat_series(thresholds):
    runs = [make_summary(run_id=f"r{i}", composite_score=0.93) for i in range(10)]
    assert detect_drift(runs, thresholds).triggered is False


@pytest.mark.unit
def test_drift_catches_a_gradual_slide_no_single_run_would_flag(thresholds):
    """One point of composite score lost per run. Every consecutive delta is
    far inside the per-run thresholds, so only the rolling view sees it."""
    scores = [0.960, 0.955, 0.950, 0.945, 0.940, 0.935, 0.930, 0.925, 0.920]
    runs = [make_summary(run_id=f"r{i}", composite_score=s) for i, s in enumerate(scores)]
    newest_first = list(reversed(runs))
    report = detect_drift(newest_first, thresholds)
    assert report.falling is True
    assert report.below_floor is False
    assert "trending down" in report.message


@pytest.mark.unit
def test_drift_ignores_a_single_noisy_dip(thresholds):
    scores = [0.95] * 6 + [0.93] + [0.95] * 2
    runs = [make_summary(run_id=f"r{i}", composite_score=s) for i, s in enumerate(scores)]
    assert detect_drift(list(reversed(runs)), thresholds).triggered is False


# --------------------------------------------------------------------------
# slack
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_slack_payload_carries_headline_numbers(thresholds):
    baseline_results = [make_result(f"c{i}", passed=True) for i in range(100)]
    current_results = [make_result(f"c{i}", passed=i >= 10) for i in range(100)]
    comparison = compare_runs(
        make_summary(run_id="curr", pass_rate=0.90),
        current_results,
        make_summary(run_id="base", pass_rate=1.00),
        baseline_results,
        thresholds,
    )
    payload = build_payload(comparison, report_url="https://example.test/report.html")
    body = json.dumps(payload)
    assert comparison.verdict is Verdict.CRITICAL
    assert "10 regressions" in body
    assert "100.0% to 90.0%" in body
    assert "https://example.test/report.html" in body


@pytest.mark.unit
def test_slack_payload_omits_button_without_a_url(thresholds):
    comparison = compare_runs(make_summary(), [], None, [], thresholds)
    payload = build_payload(comparison)
    assert not any(block["type"] == "actions" for block in payload["blocks"])


@pytest.mark.unit
def test_send_alert_without_webhook_is_a_no_op(thresholds):
    comparison = compare_runs(make_summary(), [], None, [], thresholds)
    assert send_alert(None, comparison) is False


@pytest.mark.unit
def test_send_alert_refuses_a_non_https_webhook(thresholds, monkeypatch):
    """The webhook comes from the environment. If it were ever set to a file://
    or custom scheme, urlopen would follow it, so the scheme is pinned."""
    comparison = compare_runs(make_summary(), [], None, [], thresholds)
    called: list[str] = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append("x"))
    assert send_alert("http://hooks.slack.test/x", comparison) is False
    assert send_alert("file:///etc/passwd", comparison) is False
    assert called == []


@pytest.mark.unit
def test_send_alert_swallows_network_failure(thresholds, monkeypatch):
    """An unreachable Slack must never turn a passing eval into a failed build."""
    comparison = compare_runs(make_summary(), [], None, [], thresholds)

    def explode(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    assert send_alert("https://hooks.slack.test/x", comparison) is False


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_report_renders_scorecard_and_regressions(thresholds):
    baseline_results = [make_result(f"c{i}", passed=True) for i in range(10)]
    current_results = [make_result(f"c{i}", passed=i >= 2) for i in range(10)]
    comparison = compare_runs(
        make_summary(run_id="curr", pass_rate=0.80),
        current_results,
        make_summary(run_id="base", pass_rate=1.00),
        baseline_results,
        thresholds,
    )
    html = render_report(comparison, history=[make_summary(run_id=f"h{i}") for i in range(4)])
    assert "Eval diff report" in html
    assert "CRITICAL" in html
    assert "<svg" in html
    assert html.count("<tr>") >= 2


@pytest.mark.unit
def test_report_escapes_model_output(thresholds):
    """Summaries are model text derived from untrusted customer email, so the
    report must not be able to execute anything a customer wrote."""
    payload = '<script>alert("xss")</script>'
    old = make_result("c1", passed=True, summary=payload)
    new = make_result("c1", passed=False, summary=payload)
    comparison = compare_runs(
        make_summary(run_id="curr", pass_rate=0.0),
        [new],
        make_summary(run_id="base", pass_rate=1.0),
        [old],
        thresholds,
    )
    html = render_report(comparison)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.unit
def test_report_handles_a_first_run_with_no_baseline(thresholds):
    comparison = compare_runs(make_summary(), [], None, [], thresholds)
    html = render_report(comparison, history=[])
    assert "no baseline" in html
    assert "Not enough history" in html


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_shipped_prompts_all_load():
    from regression_guard.config import DEFAULT_PROMPT_DIR
    from regression_guard.prompts import list_prompts

    files = list_prompts(DEFAULT_PROMPT_DIR)
    assert len(files) >= 3
    for path in files:
        config = load_prompt(path)
        assert config.version and config.system_prompt


@pytest.mark.integration
def test_resolve_prompt_by_version_stem_and_latest():
    from regression_guard.config import DEFAULT_PROMPT_DIR

    assert resolve_prompt(DEFAULT_PROMPT_DIR, "v1-baseline").version == "v1-baseline"
    assert resolve_prompt(DEFAULT_PROMPT_DIR, "v1_baseline").version == "v1-baseline"
    assert resolve_prompt(DEFAULT_PROMPT_DIR, "latest").version == "v3-terse"
    with pytest.raises(FileNotFoundError, match="Available"):
        resolve_prompt(DEFAULT_PROMPT_DIR, "does-not-exist")


@pytest.mark.unit
def test_prompt_requires_the_email_placeholder():
    from regression_guard.models import PromptConfig

    with pytest.raises(ValueError, match=r"\{email\}"):
        PromptConfig(
            version="bad",
            created_at="2026-01-01",
            system_prompt="x",
            user_template="no placeholder here",
        )


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.mark.integration
async def test_terse_prompt_regresses_against_the_baseline_prompt(settings, dataset):
    """The headline claim of this repo, asserted end to end: a prompt change
    that looks like a harmless cleanup is caught as a regression."""
    from regression_guard.config import DEFAULT_PROMPT_DIR

    rich = resolve_prompt(DEFAULT_PROMPT_DIR, "v2-tightened")
    terse = resolve_prompt(DEFAULT_PROMPT_DIR, "v3-terse")
    full_dataset = __import__(
        "regression_guard.runner", fromlist=["load_dataset"]
    ).load_dataset(settings.dataset_path)

    provider = MockProvider()
    good_summary, good_results = await run_eval(rich, full_dataset, provider, settings)
    bad_summary, bad_results = await run_eval(terse, full_dataset, provider, settings)

    comparison = compare_runs(
        bad_summary,
        bad_results,
        good_summary,
        good_results,
        Thresholds(),
        {case.id: case for case in full_dataset.cases},
    )
    assert comparison.verdict is Verdict.CRITICAL
    assert len(comparison.regressions) > len(comparison.improvements)
    assert comparison.pass_rate_delta < -Thresholds().critical_delta
