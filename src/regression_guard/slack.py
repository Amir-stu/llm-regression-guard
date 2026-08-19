"""Slack incoming-webhook alerting.

Kept dumb on purpose: it formats a payload and posts it. Building the payload
is pure and unit-tested; the network call is the only impure part and is
skipped when no webhook is configured, so a missing secret degrades to a log
line instead of failing the build.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .drift import DriftReport
from .models import Comparison, Verdict

logger = logging.getLogger(__name__)

_EMOJI = {Verdict.PASS: ":white_check_mark:", Verdict.WARN: ":warning:", Verdict.CRITICAL: ":rotating_light:"}
_HEADLINE = {
    Verdict.PASS: "Eval passed",
    Verdict.WARN: "Eval warning",
    Verdict.CRITICAL: "Eval regression detected",
}


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def build_payload(
    comparison: Comparison,
    report_url: str | None = None,
    drift: DriftReport | None = None,
) -> dict[str, Any]:
    current = comparison.current
    baseline = comparison.baseline
    verdict = comparison.verdict

    if baseline is None:
        headline = (
            f"Baseline established: {_fmt_pct(current.pass_rate)} pass rate "
            f"across {current.total_cases} cases."
        )
    else:
        headline = (
            f"{len(comparison.regressions)} regressions, {len(comparison.improvements)} improvements. "
            f"Pass rate {_fmt_pct(baseline.pass_rate)} to {_fmt_pct(current.pass_rate)} "
            f"({comparison.pass_rate_delta:+.1%})."
        )

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{_EMOJI[verdict]} {_HEADLINE[verdict]}"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Prompt*\n`{current.prompt_version}`"},
                {"type": "mrkdwn", "text": f"*Model*\n`{current.model}`"},
                {"type": "mrkdwn", "text": f"*Dataset*\n`{current.dataset_version}`"},
                {"type": "mrkdwn", "text": f"*Composite*\n{current.composite_score:.3f}"},
                {"type": "mrkdwn", "text": f"*Category acc.*\n{_fmt_pct(current.category_accuracy)}"},
                {"type": "mrkdwn", "text": f"*p95 latency*\n{current.p95_latency_ms:.0f} ms"},
            ],
        },
    ]

    if comparison.reasons:
        reasons = "\n".join(f"- {r}" for r in comparison.reasons[:4])
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": reasons}})

    if comparison.regressions:
        listed = ", ".join(f"`{f.case_id}`" for f in comparison.regressions[:8])
        extra = "" if len(comparison.regressions) <= 8 else f" (+{len(comparison.regressions) - 8} more)"
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Regressed cases:* {listed}{extra}"}}
        )

    if drift is not None and drift.message:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f":chart_with_downwards_trend: {drift.message}"}})

    if report_url:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open diff report"},
                        "url": report_url,
                    }
                ],
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"run `{current.run_id}` | commit `{current.git_sha[:8] or 'local'}` | {current.created_at}",
                }
            ],
        }
    )

    return {"text": f"{_HEADLINE[verdict]}: {headline}", "blocks": blocks}


def send_alert(
    webhook_url: str | None,
    comparison: Comparison,
    report_url: str | None = None,
    drift: DriftReport | None = None,
    timeout: float = 10.0,
) -> bool:
    """Post to Slack. Returns True if delivered.

    Never raises: alerting failures must not turn a passing eval into a failed
    build, and must not mask a real regression behind a network error.
    """
    if not webhook_url:
        logger.info("SLACK_WEBHOOK_URL not set; skipping Slack alert.")
        return False

    # The webhook arrives from the environment, so pin the scheme here rather
    # than trusting it: urlopen would otherwise happily follow file:// or a
    # custom scheme if the secret were ever set to something unexpected.
    if not webhook_url.startswith("https://"):
        logger.warning("Refusing to post the alert: webhook URL is not https.")
        return False

    payload = build_payload(comparison, report_url, drift)
    request = urllib.request.Request(  # noqa: S310 - scheme pinned to https above
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Slack alert failed: %s", exc)
        return False
