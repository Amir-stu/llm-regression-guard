"""Self-contained HTML diff report.

No CDN, no JS framework, no build step - the report is a single file that can
be opened from a CI artifact, attached to a PR, or emailed. The trend chart is
inline SVG generated here rather than a charting library, for the same reason.

All model output is escaped before it reaches the template: the summaries in
this report come from an LLM reading untrusted customer email text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from pathlib import Path

from .drift import DriftReport
from .error_analysis import ErrorAnalysis
from .models import Comparison, RunSummary, Verdict

_VERDICT_LABEL = {
    Verdict.PASS: "PASS",
    Verdict.WARN: "WARN",
    Verdict.CRITICAL: "CRITICAL",
}

_CSS = """
:root {
  --bg: #ffffff; --fg: #16191d; --muted: #5b636d; --line: #d9dde2;
  --panel: #f6f7f9; --pass: #17692f; --warn: #8a5a00; --crit: #a3241c;
  --pos: #17692f; --neg: #a3241c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16191d; --fg: #e8eaed; --muted: #9aa3ad; --line: #2f353c;
    --panel: #1e2227; --pass: #58c477; --warn: #d9a441; --crit: #ef7a72;
    --pos: #58c477; --neg: #ef7a72;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.5rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2.25rem 0 .75rem; padding-bottom: .35rem;
     border-bottom: 1px solid var(--line); }
.meta { color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }
.meta code { background: var(--panel); padding: .1rem .35rem; border-radius: 3px; }
.verdict { display: inline-block; padding: .3rem .7rem; border: 1px solid currentColor;
           border-radius: 3px; font-weight: 700; letter-spacing: .06em; font-size: .8rem; }
.verdict.pass { color: var(--pass); }
.verdict.warn { color: var(--warn); }
.verdict.critical { color: var(--crit); }
.reasons { margin: 1rem 0 0; padding-left: 1.1rem; color: var(--muted); }
.reasons li { margin-bottom: .3rem; }
.scorecard { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 1px; background: var(--line); border: 1px solid var(--line); margin-top: 1rem; }
.cell { background: var(--bg); padding: .8rem .9rem; }
.cell .label { color: var(--muted); font-size: .75rem; text-transform: uppercase;
               letter-spacing: .06em; }
.cell .value { font-size: 1.35rem; font-variant-numeric: tabular-nums; margin-top: .2rem; }
.cell .delta { font-size: .82rem; font-variant-numeric: tabular-nums; }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .87rem; }
th, td { text-align: left; padding: .55rem .6rem; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .75rem; text-transform: uppercase;
     letter-spacing: .05em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .82em; }
.old { color: var(--neg); }
.new { color: var(--pos); }
.empty { color: var(--muted); padding: .8rem 0; }
.note { background: var(--panel); border-left: 3px solid var(--warn); padding: .7rem .9rem;
        margin-top: 1rem; font-size: .88rem; }
figure { margin: 1rem 0 0; }
svg { max-width: 100%; height: auto; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
         border-top: 1px solid var(--line); padding-top: .8rem; }
"""


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _delta_span(delta: float, as_pct: bool = True) -> str:
    if abs(delta) < 1e-9:
        return '<span class="delta">no change</span>'
    css = "pos" if delta > 0 else "neg"
    text = f"{delta:+.1%}" if as_pct else f"{delta:+.3f}"
    return f'<span class="delta {css}">{text} vs baseline</span>'


def _cell(label: str, value: str, delta_html: str = "") -> str:
    return (
        f'<div class="cell"><div class="label">{escape(label)}</div>'
        f'<div class="value">{value}</div>{delta_html}</div>'
    )


def _trend_svg(runs: list[RunSummary], width: int = 940, height: int = 190) -> str:
    """Inline sparkline of composite score across recent runs (oldest first)."""
    if len(runs) < 2:
        return '<p class="empty">Not enough history for a trend chart yet.</p>'

    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 30
    scores = [r.composite_score for r in runs]
    lo, hi = min(scores), max(scores)
    span = max(0.05, hi - lo)
    lo, hi = max(0.0, lo - span * 0.15), min(1.0, hi + span * 0.15)
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x(i: int) -> float:
        return pad_l + (plot_w * i / (len(runs) - 1))

    def y(score: float) -> float:
        return pad_t + plot_h * (1 - (score - lo) / (hi - lo))

    points = " ".join(f"{x(i):.1f},{y(s):.1f}" for i, s in enumerate(scores))
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(s):.1f}" r="3" fill="currentColor">'
        f"<title>{escape(runs[i].run_id)} | {escape(runs[i].prompt_version)} | {s:.3f}</title></circle>"
        for i, s in enumerate(scores)
    )
    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{y(v):.1f}" x2="{width - pad_r}" y2="{y(v):.1f}" '
        f'stroke="var(--line)" stroke-width="1"/>'
        f'<text x="{pad_l - 8}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="10" '
        f'fill="var(--muted)">{v:.2f}</text>'
        for v in (lo, (lo + hi) / 2, hi)
    )
    labels = "".join(
        f'<text x="{x(i):.1f}" y="{height - 10}" text-anchor="middle" font-size="9" '
        f'fill="var(--muted)">{escape(runs[i].prompt_version[:9])}</text>'
        for i in range(0, len(runs), max(1, len(runs) // 8))
    )
    return (
        f'<figure><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Composite score across the last {len(runs)} runs">'
        f"{gridlines}"
        f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2"/>'
        f"{dots}{labels}</svg>"
        f"<figcaption class=\"meta\">Composite score, oldest run on the left.</figcaption></figure>"
    )


def _regression_rows(comparison: Comparison) -> str:
    if not comparison.regressions:
        return '<p class="empty">No cases regressed against the baseline.</p>'
    rows = []
    for flip in comparison.regressions:
        rows.append(
            "<tr>"
            f'<td class="mono">{escape(flip.case_id)}</td>'
            f"<td>{escape(flip.difficulty.value)}</td>"
            f"<td><code>{escape(flip.expected_category.value)}</code></td>"
            f'<td class="old"><code>{escape(flip.old_category.value if flip.old_category else "error")}</code>'
            f'<br><span class="meta">judge {flip.old_judge_score if flip.old_judge_score is not None else "-"}/5</span>'
            f"<br>{escape(flip.old_summary[:220])}</td>"
            f'<td class="new"><code>{escape(flip.new_category.value if flip.new_category else "error")}</code>'
            f'<br><span class="meta">judge {flip.new_judge_score if flip.new_judge_score is not None else "-"}/5</span>'
            f"<br>{escape(flip.new_summary[:220])}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Case</th><th>Difficulty</th><th>Expected</th>"
        "<th>Baseline output</th><th>Current output</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _per_category_table(comparison: Comparison) -> str:
    current = comparison.current.per_category_accuracy
    baseline = comparison.baseline.per_category_accuracy if comparison.baseline else {}
    keys = sorted(set(current) | set(baseline))
    if not keys:
        return ""
    rows = []
    for key in keys:
        now = current.get(key, 0.0)
        before = baseline.get(key)
        delta = "-" if before is None else _delta_span(now - before)
        rows.append(
            f"<tr><td><code>{escape(key)}</code></td>"
            f'<td class="num">{"-" if before is None else _pct(before)}</td>'
            f'<td class="num">{_pct(now)}</td><td class="num">{delta}</td></tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Category</th><th class="num">Baseline</th>'
        '<th class="num">Current</th><th class="num">Delta</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _failure_mode_table(analysis: ErrorAnalysis | None) -> str:
    if analysis is None:
        return '<p class="empty">Error analysis was not run for this report.</p>'
    if not analysis.buckets:
        return '<p class="empty">No failed cases.</p>'
    rows = []
    for bucket in analysis.buckets:
        delta = "-" if bucket.delta is None else f"{bucket.delta:+d}"
        examples = "<br>".join(
            f'<span class="mono">{escape(e.case_id)}</span> {escape(e.detail[:160])}' for e in bucket.examples
        )
        rows.append(
            f"<tr><td><code>{escape(bucket.mode.value)}</code></td>"
            f'<td class="num">{bucket.count}</td><td class="num">{_pct(bucket.share_of_failures)}</td>'
            f'<td class="num">{delta}</td><td>{examples}</td></tr>'
        )
    misroutes = ", ".join(f"{escape(pair)} ({count})" for pair, count in analysis.misroutes.items())
    soft = ", ".join(f"{escape(b.mode.value)} ({b.count})" for b in analysis.soft_findings)
    notes = "".join(
        f'<p class="meta">{label}: {text}</p>'
        for label, text in (("Misroutes", misroutes), ("Soft findings, not counted as failures", soft))
        if text
    )
    return (
        f'<p class="meta">{analysis.failed_cases} of {analysis.total_cases} cases failed. '
        "A case can carry several modes, so shares can sum past 100%.</p>"
        '<div class="scroll"><table><thead><tr><th>Failure mode</th><th class="num">Cases</th>'
        '<th class="num">Share of failures</th><th class="num">vs baseline</th><th>Examples</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + notes
    )


def render_report(
    comparison: Comparison,
    history: list[RunSummary] | None = None,
    drift: DriftReport | None = None,
    analysis: ErrorAnalysis | None = None,
) -> str:
    current = comparison.current
    baseline = comparison.baseline
    verdict = comparison.verdict
    history = history or []

    scorecard = "".join(
        [
            _cell(
                "Pass rate",
                _pct(current.pass_rate),
                _delta_span(comparison.pass_rate_delta) if baseline else "",
            ),
            _cell(
                "Category accuracy",
                _pct(current.category_accuracy),
                _delta_span(comparison.category_accuracy_delta) if baseline else "",
            ),
            _cell(
                "Composite score",
                f"{current.composite_score:.3f}",
                _delta_span(comparison.composite_delta, as_pct=False) if baseline else "",
            ),
            _cell("Mean judge score", f"{current.mean_judge_score:.2f} / 5"),
            _cell("Regressions", str(len(comparison.regressions))),
            _cell("Improvements", str(len(comparison.improvements))),
            _cell("p50 / p95 latency", f"{current.p50_latency_ms:.0f} / {current.p95_latency_ms:.0f} ms"),
            _cell("Tokens", f"{current.total_tokens:,}"),
            _cell("Hard errors", str(current.error_count)),
        ]
    )

    significance = (
        f"McNemar exact p = {comparison.p_value:.4f} over "
        f"{len(comparison.regressions) + len(comparison.improvements)} flipped cases "
        f"({'significant' if comparison.significant else 'not significant'})."
        if comparison.p_value is not None
        else "No baseline to test against."
    )

    reasons = "".join(f"<li>{escape(r)}</li>" for r in comparison.reasons)
    drift_html = f'<div class="note">{escape(drift.message)}</div>' if drift and drift.message else ""
    baseline_line = (
        f"baseline <code>{escape(baseline.run_id)}</code> "
        f"(prompt <code>{escape(baseline.prompt_version)}</code>, {escape(baseline.created_at[:19])})"
        if baseline
        else "no baseline - this run establishes one"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval diff - {escape(current.prompt_version)} - {escape(current.run_id)}</title>
<style>{_CSS}</style></head>
<body><main>
<h1>Eval diff report</h1>
<p class="meta">
run <code>{escape(current.run_id)}</code> &middot;
prompt <code>{escape(current.prompt_version)}</code> &middot;
dataset <code>{escape(current.dataset_version)}</code> &middot;
model <code>{escape(current.model)}</code> &middot;
provider <code>{escape(current.provider)}</code> &middot;
commit <code>{escape(current.git_sha[:12] or 'local')}</code><br>
{escape(current.created_at)} &middot; {baseline_line}
</p>

<p><span class="verdict {verdict.value}">{_VERDICT_LABEL[verdict]}</span></p>
<ul class="reasons">{reasons}</ul>
<p class="meta">{escape(significance)}</p>
{drift_html}

<h2>Scorecard</h2>
<div class="scorecard">{scorecard}</div>

<h2>Per-category accuracy</h2>
{_per_category_table(comparison)}

<h2>Failure modes</h2>
{_failure_mode_table(analysis)}

<h2>Regressed cases ({len(comparison.regressions)})</h2>
{_regression_rows(comparison)}

<h2>Trend across recent runs</h2>
{_trend_svg(history)}

<footer>Generated by regression-guard at {escape(datetime.now(UTC).isoformat(timespec='seconds'))}.
Thresholds and dataset version are part of the run record; changing either changes what
these numbers mean.</footer>
</main></body></html>"""


def write_report(
    comparison: Comparison,
    output_dir: str | Path,
    history: list[RunSummary] | None = None,
    drift: DriftReport | None = None,
    analysis: ErrorAnalysis | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"report-{comparison.current.run_id}.html"
    path.write_text(render_report(comparison, history, drift, analysis), encoding="utf-8")
    latest = output_dir / "index.html"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path
