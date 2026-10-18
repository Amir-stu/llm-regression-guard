"""Streamlit dashboard over the run history.

    pip install -e ".[dashboard]"
    streamlit run dashboard.py

It opens the same SQLite file `rg run` writes (RG_DB_PATH) and never calls a
model. When that file holds no runs, as on a fresh Streamlit Community Cloud
deployment, it is seeded once with the offline mock demo and says so on the
page. All data shaping lives in `regression_guard.dashboard_data`, which is
unit-tested; this file only lays out tables and charts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Hosted deployments install requirements.txt but not the package itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from regression_guard.config import DEFAULT_SUMMARY_PASS_THRESHOLD, JUDGE_SCALE_MAX, JUDGE_SCALE_MIN, Settings
from regression_guard.dashboard_data import (
    STATUS_REGRESSED,
    case_log_rows,
    failure_mode_trend,
    pass_rate_trend,
    side_by_side,
)
from regression_guard.demo_seed import seed_demo_history
from regression_guard.error_analysis import analyse
from regression_guard.models import RunSummary
from regression_guard.store import RunStore

HISTORY_LIMIT = 50
NO_BASELINE = "(none)"


def _label(run: RunSummary) -> str:
    return f"{run.run_id} | {run.prompt_version} | {run.created_at[:16]} | {run.pass_rate:.1%}"


def _sidebar(runs: list[RunSummary]) -> tuple[RunSummary, RunSummary | None, int]:
    st.sidebar.header("Run")
    labels = {_label(run): run for run in runs}
    current = labels[st.sidebar.selectbox("Current run", list(labels))]
    others = [label for label, run in labels.items() if run.run_id != current.run_id]
    choice = st.sidebar.selectbox("Baseline", [NO_BASELINE, *others], index=1 if others else 0)
    threshold = st.sidebar.slider(
        "Judge pass threshold",
        min_value=JUDGE_SCALE_MIN,
        max_value=JUDGE_SCALE_MAX,
        value=DEFAULT_SUMMARY_PASS_THRESHOLD,
        help="Judge score below which a summary counts as off-target in error analysis.",
    )
    return current, labels.get(choice), threshold


def _headline(current: RunSummary, baseline: RunSummary | None) -> None:
    columns = st.columns(4)
    metrics = [
        ("Pass rate", current.pass_rate, baseline.pass_rate if baseline else None, "{:.1%}"),
        (
            "Category accuracy",
            current.category_accuracy,
            baseline.category_accuracy if baseline else None,
            "{:.1%}",
        ),
        ("Composite", current.composite_score, baseline.composite_score if baseline else None, "{:.3f}"),
        ("Mean judge", current.mean_judge_score, baseline.mean_judge_score if baseline else None, "{:.2f}"),
    ]
    for column, (label, value, before, fmt) in zip(columns, metrics, strict=True):
        delta = None if before is None else fmt.format(value - before)
        column.metric(label, fmt.format(value), delta)


def _trend_section(runs: list[RunSummary], store: RunStore, threshold: int) -> None:
    st.subheader("Pass rate across runs")
    trend = pd.DataFrame(pass_rate_trend(runs))
    trend["run"] = trend["prompt_version"] + " · " + trend["run_id"]
    st.line_chart(trend, x="run", y=["pass_rate", "category_accuracy", "composite_score"])

    st.subheader("Failure modes across runs")
    history = [(run, store.get_case_results(run.run_id)) for run in runs]
    modes = pd.DataFrame(failure_mode_trend(history, summary_pass_threshold=threshold))
    if modes.empty:
        st.info("No failed cases in any recorded run.")
        return
    modes["run"] = modes["prompt_version"] + " · " + modes["run_id"]
    st.bar_chart(modes, x="run", y="count", color="mode", stack=True)


def _analysis_section(
    current: RunSummary, baseline: RunSummary | None, store: RunStore, threshold: int
) -> None:
    results = store.get_case_results(current.run_id)
    baseline_results = store.get_case_results(baseline.run_id) if baseline else None
    analysis = analyse(results, summary_pass_threshold=threshold, baseline_results=baseline_results)

    st.subheader(f"Error analysis: {analysis.failed_cases}/{analysis.total_cases} cases failed")
    st.caption("A case can carry several failure modes, so shares can sum past 100%.")
    buckets = pd.DataFrame(
        [
            {
                "mode": b.mode.value,
                "cases": b.count,
                "share_of_failures": b.share_of_failures,
                "vs_baseline": b.delta,
                "example": f"{b.examples[0].case_id}: {b.examples[0].detail}" if b.examples else "",
            }
            for b in analysis.buckets
        ]
    )
    if buckets.empty:
        st.success("No failed cases in this run.")
    else:
        st.bar_chart(buckets, x="mode", y="cases", horizontal=True)
        st.dataframe(
            buckets,
            hide_index=True,
            width="stretch",
            column_config={"share_of_failures": st.column_config.NumberColumn(format="percent")},
        )
    if analysis.misroutes:
        st.markdown("**Misroutes** (expected → predicted)")
        st.dataframe(
            pd.DataFrame([{"pair": k, "cases": v} for k, v in analysis.misroutes.items()]),
            hide_index=True,
        )
    if analysis.soft_findings:
        st.markdown("**Soft findings** (counted, never failing)")
        st.dataframe(
            pd.DataFrame([{"finding": b.mode.value, "cases": b.count} for b in analysis.soft_findings]),
            hide_index=True,
        )

    if baseline_results is not None:
        st.subheader(f"Side by side: {baseline.prompt_version} → {current.prompt_version}")
        pairs = pd.DataFrame(side_by_side(results, baseline_results, summary_pass_threshold=threshold))
        statuses = st.multiselect("Show", sorted(pairs["status"].unique()), default=[STATUS_REGRESSED])
        st.dataframe(pairs[pairs["status"].isin(statuses)], hide_index=True, width="stretch")

    st.subheader("Case log")
    log = pd.DataFrame(case_log_rows(results, analysis))
    mode_filter = st.multiselect("Primary failure mode", sorted(m for m in log["primary_mode"].unique() if m))
    st.dataframe(
        log[log["primary_mode"].isin(mode_filter)] if mode_filter else log, hide_index=True, width="stretch"
    )


@st.cache_resource
def _seed_once(db_path: str) -> int:
    """Seed at most once per server process, however many visitors arrive."""
    return seed_demo_history(RunStore(db_path), Settings.from_env())


def main() -> None:
    st.set_page_config(page_title="regression-guard", layout="wide")
    st.title("regression-guard")
    settings = Settings.from_env()
    store = RunStore(settings.db_path)
    _seed_once(str(settings.db_path))
    runs = store.recent_runs(limit=HISTORY_LIMIT)
    current, baseline, threshold = _sidebar(runs)
    simulated = (
        " Runs from the offline mock provider: simulated outputs, no model was called."
        if all(run.provider == "mock" for run in runs)
        else ""
    )
    st.caption(f"Reading {settings.db_path.name}, {len(runs)} runs.{simulated}")
    _headline(current, baseline)
    _analysis_section(current, baseline, store, threshold)
    _trend_section(runs, store, threshold)


main()
