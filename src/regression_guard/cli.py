"""Command line entry point.

    rg run       - evaluate a prompt, diff against baseline, report, alert
    rg history   - list recent runs
    rg validate  - check the golden dataset and prompt files without spending tokens

Exit codes are the contract with CI:
    0 = pass, 1 = warn, 2 = critical regression, 3 = usage/config error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .compare import compare_runs
from .config import DEFAULT_SUMMARY_PASS_THRESHOLD, Settings
from .drift import detect_drift
from .models import Comparison, Verdict
from .prompts import list_prompts, load_prompt, resolve_prompt
from .report import write_report
from .runner import load_dataset, run_eval
from .slack import send_alert
from .store import RunStore

EXIT_PASS, EXIT_WARN, EXIT_CRITICAL, EXIT_ERROR = 0, 1, 2, 3

logger = logging.getLogger("regression_guard")


def git_sha() -> str:
    for env_var in ("GITHUB_SHA", "GIT_COMMIT"):
        if os.getenv(env_var):
            return os.environ[env_var]
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, never a shell string
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - PATH lookup is intended; CI images vary
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _markdown_summary(comparison: Comparison, report_path: Path) -> str:
    current = comparison.current
    icon = {Verdict.PASS: "✅", Verdict.WARN: "⚠️", Verdict.CRITICAL: "🚨"}[comparison.verdict]
    lines = [
        f"### {icon} Prompt eval: **{comparison.verdict.value.upper()}**",
        "",
        f"`{current.prompt_version}` on dataset `{current.dataset_version}` "
        f"via `{current.model}` ({current.provider})",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|---|---:|---:|---:|",
    ]
    baseline = comparison.baseline
    lines.append(
        f"| Pass rate | {baseline.pass_rate:.1%} | {current.pass_rate:.1%} | "
        f"{comparison.pass_rate_delta:+.1%} |"
        if baseline
        else f"| Pass rate | - | {current.pass_rate:.1%} | - |"
    )
    lines.append(
        f"| Category accuracy | {baseline.category_accuracy:.1%} | {current.category_accuracy:.1%} | "
        f"{comparison.category_accuracy_delta:+.1%} |"
        if baseline
        else f"| Category accuracy | - | {current.category_accuracy:.1%} | - |"
    )
    lines.append(
        f"| Composite | {baseline.composite_score:.3f} | {current.composite_score:.3f} | "
        f"{comparison.composite_delta:+.3f} |"
        if baseline
        else f"| Composite | - | {current.composite_score:.3f} | - |"
    )
    lines += [
        "",
        f"**{len(comparison.regressions)} regressions**, "
        f"{len(comparison.improvements)} improvements"
        + (f", McNemar p={comparison.p_value:.3f}" if comparison.p_value is not None else ""),
        "",
    ]
    for reason in comparison.reasons:
        lines.append(f"- {reason}")
    if comparison.regressions:
        lines += ["", "<details><summary>Regressed cases</summary>", ""]
        lines.append("| Case | Expected | Baseline | Current |")
        lines.append("|---|---|---|---|")
        for flip in comparison.regressions[:25]:
            old = flip.old_category.value if flip.old_category else "error"
            new = flip.new_category.value if flip.new_category else "error"
            lines.append(
                f"| `{flip.case_id}` | `{flip.expected_category.value}` | `{old}` | `{new}` |"
            )
        lines += ["", "</details>"]
    lines += ["", f"Full HTML diff report: `{report_path.name}` (CI artifact `eval-report`)."]
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.provider:
        settings = replace(settings, provider=args.provider)
    if args.model:
        settings = replace(settings, model=args.model, judge_model=args.model)

    config = resolve_prompt(settings.prompt_dir, args.prompt)
    dataset = load_dataset(settings.dataset_path)
    store = RunStore(settings.db_path)

    from .providers import build_provider

    try:
        provider = build_provider(
            settings.provider,
            **(
                {}
                if settings.provider == "mock"
                else {"timeout": settings.request_timeout, "max_retries": settings.max_retries}
            ),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(
        f"Running {len(dataset.cases)} cases | prompt {config.version} | "
        f"model {settings.model} | provider {settings.provider}",
        file=sys.stderr,
    )
    summary, results = asyncio.run(
        run_eval(
            config,
            dataset,
            provider,
            settings,
            git_sha=git_sha(),
            summary_pass_threshold=args.summary_threshold,
        )
    )

    if args.baseline:
        baseline = store.get_run(args.baseline)
        if baseline is None:
            print(f"error: no run with id {args.baseline!r}", file=sys.stderr)
            return EXIT_ERROR
    else:
        baseline = store.latest_baseline(
            prompt_version=summary.prompt_version, model=summary.model
        )
    baseline_results = store.get_case_results(baseline.run_id) if baseline else []
    comparison = compare_runs(
        summary,
        results,
        baseline,
        baseline_results,
        settings.thresholds,
        {case.id: case for case in dataset.cases},
    )

    store.save_run(summary, results)
    history = list(reversed(store.recent_runs(limit=max(15, settings.thresholds.drift_window + 3))))
    drift = detect_drift(list(reversed(history)), settings.thresholds)
    if drift.message:
        comparison.drift_warning = drift.message

    report_path = write_report(comparison, settings.report_dir, history, drift)
    report_url = (
        f"{settings.report_base_url.rstrip('/')}/{report_path.name}"
        if settings.report_base_url
        else None
    )

    if not args.no_slack:
        send_alert(settings.slack_webhook_url, comparison, report_url, drift)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(comparison.as_dict(), indent=2), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(_markdown_summary(comparison, report_path), encoding="utf-8")

    print(f"\nverdict: {comparison.verdict.value.upper()}")
    print(f"pass rate: {summary.pass_rate:.1%} ({summary.passed_cases}/{summary.total_cases})")
    print(f"composite: {summary.composite_score:.3f}")
    print(f"regressions: {len(comparison.regressions)}  improvements: {len(comparison.improvements)}")
    for reason in comparison.reasons:
        print(f"  - {reason}")
    if drift.message:
        print(f"  ! {drift.message}")
    print(f"report: {report_path}")

    if comparison.verdict is Verdict.CRITICAL:
        return EXIT_CRITICAL
    if comparison.verdict is Verdict.WARN and args.fail_on_warn:
        return EXIT_WARN
    return EXIT_PASS


def cmd_history(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = RunStore(settings.db_path)
    runs = store.recent_runs(limit=args.limit, prompt_version=args.prompt)
    if not runs:
        print("no runs recorded yet")
        return EXIT_PASS
    header = f"{'run':<14}{'when':<21}{'prompt':<18}{'pass':>8}{'composite':>11}{'errors':>8}"
    print(header)
    print("-" * len(header))
    for run in runs:
        print(
            f"{run.run_id:<14}{run.created_at[:19]:<21}{run.prompt_version:<18}"
            f"{run.pass_rate:>7.1%} {run.composite_score:>10.3f}{run.error_count:>8}"
        )
    return EXIT_PASS


def cmd_validate(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    problems: list[str] = []

    try:
        dataset = load_dataset(settings.dataset_path)
        print(f"dataset {settings.dataset_path.name}: {len(dataset.cases)} cases, "
              f"version {dataset.dataset_version}")
        by_category: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        for case in dataset.cases:
            by_category[case.expected_category.value] = by_category.get(case.expected_category.value, 0) + 1
            by_difficulty[case.expected_difficulty.value] = (
                by_difficulty.get(case.expected_difficulty.value, 0) + 1
            )
            if not case.notes.strip():
                problems.append(f"case {case.id} has no notes explaining why it is in the set")
        print(f"  by category:   {by_category}")
        print(f"  by difficulty: {by_difficulty}")
        if len(dataset.cases) < 50:
            problems.append(f"only {len(dataset.cases)} cases; 50 is the working minimum")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        problems.append(f"dataset invalid: {exc}")

    prompt_files = list_prompts(settings.prompt_dir)
    if not prompt_files:
        problems.append(f"no prompt files in {settings.prompt_dir}")
    for path in prompt_files:
        try:
            config = load_prompt(path)
            print(f"prompt {path.name}: version {config.version}, {len(config.few_shot)} few-shot examples")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"prompt {path.name} invalid: {exc}")

    if problems:
        print("\nproblems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_ERROR
    print("\nall good")
    return EXIT_PASS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rg", description="CI/CD regression detection for LLM features")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the eval and diff it against the baseline")
    run.add_argument("--prompt", default=None, help="prompt version, file path, or 'latest'")
    run.add_argument("--provider", default=None, choices=["mock", "openai"])
    run.add_argument("--model", default=None)
    run.add_argument("--summary-threshold", type=int, default=DEFAULT_SUMMARY_PASS_THRESHOLD,
                     help="minimum judge score (1-5) for a case to pass")
    run.add_argument("--baseline", default=None,
                     help="run id to diff against; default is the last run of a different prompt/model")
    run.add_argument("--fail-on-warn", action="store_true", help="exit non-zero on WARN as well as CRITICAL")
    run.add_argument("--no-slack", action="store_true")
    run.add_argument("--json-out", default=None, help="write the comparison as JSON")
    run.add_argument("--markdown-out", default=None, help="write a PR-comment-ready markdown summary")
    run.set_defaults(func=cmd_run)

    history = sub.add_parser("history", help="list recent runs")
    history.add_argument("--limit", type=int, default=20)
    history.add_argument("--prompt", default=None, help="filter by prompt version")
    history.set_defaults(func=cmd_history)

    validate = sub.add_parser("validate", help="validate dataset and prompts without calling any model")
    validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.getenv("RG_LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
