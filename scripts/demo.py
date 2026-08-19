"""End-to-end demo: three prompt versions, one caught regression.

Run it with `python scripts/demo.py`. Nothing here touches the network - the
offline provider makes the whole story reproducible on a laptop with no keys.

    1. v1-baseline   establishes the baseline
    2. v2-tightened  improves on it, merges clean
    3. v3-terse      a token-saving cleanup that quietly breaks triage

Add --drift to also seed a slow-decline history and show the drift warning
firing on runs where no single delta ever crossed a threshold.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from regression_guard.cli import main as cli_main  # noqa: E402

RUNS = ROOT / "runs"


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def demo_prompt_regression() -> int:
    steps = [
        ("v1-baseline", "First run. Nothing to compare against, so it becomes the baseline."),
        ("v2-tightened", "Adds tie-break rules for the boundary cases. Should improve."),
        ("v3-terse", "Strips the cue lists to cut prompt cost. Looks harmless in review."),
    ]
    last = 0
    for version, note in steps:
        banner(f"{version}\n{note}")
        last = cli_main(["run", "--prompt", version, "--no-slack"])
        print(f"exit code: {last}")
    return last


def _synthetic_run(index: int, composite_score: float):
    """A recorded run with a *stable* pass rate and a decaying composite.

    That combination is the realistic shape of slow drift: categories keep
    landing, so the headline number holds, while summary quality erodes
    underneath it. Holding pass rate flat also keeps the per-run verdict at
    PASS, which is what makes the drift warning the only thing that fires.
    """
    from regression_guard.models import RunSummary

    return RunSummary(
        run_id=f"seed-{index:02d}",
        created_at=f"2026-07-{index + 1:02d}T09:00:00+00:00",
        prompt_version="v2-tightened",
        dataset_version="2026-08-20.v1",
        model="gpt-4o-mini",
        provider="mock",
        total_cases=60,
        passed_cases=48,
        pass_rate=0.80,
        category_accuracy=0.90,
        mean_judge_score=1 + 4 * composite_score,
        composite_score=composite_score,
        p50_latency_ms=120.0,
        p95_latency_ms=260.0,
        total_tokens=24000,
        error_count=0,
    )


def demo_drift() -> None:
    """Seed a history that degrades one notch per run.

    Each consecutive delta is around 1pp: far inside both thresholds, so the
    per-run check stays green the whole way down. Only the rolling window
    sees it.
    """
    from regression_guard.config import Settings
    from regression_guard.store import RunStore

    settings = Settings.from_env()
    store = RunStore(settings.db_path)
    banner("Seeding a slow decline: 0.960 -> 0.905 composite, about 1pp per run")
    for index, score in enumerate([0.960, 0.950, 0.945, 0.935, 0.930, 0.920, 0.915, 0.905]):
        store.save_run(_synthetic_run(index, score), [])
        print(f"  run {index}: composite {score:.3f}")
    print()
    print("Every consecutive delta above is inside both thresholds.")
    cli_main(["run", "--prompt", "v2-tightened", "--no-slack"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep existing run history")
    parser.add_argument("--drift", action="store_true", help="also demo slow-drift detection")
    args = parser.parse_args()

    if not args.keep and RUNS.exists():
        shutil.rmtree(RUNS, ignore_errors=True)
    os.environ.setdefault("RG_PROVIDER", "mock")

    if args.drift:
        demo_drift()
        return 0

    exit_code = demo_prompt_regression()

    banner("Run history")
    cli_main(["history"])

    report = RUNS / "reports" / "index.html"
    if report.exists():
        print(f"\nOpen the diff report: {report}")
    print(
        "\nThe last run exited "
        f"{exit_code}, which is what makes the GitHub Action fail the merge."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
