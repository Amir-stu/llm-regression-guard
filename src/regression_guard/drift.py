"""Slow-drift detection across a rolling window of runs.

Per-run diffing is blind to a specific, common failure: a prompt or model that
loses one case per week. Each individual run is inside every threshold, so
nothing ever fires, and three months later the feature is materially worse
than anyone signed off on.

This module watches the rolling mean of the composite score instead of any
single run, and warns on two conditions:

  * ``floor``  - the rolling mean has dropped below an absolute quality bar.
  * ``slope``  - the rolling mean is falling monotonically enough that the
                 direction is not plausibly noise.

It is deliberately separate from the per-run verdict. Drift never blocks a
merge, because the offending commit is usually not the one being reviewed.
It raises a warning aimed at the team, not at the PR author.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Thresholds
from .models import RunSummary


@dataclass(frozen=True)
class DriftReport:
    window: int
    rolling_mean: float
    previous_mean: float | None
    below_floor: bool
    falling: bool
    message: str | None

    @property
    def triggered(self) -> bool:
        return self.below_floor or self.falling


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def detect_drift(runs: list[RunSummary], thresholds: Thresholds) -> DriftReport:
    """``runs`` is newest-first, as returned by ``RunStore.recent_runs``."""
    window = max(2, thresholds.drift_window)
    ordered = list(reversed(runs))  # oldest -> newest
    scores = [r.composite_score for r in ordered]

    if len(scores) < window:
        return DriftReport(window, _mean(scores), None, False, False, None)

    current_window = scores[-window:]
    rolling_mean = _mean(current_window)

    previous_mean: float | None = None
    if len(scores) >= window + 1:
        previous_mean = _mean(scores[-window - 1 : -1])

    below_floor = rolling_mean < thresholds.drift_floor

    # 'Falling' means the window's second half is worse than its first half and
    # the newest run is below the window mean. One noisy run cannot satisfy both.
    half = window // 2
    first_half = _mean(current_window[:half])
    second_half = _mean(current_window[-half:])
    falling = (
        half >= 1
        and second_half < first_half
        and current_window[-1] < rolling_mean
        and (previous_mean is None or rolling_mean < previous_mean)
    )

    message: str | None = None
    if below_floor:
        message = (
            f"Slow drift: {window}-run rolling composite score is {rolling_mean:.3f}, "
            f"below the {thresholds.drift_floor:.3f} floor. No single run tripped a "
            "threshold, so this degraded gradually."
        )
    elif falling:
        message = (
            f"Slow drift: {window}-run rolling composite score is trending down "
            f"({first_half:.3f} -> {second_half:.3f} within the window). Still above the "
            f"{thresholds.drift_floor:.3f} floor, but the direction is consistent."
        )

    return DriftReport(window, rolling_mean, previous_mean, below_floor, falling, message)
