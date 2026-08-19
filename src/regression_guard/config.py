"""Runtime configuration. Every value is env-overridable so the container
can be reconfigured without a rebuild."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_DB_PATH = PROJECT_ROOT / "runs" / "eval_runs.db"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "runs" / "reports"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "golden_dataset_v1.json"
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "prompts"

# Scoring weights for the composite quality score. Category accuracy dominates
# because it is objectively checkable; the judge score is a softer signal.
CATEGORY_WEIGHT = 0.7
SUMMARY_WEIGHT = 0.3

# The judge returns 1-5; normalise to 0-1 before weighting.
JUDGE_SCALE_MIN = 1
JUDGE_SCALE_MAX = 5

# A test case passes when the category is exact AND the judge score clears this.
DEFAULT_SUMMARY_PASS_THRESHOLD = 3


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Thresholds:
    """Delta thresholds, expressed as absolute percentage points of pass rate.

    A negative delta of 3pp on an 80-case set is roughly 2-3 cases. That is
    inside sampling noise for a single run, which is why `warn_delta` only
    warns and the significance test is consulted before escalating.
    """

    warn_delta: float = 0.03
    critical_delta: float = 0.08
    # Two-sided alpha for the proportion test used to qualify a delta.
    significance_alpha: float = 0.05
    # Rolling window for slow-drift detection.
    drift_window: int = 7
    # Absolute floor for the rolling mean of the composite score.
    drift_floor: float = 0.85

    @classmethod
    def from_env(cls) -> Thresholds:
        return cls(
            warn_delta=_env_float("RG_WARN_DELTA", 0.03),
            critical_delta=_env_float("RG_CRITICAL_DELTA", 0.08),
            significance_alpha=_env_float("RG_SIGNIFICANCE_ALPHA", 0.05),
            drift_window=_env_int("RG_DRIFT_WINDOW", 7),
            drift_floor=_env_float("RG_DRIFT_FLOOR", 0.85),
        )


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    concurrency: int = 8
    request_timeout: float = 30.0
    max_retries: int = 3
    db_path: Path = DEFAULT_DB_PATH
    report_dir: Path = DEFAULT_REPORT_DIR
    dataset_path: Path = DEFAULT_DATASET
    prompt_dir: Path = DEFAULT_PROMPT_DIR
    slack_webhook_url: str | None = None
    report_base_url: str | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            provider=os.getenv("RG_PROVIDER", "mock"),
            model=os.getenv("RG_MODEL", "gpt-4o-mini"),
            judge_model=os.getenv("RG_JUDGE_MODEL", os.getenv("RG_MODEL", "gpt-4o-mini")),
            concurrency=_env_int("RG_CONCURRENCY", 8),
            request_timeout=_env_float("RG_REQUEST_TIMEOUT", 30.0),
            max_retries=_env_int("RG_MAX_RETRIES", 3),
            db_path=Path(os.getenv("RG_DB_PATH", str(DEFAULT_DB_PATH))),
            report_dir=Path(os.getenv("RG_REPORT_DIR", str(DEFAULT_REPORT_DIR))),
            dataset_path=Path(os.getenv("RG_DATASET", str(DEFAULT_DATASET))),
            prompt_dir=Path(os.getenv("RG_PROMPT_DIR", str(DEFAULT_PROMPT_DIR))),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL") or None,
            report_base_url=os.getenv("RG_REPORT_BASE_URL") or None,
            thresholds=Thresholds.from_env(),
        )
