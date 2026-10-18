"""Error analysis: turn a pile of failed cases into a ranked list of causes.

A pass rate says *how much* is broken. It does not say *what* to fix next.
The working method (read the failures, label each one, count the labels,
fix the biggest bucket) only scales if the labelling is mechanical, so this
module assigns every failed case to named failure modes using signals the
pipeline already records: the parse error, the category match, the judge
score, and the deterministic assertions.

Counting rules, stated because they change how the numbers read:

* Buckets are multi-label. A misrouted case with a bad summary sits in both
  buckets, so shares can sum to more than 100%.
* Each failed case also gets exactly one *primary* mode, the most severe of
  its labels in ``_PRECEDENCE`` order. Use that when you need a partition.
* Soft findings are counted on every case, passing or not, and reported
  separately. They never contribute to the failed-case count.
* A judge call that failed is infrastructure noise, not a quality signal. It
  is listed under ``judge_unavailable`` and never labelled off-target.
"""

from __future__ import annotations

import enum
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict, Field

from .assertions import (
    CHECK_FORBIDDEN_CONTENT,
    CHECK_MAX_CHARS,
    CHECK_NUMBERS_GROUNDED,
    CHECK_REFERENCE_DETAILS,
    CHECK_REQUIRED_DETAIL,
    CHECK_SINGLE_SENTENCE,
    CHECK_STRICT_JSON,
)
from .judge import JUDGE_CALL_FAILED, JUDGE_UNPARSEABLE
from .models import CaseResult, Severity

EXAMPLES_PER_BUCKET = 3
_MD_DETAIL_MAX_CHARS = 90


class FailureMode(enum.StrEnum):
    OUTPUT_CONTRACT = "output_contract"
    MISROUTED = "misrouted"
    HALLUCINATED_NUMBER = "hallucinated_number"
    FORBIDDEN_CONTENT = "forbidden_content"
    MISSING_REQUIRED_DETAIL = "missing_required_detail"
    FORMAT_VIOLATION = "format_violation"
    SUMMARY_OFF_TARGET = "summary_off_target"
    UNCLASSIFIED = "unclassified"
    # Soft findings: counted, never failing.
    NON_STRICT_JSON = "non_strict_json"
    DROPPED_KEY_DETAIL = "dropped_key_detail"


# Most severe first. A wrong category sends the ticket to the wrong team,
# which outranks anything wrong inside the summary; an invented number
# outranks a style problem because an agent will act on it.
_PRECEDENCE: tuple[FailureMode, ...] = (
    FailureMode.OUTPUT_CONTRACT,
    FailureMode.MISROUTED,
    FailureMode.HALLUCINATED_NUMBER,
    FailureMode.FORBIDDEN_CONTENT,
    FailureMode.MISSING_REQUIRED_DETAIL,
    FailureMode.FORMAT_VIOLATION,
    FailureMode.SUMMARY_OFF_TARGET,
    FailureMode.UNCLASSIFIED,
    FailureMode.NON_STRICT_JSON,
    FailureMode.DROPPED_KEY_DETAIL,
)
_RANK = {mode: index for index, mode in enumerate(_PRECEDENCE)}

_CHECK_TO_MODE: dict[str, FailureMode] = {
    CHECK_NUMBERS_GROUNDED: FailureMode.HALLUCINATED_NUMBER,
    CHECK_FORBIDDEN_CONTENT: FailureMode.FORBIDDEN_CONTENT,
    CHECK_REQUIRED_DETAIL: FailureMode.MISSING_REQUIRED_DETAIL,
    CHECK_SINGLE_SENTENCE: FailureMode.FORMAT_VIOLATION,
    CHECK_MAX_CHARS: FailureMode.FORMAT_VIOLATION,
    CHECK_STRICT_JSON: FailureMode.NON_STRICT_JSON,
    CHECK_REFERENCE_DETAILS: FailureMode.DROPPED_KEY_DETAIL,
}


class CaseExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    detail: str


class ModeBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: FailureMode
    count: int
    share_of_failures: float
    case_ids: list[str]
    examples: list[CaseExample] = Field(default_factory=list)
    baseline_count: int | None = None
    delta: int | None = None


class ErrorAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int
    failed_cases: int
    buckets: list[ModeBucket]
    soft_findings: list[ModeBucket]
    misroutes: dict[str, int]
    primary_mode: dict[str, FailureMode]
    judge_unavailable: list[str]


def _judge_unavailable(result: CaseResult) -> bool:
    return result.judge_reason.startswith((JUDGE_CALL_FAILED, JUDGE_UNPARSEABLE))


def _misroute_label(result: CaseResult) -> str:
    predicted = result.predicted_category.value if result.predicted_category else "none"
    return f"{result.expected_category.value} -> {predicted}"


def _hard_labels(result: CaseResult, threshold: int) -> list[tuple[FailureMode, str]]:
    """Every failure mode a failed case exhibits, with a triage detail each."""
    if result.error:
        return [(FailureMode.OUTPUT_CONTRACT, result.error)]
    labels: list[tuple[FailureMode, str]] = []
    if not result.category_match:
        labels.append((FailureMode.MISROUTED, _misroute_label(result)))
    for failure in result.assertion_failures:
        if failure.severity is Severity.HARD:
            labels.append((_CHECK_TO_MODE.get(failure.check, FailureMode.UNCLASSIFIED), failure.detail))
    if result.judge_score is not None and result.judge_score < threshold and not _judge_unavailable(result):
        labels.append(
            (FailureMode.SUMMARY_OFF_TARGET, f"judge {result.judge_score}/5: {result.judge_reason}")
        )
    return labels or [(FailureMode.UNCLASSIFIED, "failed without a recorded cause")]


def _soft_labels(result: CaseResult) -> list[tuple[FailureMode, str]]:
    return [
        (_CHECK_TO_MODE.get(failure.check, FailureMode.UNCLASSIFIED), failure.detail)
        for failure in result.assertion_failures
        if failure.severity is Severity.SOFT
    ]


def _group(labelled: list[tuple[str, list[tuple[FailureMode, str]]]]) -> dict[FailureMode, list[CaseExample]]:
    grouped: dict[FailureMode, list[CaseExample]] = defaultdict(list)
    for case_id, labels in labelled:
        for mode in dict.fromkeys(mode for mode, _ in labels):
            detail = next(d for m, d in labels if m is mode)
            grouped[mode].append(CaseExample(case_id=case_id, detail=detail))
    return grouped


def _hard_counts(results: list[CaseResult], threshold: int) -> Counter[FailureMode]:
    grouped = _group([(r.case_id, _hard_labels(r, threshold)) for r in results if not r.passed])
    return Counter({mode: len(examples) for mode, examples in grouped.items()})


def _buckets(
    grouped: dict[FailureMode, list[CaseExample]],
    denominator: int,
    baseline: Counter[FailureMode] | None,
) -> list[ModeBucket]:
    modes = set(grouped) | (set(baseline) if baseline is not None else set())
    buckets = [
        ModeBucket(
            mode=mode,
            count=len(grouped.get(mode, [])),
            share_of_failures=len(grouped.get(mode, [])) / denominator if denominator else 0.0,
            case_ids=[example.case_id for example in grouped.get(mode, [])],
            examples=grouped.get(mode, [])[:EXAMPLES_PER_BUCKET],
            baseline_count=baseline.get(mode, 0) if baseline is not None else None,
            delta=len(grouped.get(mode, [])) - baseline.get(mode, 0) if baseline is not None else None,
        )
        for mode in modes
    ]
    return sorted(buckets, key=lambda bucket: (-bucket.count, _RANK[bucket.mode]))


def analyse(
    results: list[CaseResult],
    *,
    summary_pass_threshold: int,
    baseline_results: list[CaseResult] | None = None,
) -> ErrorAnalysis:
    failed = [result for result in results if not result.passed]
    hard = [(r.case_id, _hard_labels(r, summary_pass_threshold)) for r in failed]
    soft = [(r.case_id, _soft_labels(r)) for r in results]
    baseline = (
        _hard_counts(baseline_results, summary_pass_threshold) if baseline_results is not None else None
    )
    misroutes = Counter(_misroute_label(r) for r in failed if not r.error and not r.category_match)
    return ErrorAnalysis(
        total_cases=len(results),
        failed_cases=len(failed),
        buckets=_buckets(_group(hard), len(failed), baseline),
        soft_findings=_buckets(_group(soft), len(results), None),
        misroutes=dict(misroutes.most_common()),
        primary_mode={
            case_id: min((m for m, _ in labels), key=_RANK.__getitem__) for case_id, labels in hard
        },
        judge_unavailable=[r.case_id for r in results if _judge_unavailable(r)],
    )


def _md_cell(text: str) -> str:
    """Model output and parse errors come from untrusted email text."""
    flat = " ".join(text.split()).replace("|", "\\|").replace("`", "'")
    return flat if len(flat) <= _MD_DETAIL_MAX_CHARS else flat[: _MD_DETAIL_MAX_CHARS - 3] + "..."


def _delta_text(bucket: ModeBucket) -> str:
    return "-" if bucket.delta is None else f"{bucket.delta:+d}"


def render_markdown(analysis: ErrorAnalysis) -> str:
    header = f"#### Error analysis: {analysis.failed_cases}/{analysis.total_cases} cases failed"
    if not analysis.buckets:
        return f"{header}\n\nNo failed cases."
    lines = [
        header,
        "",
        "| Failure mode | Cases | Share of failures | vs baseline | Example |",
        "|---|---:|---:|---:|---|",
    ]
    for bucket in analysis.buckets:
        example = (
            f"`{bucket.examples[0].case_id}`: {_md_cell(bucket.examples[0].detail)}"
            if bucket.examples
            else ""
        )
        lines.append(
            f"| {bucket.mode.value} | {bucket.count} | {bucket.share_of_failures:.0%} | "
            f"{_delta_text(bucket)} | {example} |"
        )
    if analysis.misroutes:
        pairs = ", ".join(f"{pair} ({count})" for pair, count in analysis.misroutes.items())
        lines += ["", f"Misroutes: {pairs}"]
    if analysis.soft_findings:
        soft = ", ".join(f"{b.mode.value} ({b.count})" for b in analysis.soft_findings)
        lines += ["", f"Soft findings, not counted as failures: {soft}"]
    if analysis.judge_unavailable:
        lines += ["", f"Judge unavailable on {len(analysis.judge_unavailable)} cases; scored neutral."]
    return "\n".join(lines)


def render_text(analysis: ErrorAnalysis, limit: int = 5) -> str:
    """Terminal summary for `rg run` and `rg analyze`."""
    header = f"error analysis: {analysis.failed_cases}/{analysis.total_cases} cases failed"
    if not analysis.buckets:
        return f"{header}\n  No failed cases."
    lines = [header]
    for bucket in analysis.buckets[:limit]:
        delta = "" if bucket.delta is None else f"  ({bucket.delta:+d} vs baseline)"
        lines.append(f"  {bucket.mode.value:<24}{bucket.count:>4}  {bucket.share_of_failures:>4.0%}{delta}")
    if analysis.misroutes:
        top_pair, top_count = next(iter(analysis.misroutes.items()))
        lines.append(f"  most common misroute: {top_pair} ({top_count})")
    if analysis.soft_findings:
        soft = ", ".join(f"{b.mode.value} {b.count}" for b in analysis.soft_findings)
        lines.append(f"  soft findings: {soft}")
    return "\n".join(lines)
