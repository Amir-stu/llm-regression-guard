"""Deterministic assertions: the checks that need no model to decide.

The judge is the expensive, noisy half of scoring. Everything that can be
decided by a rule is decided here instead, before and independently of it -
so a judge that is having a bad day cannot wave through an invented invoice
number, and a check that costs microseconds never waits on an API call.

Two severities:

* ``hard`` - the output broke the contract or said something false. The case
  fails regardless of category match or judge score.
* ``soft`` - the output is acceptable but shows a weakness worth counting,
  such as JSON the parser had to salvage. Soft findings never change a
  verdict; they feed error analysis so the trend is visible before it
  becomes a failure.

Known blind spots, kept deliberately simple rather than clever: numbers
written as words ("two") are not matched to digits, and an abbreviation
followed by a capitalised word ("Dr. Smith") reads as a sentence break.
"""

from __future__ import annotations

import json
import re

from .config import MAX_SUMMARY_CHARS
from .models import AssertionFailure, ClassificationOutput, GoldenCase, Severity

__all__ = [
    "CHECK_FORBIDDEN_CONTENT",
    "CHECK_MAX_CHARS",
    "CHECK_NUMBERS_GROUNDED",
    "CHECK_REFERENCE_DETAILS",
    "CHECK_REQUIRED_DETAIL",
    "CHECK_SINGLE_SENTENCE",
    "CHECK_STRICT_JSON",
    "Severity",
    "has_hard_failure",
    "run_assertions",
]

CHECK_STRICT_JSON = "strict_json"
CHECK_SINGLE_SENTENCE = "single_sentence"
CHECK_MAX_CHARS = "max_summary_chars"
CHECK_NUMBERS_GROUNDED = "numbers_grounded"
CHECK_REFERENCE_DETAILS = "reference_details_kept"
CHECK_REQUIRED_DETAIL = "must_include"
CHECK_FORBIDDEN_CONTENT = "must_not_include"

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _numbers(text: str) -> set[str]:
    """Digit tokens with thousands separators removed, so 2,000 == 2000."""
    return {token.replace(",", "") for token in _NUMBER_RE.findall(text)}


def _sentence_count(text: str) -> int:
    return len([part for part in _SENTENCE_BREAK_RE.split(text.strip()) if part.strip()])


def _fail(check: str, severity: Severity, detail: str) -> AssertionFailure:
    return AssertionFailure(check=check, severity=severity, detail=detail)


def _contract_checks(summary: str, raw: str) -> list[AssertionFailure]:
    failures: list[AssertionFailure] = []
    try:
        strict = isinstance(json.loads(raw.strip()), dict)
    except json.JSONDecodeError:
        strict = False
    if not strict:
        failures.append(
            _fail(CHECK_STRICT_JSON, Severity.SOFT, "response was not bare JSON; parser salvaged it")
        )
    sentences = _sentence_count(summary)
    if sentences > 1:
        failures.append(
            _fail(CHECK_SINGLE_SENTENCE, Severity.HARD, f"{sentences} sentences, contract is one")
        )
    if len(summary) > MAX_SUMMARY_CHARS:
        failures.append(
            _fail(CHECK_MAX_CHARS, Severity.HARD, f"{len(summary)} chars, limit is {MAX_SUMMARY_CHARS}")
        )
    return failures


def _grounding_checks(case: GoldenCase, summary: str) -> list[AssertionFailure]:
    failures: list[AssertionFailure] = []
    in_summary = _numbers(summary)
    invented = sorted(in_summary - _numbers(case.email))
    if invented:
        failures.append(
            _fail(CHECK_NUMBERS_GROUNDED, Severity.HARD, f"numbers not in the email: {', '.join(invented)}")
        )
    dropped = sorted(_numbers(case.expected_summary) - in_summary)
    if dropped:
        failures.append(
            _fail(CHECK_REFERENCE_DETAILS, Severity.SOFT, f"reference numbers missing: {', '.join(dropped)}")
        )
    return failures


def _case_checks(case: GoldenCase, summary: str) -> list[AssertionFailure]:
    lowered = summary.lower()
    missing = [term for term in case.must_include if term.lower() not in lowered]
    leaked = [term for term in case.must_not_include if term.lower() in lowered]
    failures: list[AssertionFailure] = []
    if missing:
        failures.append(_fail(CHECK_REQUIRED_DETAIL, Severity.HARD, f"missing: {', '.join(missing)}"))
    if leaked:
        failures.append(
            _fail(CHECK_FORBIDDEN_CONTENT, Severity.HARD, f"must not contain: {', '.join(leaked)}")
        )
    return failures


def run_assertions(case: GoldenCase, output: ClassificationOutput, raw: str) -> list[AssertionFailure]:
    """Every deterministic check for one parsed output, in a stable order."""
    summary = output.summary
    return [
        *_contract_checks(summary, raw),
        *_grounding_checks(case, summary),
        *_case_checks(case, summary),
    ]


def has_hard_failure(failures: list[AssertionFailure]) -> bool:
    return any(failure.severity is Severity.HARD for failure in failures)
