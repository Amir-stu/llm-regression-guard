"""Deterministic assertion layer: the checks that need no model to decide."""

from __future__ import annotations

import pytest

from regression_guard.assertions import (
    CHECK_FORBIDDEN_CONTENT,
    CHECK_MAX_CHARS,
    CHECK_NUMBERS_GROUNDED,
    CHECK_REFERENCE_DETAILS,
    CHECK_REQUIRED_DETAIL,
    CHECK_SINGLE_SENTENCE,
    CHECK_STRICT_JSON,
    Severity,
    has_hard_failure,
    run_assertions,
)
from regression_guard.models import AssertionFailure, Category, ClassificationOutput, Difficulty, GoldenCase


def make_case(**overrides) -> GoldenCase:
    base = {
        "id": "a1",
        "email": "Invoice 90211 bills 15 seats but we ordered 5. Please fix.",
        "expected_category": Category.BILLING,
        "expected_summary": "Invoice 90211 bills for 15 seats instead of 5.",
        "expected_difficulty": Difficulty.EASY,
        "notes": "assertion fixture",
    }
    base.update(overrides)
    return GoldenCase(**base)


def output(summary: str) -> ClassificationOutput:
    return ClassificationOutput(category=Category.BILLING, summary=summary)


def raw_for(summary: str) -> str:
    return f'{{"category": "billing", "summary": "{summary}"}}'


def checks(failures: list[AssertionFailure]) -> set[str]:
    return {failure.check for failure in failures}


@pytest.mark.unit
def test_clean_summary_passes_every_check():
    summary = "Invoice 90211 bills for 15 seats instead of the 5 ordered."
    assert run_assertions(make_case(), output(summary), raw_for(summary)) == []


@pytest.mark.unit
def test_invented_number_is_a_hard_hallucination_failure():
    summary = "Invoice 90211 bills for 15 seats instead of 5 and is due on 2026-11-30."
    failures = run_assertions(make_case(), output(summary), raw_for(summary))
    hallucination = next(f for f in failures if f.check == CHECK_NUMBERS_GROUNDED)
    assert hallucination.severity is Severity.HARD
    assert "2026" in hallucination.detail
    assert has_hard_failure(failures)


@pytest.mark.unit
def test_number_grounding_ignores_thousands_separators():
    case = make_case(email="Export fails on reports over 2,000 rows.", expected_summary="Export fails.")
    summary = "Export fails on reports over 2000 rows."
    assert CHECK_NUMBERS_GROUNDED not in checks(run_assertions(case, output(summary), raw_for(summary)))


@pytest.mark.unit
def test_two_sentences_break_the_one_sentence_contract():
    summary = "Invoice 90211 is wrong. It bills 15 seats instead of 5."
    failures = run_assertions(make_case(), output(summary), raw_for(summary))
    assert CHECK_SINGLE_SENTENCE in checks(failures)
    assert has_hard_failure(failures)


@pytest.mark.unit
def test_decimal_numbers_are_not_mistaken_for_sentence_breaks():
    case = make_case(email="App crashes on iOS 18.2 after the update.", expected_summary="App crashes.")
    summary = "App crashes on iOS 18.2 after the update."
    assert CHECK_SINGLE_SENTENCE not in checks(run_assertions(case, output(summary), raw_for(summary)))


@pytest.mark.unit
def test_overlong_summary_fails_the_length_contract():
    summary = "Invoice 90211 bills for 15 seats instead of 5 " + "and the customer is unhappy " * 8
    failures = run_assertions(make_case(), output(summary), raw_for(summary))
    assert CHECK_MAX_CHARS in checks(failures)


@pytest.mark.unit
def test_fenced_json_is_a_soft_finding_not_a_case_failure():
    """The parser salvages fenced JSON on purpose, so it must not fail the
    case - but it is still worth counting, because it means the prompt is not
    holding the output contract."""
    summary = "Invoice 90211 bills for 15 seats instead of 5."
    raw = f"```json\n{raw_for(summary)}\n```"
    failures = run_assertions(make_case(), output(summary), raw)
    strict = next(f for f in failures if f.check == CHECK_STRICT_JSON)
    assert strict.severity is Severity.SOFT
    assert not has_hard_failure(failures)


@pytest.mark.unit
def test_dropping_a_reference_number_is_a_soft_finding():
    summary = "Customer was billed for the wrong number of seats."
    failures = run_assertions(make_case(), output(summary), raw_for(summary))
    dropped = next(f for f in failures if f.check == CHECK_REFERENCE_DETAILS)
    assert dropped.severity is Severity.SOFT
    assert "90211" in dropped.detail


@pytest.mark.unit
def test_missing_required_detail_is_hard_and_case_insensitive():
    case = make_case(must_include=["INV-2291"])
    kept = "Customer wants receipt inv-2291 reissued."
    dropped = "Customer wants a receipt reissued."
    assert CHECK_REQUIRED_DETAIL not in checks(run_assertions(case, output(kept), raw_for(kept)))
    failures = run_assertions(case, output(dropped), raw_for(dropped))
    required = next(f for f in failures if f.check == CHECK_REQUIRED_DETAIL)
    assert required.severity is Severity.HARD
    assert "INV-2291" in required.detail


@pytest.mark.unit
def test_forbidden_content_is_a_hard_failure():
    case = make_case(
        email="Card ends 4471 and payment failed.",
        expected_summary="Card payment failed.",
        must_not_include=["4471"],
    )
    summary = "Payment failed for the card ending 4471."
    failures = run_assertions(case, output(summary), raw_for(summary))
    assert CHECK_FORBIDDEN_CONTENT in checks(failures)
    assert has_hard_failure(failures)


@pytest.mark.unit
def test_case_without_numbers_or_constraints_only_runs_contract_checks():
    case = make_case(email="Please help with my invoice.", expected_summary="Customer needs invoice help.")
    summary = "Customer needs help with an invoice."
    assert run_assertions(case, output(summary), raw_for(summary)) == []


@pytest.mark.unit
def test_has_hard_failure_is_false_for_an_empty_list():
    assert has_hard_failure([]) is False
