"""Guards on the golden dataset itself.

Evaluation quality is bounded by data quality, so the dataset gets the same
CI treatment as the code: shape, coverage, and the invariants that make the
scores mean anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from regression_guard.config import DEFAULT_DATASET
from regression_guard.models import Category, Difficulty, GoldenDataset
from regression_guard.runner import load_dataset


@pytest.fixture(scope="module")
def golden() -> GoldenDataset:
    return load_dataset(DEFAULT_DATASET)


@pytest.mark.unit
def test_dataset_parses_and_meets_minimum_size(golden):
    assert len(golden.cases) >= 50


@pytest.mark.unit
def test_every_category_is_represented_at_least_ten_times(golden):
    counts = dict.fromkeys(Category, 0)
    for case in golden.cases:
        counts[case.expected_category] += 1
    for category, count in counts.items():
        assert count >= 10, f"{category.value} has only {count} cases"


@pytest.mark.unit
def test_hard_cases_are_at_least_fifteen_percent(golden):
    """A set of only easy cases cannot detect a regression worth catching."""
    hard = sum(1 for c in golden.cases if c.expected_difficulty is Difficulty.HARD)
    assert hard / len(golden.cases) >= 0.15


@pytest.mark.unit
def test_every_case_has_notes_explaining_why_it_exists(golden):
    missing = [c.id for c in golden.cases if len(c.notes.strip()) < 10]
    assert missing == [], f"cases without a rationale: {missing}"


@pytest.mark.unit
def test_case_ids_are_unique_and_stable_shaped(golden):
    ids = [c.id for c in golden.cases]
    assert len(ids) == len(set(ids))
    assert all("-" in case_id for case_id in ids)


@pytest.mark.unit
def test_expected_summaries_are_single_sentence_and_bounded(golden):
    """Counting periods would be wrong - 'iOS 18.2' is not a sentence break -
    so split on an actual boundary: a terminator followed by whitespace."""
    for case in golden.cases:
        summary = case.expected_summary.strip()
        assert summary.endswith("."), case.id
        assert len(re.split(r"(?<=[.!?])\s+", summary)) == 1, case.id
        assert len(summary) <= 220, case.id


@pytest.mark.unit
def test_acceptable_categories_never_repeat_the_expected_one(golden):
    for case in golden.cases:
        assert case.expected_category not in case.acceptable_categories, case.id


@pytest.mark.unit
def test_ambiguous_cases_document_the_ambiguity(golden):
    """If a case accepts two labels, the notes must say why, or the next person
    maintaining the set will 'fix' the ambiguity by deleting it."""
    for case in golden.cases:
        if case.acceptable_categories:
            assert len(case.notes) > 40, f"{case.id} accepts multiple labels without explanation"


@pytest.mark.unit
def test_accepts_helper_honours_alternatives(golden):
    ambiguous = next(c for c in golden.cases if c.acceptable_categories)
    assert ambiguous.accepts(ambiguous.expected_category)
    assert ambiguous.accepts(ambiguous.acceptable_categories[0])
    wrong = next(
        c
        for c in Category
        if c is not ambiguous.expected_category and c not in ambiguous.acceptable_categories
    )
    assert not ambiguous.accepts(wrong)


@pytest.mark.unit
def test_dataset_file_is_versioned(golden):
    assert golden.dataset_version
    assert golden.description


@pytest.mark.unit
def test_rejects_duplicate_ids(tmp_path: Path):
    payload = {
        "dataset_version": "x",
        "created_at": "2026-01-01",
        "cases": [
            {
                "id": "dup",
                "email": "a",
                "expected_category": "billing",
                "expected_summary": "x.",
                "expected_difficulty": "easy",
                "notes": "n",
            },
            {
                "id": "dup",
                "email": "b",
                "expected_category": "billing",
                "expected_summary": "y.",
                "expected_difficulty": "easy",
                "notes": "n",
            },
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="duplicate case id"):
        load_dataset(path)
