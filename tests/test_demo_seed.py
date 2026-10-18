"""Seeding an empty run store so the hosted dashboard has history to show."""

from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import make_result, make_summary
from regression_guard.demo_seed import DEMO_PROMPTS, seed_demo_history
from regression_guard.store import RunStore


@pytest.mark.integration
def test_seed_fills_an_empty_store_with_one_run_per_demo_prompt(settings):
    store = RunStore(settings.db_path)
    assert seed_demo_history(store, settings) == len(DEMO_PROMPTS)
    runs = store.recent_runs()
    assert [run.prompt_version for run in reversed(runs)] == list(DEMO_PROMPTS)
    assert all(run.provider == "mock" for run in runs)
    assert all(store.get_case_results(run.run_id) for run in runs)


@pytest.mark.integration
def test_seed_leaves_existing_history_alone(settings):
    store = RunStore(settings.db_path)
    store.save_run(make_summary(run_id="real"), [make_result("c1", passed=True)])
    assert seed_demo_history(store, settings) == 0
    assert [run.run_id for run in store.recent_runs()] == ["real"]


@pytest.mark.integration
def test_seed_always_uses_the_offline_provider(settings):
    """A hosted demo must never spend tokens, whatever RG_PROVIDER says."""
    store = RunStore(settings.db_path)
    seed_demo_history(store, replace(settings, provider="openai"))
    assert {run.provider for run in store.recent_runs()} == {"mock"}
