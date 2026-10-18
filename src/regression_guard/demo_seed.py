"""Seed an empty run store with the offline demo story.

Run history is gitignored, so a fresh deployment of the dashboard starts with
nothing to show. Seeding replays the same three prompt versions as
`scripts/demo.py` through the mock provider: zero cost, deterministic, and
clearly labelled as simulated in the dashboard. A store that already holds
any run is never touched, so real history always wins.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from .config import Settings
from .prompts import resolve_prompt
from .providers import MockProvider
from .runner import load_dataset, run_eval
from .store import RunStore

DEMO_PROMPTS: tuple[str, ...] = ("v1-baseline", "v2-tightened", "v3-terse")


def seed_demo_history(store: RunStore, settings: Settings) -> int:
    """Record one mock run per demo prompt if the store is empty. Returns runs added."""
    if store.recent_runs(limit=1):
        return 0
    offline = replace(settings, provider="mock")
    dataset = load_dataset(offline.dataset_path)
    provider = MockProvider()
    for version in DEMO_PROMPTS:
        config = resolve_prompt(offline.prompt_dir, version)
        summary, results = asyncio.run(run_eval(config, dataset, provider, offline))
        store.save_run(summary, results)
    return len(DEMO_PROMPTS)
