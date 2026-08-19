"""LLM backends.

Two implementations sit behind one protocol:

* ``OpenAIProvider``  - the real thing, used in CI and locally when a key is set.
* ``MockProvider``    - a deterministic, offline stand-in.

The mock exists because an eval harness that cannot run without a funded API
key is useless in unit tests and useless to anyone reading the repo. It is not
a stub that returns a constant: it derives its behaviour from the prompt under
test, so a weaker prompt genuinely produces weaker output and the regression
machinery has something real to detect. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("invoice", "charge", "refund", "payment", "price", "bill"),
    "technical": ("error", "bug", "crash", "broken", "fail"),
    "account": ("password", "login", "account", "profile"),
    "general": ("hello", "thanks", "question"),
}

_WORD_RE = re.compile(r"[a-z']+")

# Calibration for the offline judge. Reference summaries are abstractive while
# the mock's summaries are extractive, so their word overlap tops out well
# below 1.0 even for a summary a human would call correct. Saturating the
# scale at this overlap keeps the mock judge's 1-5 output roughly aligned with
# what a real judge returns, which is what makes offline runs comparable in
# shape to API runs. Recalibrate this if the mock summariser changes.
MOCK_JUDGE_SATURATION = 0.55


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> LLMResponse: ...


def _tokens(text: str) -> int:
    """Rough token estimate. Only used by the mock, and only for trend shape."""
    return max(1, len(text) // 4)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class MockProvider:
    """Offline provider whose accuracy is a function of the prompt it is given.

    Category choice is keyword voting. The keyword table is *extracted from the
    system prompt itself* - a prompt that spells out what belongs in each
    bucket gives the mock more to work with, exactly as a richer prompt gives a
    real model more to work with. A prompt that lists no cues falls back to a
    thin default table and misclassifies the harder cases.

    Summaries are extractive. A prompt that asks for one sentence gets one
    sentence; a prompt that does not gets a blunt truncation, which the judge
    marks down.
    """

    name = "mock"

    def __init__(self, seed: str = "regression-guard") -> None:
        self.seed = seed
        self.calls = 0

    @staticmethod
    def extract_keyword_table(system_prompt: str) -> dict[str, tuple[str, ...]]:
        """Pull ``category: cue, cue, cue`` lines out of a system prompt."""
        table: dict[str, list[str]] = {}
        for raw_line in system_prompt.splitlines():
            line = raw_line.strip().lstrip("-* ").strip()
            if ":" not in line:
                continue
            head, _, tail = line.partition(":")
            key = head.strip().strip("`").lower()
            if key not in DEFAULT_KEYWORDS:
                continue
            cues = [c.strip().strip("`.").lower() for c in tail.split(",")]
            cues = [c for c in cues if c and len(c) > 2]
            if cues:
                table.setdefault(key, []).extend(cues)
        if not table:
            return dict(DEFAULT_KEYWORDS)
        merged = {k: tuple(v) for k, v in DEFAULT_KEYWORDS.items()}
        for key, cues in table.items():
            merged[key] = tuple(dict.fromkeys(merged[key] + tuple(cues)))
        return merged

    def _classify(self, email: str, table: dict[str, tuple[str, ...]]) -> str:
        haystack = email.lower()
        scores: dict[str, float] = {}
        for category, cues in table.items():
            hit = 0.0
            for cue in cues:
                if cue in haystack:
                    # Multi-word cues are more specific, so weight them higher.
                    hit += 1.0 + 0.5 * cue.count(" ")
            scores[category] = hit
        best = max(scores.values())
        if best == 0:
            return "general"
        # Deterministic tie-break seeded by the email so results are stable.
        winners = sorted(c for c, s in scores.items() if s == best)
        if len(winners) == 1:
            return winners[0]
        digest = hashlib.sha256((self.seed + email).encode()).hexdigest()
        return winners[int(digest, 16) % len(winners)]

    @staticmethod
    def _summarise(email: str, one_sentence: bool) -> str:
        cleaned = " ".join(email.split())
        if one_sentence:
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            body = next(
                (s for s in sentences if len(_words(s)) >= 4),
                sentences[0] if sentences else cleaned,
            )
            return f"Customer reports: {body.strip()}"[:280]
        return cleaned[:120]

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> LLMResponse:
        self.calls += 1
        await asyncio.sleep(0)
        email = messages[-1]["content"]

        if "JUDGE" in system.upper():
            return self._judge(email)

        table = self.extract_keyword_table(system)
        lowered = system.lower()
        one_sentence = "one sentence" in lowered or "one-sentence" in lowered
        payload = {
            "category": self._classify(email, table),
            "summary": self._summarise(email, one_sentence),
        }
        text = json.dumps(payload)
        return LLMResponse(text, _tokens(system + email), _tokens(text))

    @staticmethod
    def _judge(payload: str) -> LLMResponse:
        """Score a candidate summary by content overlap with the reference.

        Crude on purpose: the point of the mock judge is to be monotonic in
        summary quality, not to imitate a frontier model's taste.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return LLMResponse(json.dumps({"score": 1, "reason": "unparseable"}), 1, 1)
        stop = {"the", "a", "an", "is", "to", "and", "of", "for", "in", "on", "it", "customer"}
        reference = set(_words(data.get("reference", ""))) - stop
        candidate = set(_words(data.get("candidate", ""))) - stop
        if not data.get("candidate", "").strip():
            score = 1
        elif not reference:
            score = 3
        else:
            overlap = len(reference & candidate) / len(reference)
            scaled = min(1.0, overlap / MOCK_JUDGE_SATURATION)
            score = 1 + min(4, int(round(scaled * 4)))
        out = json.dumps({"score": score, "reason": f"overlap-based score {score}/5"})
        return LLMResponse(out, _tokens(payload), _tokens(out))


class OpenAIProvider:
    """Thin async wrapper over the OpenAI chat completions API."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - only hit without the dep
            raise RuntimeError(
                "openai package is not installed. `pip install openai` or use RG_PROVIDER=mock."
            ) from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use RG_PROVIDER=mock to run offline.")
        self._client = AsyncOpenAI(api_key=key, timeout=timeout, max_retries=max_retries)

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def build_provider(name: str, **kwargs: Any) -> LLMProvider:
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAIProvider(**kwargs)
    raise ValueError(f"unknown provider: {name!r} (expected 'mock' or 'openai')")
