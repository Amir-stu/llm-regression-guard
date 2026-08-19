"""The LLM feature under test: support-email triage.

Deliberately small. The point of this repo is not the feature, it is
everything that watches the feature. Keep the surface here at one function
so the eval engine has a single, stable contract to call.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .models import Category, ClassificationOutput, PromptConfig
from .providers import LLMProvider

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class ClassificationError(RuntimeError):
    """Raised when the model output cannot be coerced into the contract."""


@dataclass(frozen=True)
class ClassificationCall:
    output: ClassificationOutput
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    raw: str


def build_messages(config: PromptConfig, email: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for example in config.few_shot:
        messages.append({"role": "user", "content": config.user_template.format(email=example.email)})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"category": example.category.value, "summary": example.summary}
                ),
            }
        )
    messages.append({"role": "user", "content": config.user_template.format(email=email)})
    return messages


def parse_output(raw: str) -> ClassificationOutput:
    """Coerce model text into the output contract.

    Real models wrap JSON in prose or fences often enough that a bare
    ``json.loads`` makes the eval measure formatting luck rather than
    classification quality. We salvage the first JSON object, then validate
    strictly - anything that still fails is a real failure worth counting.
    """
    text = (raw or "").strip()
    if not text:
        raise ClassificationError("empty model response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ClassificationError(f"no JSON object in response: {text[:120]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"malformed JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ClassificationError("response JSON is not an object")

    category = str(data.get("category", "")).strip().lower()
    if category not in {c.value for c in Category}:
        raise ClassificationError(f"unknown category: {category!r}")
    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ClassificationError("missing summary")
    return ClassificationOutput(category=Category(category), summary=summary[:400])


async def classify_email(
    email: str,
    config: PromptConfig,
    provider: LLMProvider,
    model: str,
) -> ClassificationCall:
    started = time.perf_counter()
    response = await provider.complete(
        config.system_prompt,
        build_messages(config, email),
        model=model,
        temperature=config.temperature,
        json_mode=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return ClassificationCall(
        output=parse_output(response.text),
        latency_ms=latency_ms,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        raw=response.text,
    )
