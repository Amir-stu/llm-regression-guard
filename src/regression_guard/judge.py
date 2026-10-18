"""LLM-as-judge scoring for the free-text half of the output.

Category match is binary and objective. Summary quality is not, so a second
model rates the candidate summary against the hand-written reference on a
1-5 scale. The judge sees the reference, which turns an open-ended taste
question into a much narrower comparison - judges agree with each other far
more often when they are grading against a target than when grading in the
abstract.

The judge is pinned to its own model id (RG_JUDGE_MODEL) so that changing the
model under test does not silently change the ruler measuring it.
"""

from __future__ import annotations

import json
import re

from .providers import LLMProvider

JUDGE_SYSTEM_PROMPT = """You are a JUDGE scoring customer-support email summaries.

You receive JSON with three fields: `email`, `reference` (a human-written ideal
summary) and `candidate` (a machine-written summary).

Score the candidate 1-5 on whether it captures the same core request as the
reference:
5 - same core request, accurate, one clear sentence
4 - same core request, minor omission or clumsy phrasing
3 - broadly right but vague, or includes a wrong secondary detail
2 - misses the core request or states something the email does not say
1 - unrelated, empty, or a verbatim dump of the email

Judge content, not style. Do not reward extra length.
Reply with JSON only: {"score": <1-5>, "reason": "<max 20 words>"}"""

# Reason prefixes for judge outcomes that say nothing about output quality.
# Error analysis keys on these to separate infrastructure noise from failures.
JUDGE_CALL_FAILED = "judge call failed"
JUDGE_UNPARSEABLE = "judge response unparseable"

_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*([1-5])')


def parse_judge_response(raw: str) -> tuple[int, str]:
    """Extract (score, reason). Falls back to a regex, then to a neutral 3.

    A judge that fails to answer must not be able to fail the build on its
    own - an unparseable judge reply is an infrastructure problem, not a
    quality regression, so it lands mid-scale and is visible in the report.
    """
    text = (raw or "").strip()
    try:
        data = json.loads(text)
        score = int(data["score"])
        reason = str(data.get("reason", ""))[:200]
        if 1 <= score <= 5:
            return score, reason
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    match = _SCORE_RE.search(text)
    if match:
        return int(match.group(1)), "recovered from unstructured judge reply"
    return 3, f"{JUDGE_UNPARSEABLE}; scored neutral"


async def judge_summary(
    email: str,
    reference: str,
    candidate: str,
    provider: LLMProvider,
    model: str,
) -> tuple[int, str]:
    payload = json.dumps({"email": email, "reference": reference, "candidate": candidate})
    response = await provider.complete(
        JUDGE_SYSTEM_PROMPT,
        [{"role": "user", "content": payload}],
        model=model,
        temperature=0.0,
        json_mode=True,
    )
    return parse_judge_response(response.text)
