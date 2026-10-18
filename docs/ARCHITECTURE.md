# Architecture

Notes on why this is built the way it is. The README covers how to use it; this covers the
decisions and the things that were tried and rejected.

## Pipeline

```
prompts/*.yaml ──┐
                 ├──> runner ──> per-case results ──┐
data/*.json    ──┘      │                           │
                        │                           ├──> compare ──> verdict ──> exit code
                     provider                       │        │
                   (openai|mock)                    │        ├──> report (HTML)
                        │                           │        └──> slack
              assertions + judge               store (SQLite)
                                                    │
                                                    ├──> drift (rolling window)
                                                    ├──> error analysis (failure modes)
                                                    └──> dashboard (Streamlit, read-only)
```

The runner produces numbers. `compare` decides whether those numbers should stop a merge.
Everything else is transport.

## Data model

`models.py` is the contract between layers, and `SCHEMA_VERSION` is bumped alongside any change
to it because stored runs are read back by the comparison layer. Pydantic with `extra="forbid"`
throughout: a typo'd field in a prompt YAML or a dataset entry should fail loudly at load time,
not silently evaluate the wrong thing.

`RunSummary` is denormalised on purpose — per-category and per-difficulty breakdowns are
computed at write time and stored as JSON columns. Recomputing them from `case_results` on every
report render would mean the report and the stored verdict could disagree after a scoring change.

## Scoring

Four dimensions per case: category match (binary), judge score (1–5), latency, tokens.

A case passes on `category_match AND judge_score >= threshold AND no hard assertion failed`. The AND matters — a correct
category with a summary that says the opposite of the email is not a pass, and averaging the two
into a single number would let one hide the other.

The composite score weights category 0.7 and normalised judge 0.3. Rationale: category accuracy
is objectively checkable against a human label, judge score is a model's opinion. Weighting them
equally would let judge drift move the headline number as much as real classification quality.

### Ambiguous cases

`acceptable_categories` exists because roughly one case in eight in this domain is genuinely
two-way — a failing payment could be billing or a processor bug; SSO is both a login problem and
an integration fault. Three options were considered:

1. Pick one label and accept the eval punishing a defensible answer. Rejected: it puts noise into
   the headline number permanently.
2. Drop ambiguous cases from the set. Rejected: the ambiguous cases are the ones where prompt
   changes actually show up. Removing them makes the eval measure the easy half of the problem.
3. Accept either label and record the ambiguity in `notes`. Chosen.

The tests enforce that any case with `acceptable_categories` has a substantive `notes` field, so
the reason survives the next person who maintains the file.

## Deterministic assertions

`assertions.py` runs before the judge result is used and does not depend on it. The split is
deliberate: anything a rule can decide, a rule decides. A rule costs nothing, never drifts, and
cannot be persuaded by fluent text. The judge is kept for what rules cannot do, namely whether
the summary captures the request.

Hard checks fail the case. Soft checks are recorded and counted, and never change a verdict.
`strict_json` is soft because `parse_output` salvages fenced JSON on purpose, so failing the case
on it would undo that decision. Counting it still shows when a prompt is losing hold of the
format.

`numbers_grounded` is the cheapest hallucination check that is actually useful in support
triage: every digit token in the summary must appear in the email. It misses numbers written as
words. That is accepted rather than patched with a number-word parser that would add false
negatives of its own.

The schema moved to version 2 with an `assertion_failures` column on `case_results`.
`RunStore._migrate` adds the column in place on open, because run history is the baseline and
an upgrade must never require deleting it.

## Error analysis

`error_analysis.py` maps each failed case to failure modes from signals already recorded, with
no extra model calls. That is why `rg analyze` is free to run on any historical run. Two counts
are kept on purpose: multi-label buckets, which say how often each problem occurs, and one
primary mode per case by severity, which partitions the failures. Ranking uses the first. A
partition-only view would hide that most misrouted cases in v3 also break the format contract.

Rejected: asking the judge model to label failure modes. It would make the taxonomy as noisy as
the judge, cost a call per failure, and make `rg analyze` non-reproducible. Open-ended labelling
by a model is useful for discovering *new* modes in a sample. Once a mode is named, it should
become a rule here.

## Statistics

`mcnemar_exact_p(regressions, improvements)` — two-sided exact binomial over the discordant
pairs.

The alternative considered was a two-proportion z-test on the two pass rates. It is wrong here:
both runs are scored on the *same* cases, so the samples are paired. Treating them as independent
inflates the variance and would classify real regressions as noise. McNemar conditions on exactly
the cases that changed behaviour, which is also the quantity a human reviewer wants to look at.

Exact rather than the chi-squared approximation because the discordant counts are small — with 60
cases a typical flip count is under 10, where the approximation is unreliable.

Worked values (used as test fixtures in `tests/test_compare.py`):

| regressions | improvements | p | significant at 0.05 |
|---:|---:|---:|---|
| 5 | 0 | 0.0625 | no |
| 6 | 0 | 0.03125 | yes |
| 8 | 2 | 0.1094 | no |
| 9 | 1 | 0.0215 | yes |

The 5-vs-6 boundary is why the threshold system and the significance test are combined rather
than either alone: five one-directional flips on 60 cases is a real drop in pass rate but is not
yet distinguishable from chance.

### Verdict logic

```
delta <= -critical_delta                       -> CRITICAL   (size alone)
delta <= -warn_delta   AND significant         -> CRITICAL   (modest but real)
delta <= -warn_delta   AND NOT significant     -> WARN
delta  >  -warn_delta  AND regressions AND significant -> WARN  (behaviour shifted, headline did not)
otherwise                                      -> PASS
```

The fourth line is the case a threshold-only system misses entirely: nine cases break, one
recovers, pass rate is unchanged. Aggregate says nothing happened; the flip pattern says the
model is behaving differently.

## Drift detection

Separate module, separate trigger, deliberately non-blocking.

Two conditions, either fires:

- **Floor** — the `drift_window`-run rolling mean of the composite score is below
  `drift_floor`.
- **Slope** — the second half of the window is worse than the first half, *and* the newest run is
  below the window mean, *and* the window mean is below the previous window's mean.

The three-way AND on the slope condition is there so a single noisy run cannot trigger it;
`test_drift_ignores_a_single_noisy_dip` pins that.

Why it does not block: the commit under review is almost never the commit that caused drift. A
system that fails an innocent PR for a slide that started six weeks ago gets disabled within a
month. It warns the team instead.

## Baseline selection

The subtle one. `RunStore.latest_baseline` returns the most recent run whose
(prompt_version, model) pair differs from the current run's, falling back to the most recent
other run.

Naive "previous run" has a failure mode that defeats the whole point: a red PR, re-run, sees its
own failing run as the baseline, computes a zero delta, and passes. The pipeline would literally
reward pressing retry. `test_rerunning_a_failing_prompt_stays_red` covers it.

This also handles the model-swap case correctly: same prompt, new model id is a different
configuration, so it diffs against the same prompt on the old model.

## Provider abstraction

One `Protocol` with a single `complete()` method. Two implementations.

`MockProvider` derives behaviour from the prompt under test rather than returning a fixed
response. It parses `category: cue, cue, cue` lines out of the system prompt and classifies by
weighted keyword voting, so prompt quality maps onto output quality. Multi-word cues score higher
because they are more specific. Ties break on a hash of the email, so runs are deterministic.

This is a simulation and the code says so. What it buys: the full pipeline, including the
regression detection, is exercised in CI on every commit at zero cost, and anyone cloning the
repo sees the system work before deciding whether to spend on an API key. The failure mode it
does *not* cover is real model nondeterminism at temperature 0, which is a genuine limitation —
against the real API, a single flipped case can be sampling variance rather than a regression,
which is a second reason the significance test is there.

The mock judge scores by content-word overlap against the reference, saturating at
`MOCK_JUDGE_SATURATION = 0.55`. Extractive summaries never reach full overlap with abstractive
references, so without the calibration constant every summary scores 2/5 and the pass rate is
meaningless. The constant is documented at its definition rather than buried.

## Failure handling

The rule: nothing outside the model's actual output quality may fail the build.

- A case that throws is recorded as a failed case with the exception text, and the run continues.
  `error_count` is reported, and a rise in it is called out in the verdict reasons.
- An unparseable judge response scores a neutral 3 with the reason recorded.
- A Slack delivery failure is logged and swallowed.
- A missing webhook is an info log, not an error.

Model output parsing is deliberately forgiving in one specific way: `parse_output` salvages the
first JSON object out of prose or code fences. Real models pad JSON often enough that strict
parsing would make the eval measure formatting luck rather than classification quality. It then
validates strictly — an unknown category or empty summary is a real failure and is counted.

## Reporting

Single self-contained HTML file: no CDN, no framework, no build step. It has to open from a CI
artifact zip on a machine with no network. The trend chart is inline SVG generated in
`report.py` for the same reason.

Every piece of model output is escaped before it reaches the template. The summaries are
model-generated text derived from customer email, which is untrusted input; the report must not
be able to execute anything a customer wrote. `test_report_escapes_model_output` covers it.

Light and dark are both handled via `prefers-color-scheme`, and wide tables scroll inside their
own container rather than breaking the page.

## What is not here

Honest list of gaps, in rough priority order:

- **No inter-rater agreement check on the judge.** The judge is trusted without ever being
  validated against a human. The right fix is to hand-score a sample of summaries and measure
  agreement, then track that number over time. Without it, judge drift is invisible.
- **The mock judge rewards truncation.** It scores word overlap, and truncated email text
  overlaps the reference well, so `v3-terse` raises the mean judge score while failing 43
  format checks. The assertion layer covers this case. The general lesson is the gap above:
  any judge needs validating against human scores.
- **No cost tracking in currency.** Tokens are recorded per case but never priced, so the
  cost/quality trade-off of a prompt change has to be eyeballed.
- **No retry/variance runs.** Against a real API, running each case once conflates sampling
  variance with regression. Running each case *k* times and comparing distributions would be the
  proper fix; the significance test is a partial mitigation, not a substitute.
- **Baseline lives in a CI cache.** A cache eviction silently degrades to "no baseline", which
  passes. Storing run history in a dedicated branch or an external store would be more durable.
- **Single dataset.** No held-out split, so a prompt tuned against this file can overfit it. The
  practical guard is that cases are added from production failures rather than from tuning
  sessions.
