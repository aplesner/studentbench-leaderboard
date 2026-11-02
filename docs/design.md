# Judge and leaderboard design

The project separates the judge, its evidence, and the human reference data.
`judge.Task` contains one or two plan JSON objects, one student's pretest, and
the exact rubric/scale definitions. A `Judge` implementation returns structured
scores and concise explanations. `LiteLLMJudge` is the initial provider adapter;
tests supply a deterministic synthetic implementation of the same interface.

Named two-plan judges now wrap that provider adapter in configurable strategies.
The public `compare_plans(plan_a, plan_b, judge_id, context=...)` API selects a
TOML definition and resolves its internal subversion. See
[configuring judges](judges.md) for single-call, order-check, repeated-vote, and
panel strategies, aggregation rules, and stored call-level evidence.

Each model call evaluates all criteria in the selected review. Single mode
returns a 1–5 score per criterion; pairwise mode returns −2, −1, 0, +1, or +2
relative to the presented A plan. `null` means the judge abstained. Response
validation rejects missing criteria, duplicate JSON keys, booleans, strings,
noninteger scores, invalid ranges, and empty explanations.

## Provenance and matching

Saved judgments include the instrument, subject and plan IDs, presented order,
criterion and scale IDs, actual input-file fingerprint, prompt version and
hash, judge configuration, returned model identifier, token usage, timestamp,
raw response, and a request hash. Exact completed requests are reusable.
Named judges derive subversions from prompt content and model/reasoning settings
automatically. Other strategy changes require a new public judge ID. The
legacy ad hoc path uses `PROMPT_VERSION`, which must be incremented when that
path's built-in prompt changes. Reports keep resolved versions separate.
Library dependencies are pinned in
`uv.lock`; a provider may still update an unversioned model alias.

Human agreement units are:

- Single: instrument, review subject, plan ID, question variant, rating scale.
- Pairwise: instrument, review subject, unordered pair of plan IDs, question
  variant, rating scale; preference signs are normalized to the sorted pair.

Human ratings are averaged within each unit. Repeated judge calls, including
opposite orientations when both are supplied, are also averaged within a unit.
Agreement compares these two means, with each matched unit weighted equally.
Pairwise direction agreement treats zero as its own tie category and uses the
direct human preference field. This is consensus agreement, not exact agreement
with every individual reviewer. Undefined correlations remain `null`.

All requested units and abstentions are counted, including units without an
available human match. Rankings can include unreviewed same-subject plan pairs;
agreement uses only actual matching human observations. Individual plan/model
scores are descriptive, with no confidence intervals or rank-significance
claims in this first implementation.

## Validation before selecting a judge

1. Split by review subject, keeping all plans, orientations, and reviews for a
   student's pretest together. Use a fixed development/test assignment. Do not
   choose prompts or models using held-out scores.
2. Establish human–human agreement and label coverage by instrument, criterion,
   scale version, and authoring model. Inspect disagreement examples locally.
3. Compare candidate judges on development subjects, including both A/B orders
   to measure position bias, and track abstentions, malformed outputs, latency,
   and cost. Examine per-criterion agreement before defining a composite score.
4. Freeze the judge model, prompt, model settings, and rubric, then evaluate on
   held-out subjects. Use resampling clustered by pretest subject for uncertainty;
   consider reviewer clustering as a sensitivity analysis.
5. Investigate comparison-schedule imbalance and graph connectivity before
   replacing descriptive win shares with Bradley–Terry or another pairwise
   ranking model. Keep learning gains separate from plan-quality ratings.

The CLI's `--limit` selection is deterministic and intended for small trial
runs. It does not implement a randomized split, representativeness guarantees,
or a validated benchmark protocol. Those require an explicit experiment
configuration once judge J and the evaluation scope are chosen.
