# StudentBench leaderboard roadmap

## v1: first public leaderboard

Include five new lesson-plan generators alongside the 12 existing presets:
Claude Haiku 5.5, Kimi K3.1 Pro, GLM 5.3, DeepSeek v4.1 Flash, and Gemini 4 Argon.
Freeze exact provider model IDs, generation settings, benchmark inputs and
judging policy. The first release therefore targets 17 systems.

GPT-6.1 Terra and MiMo v2.6 Flash/Pro were also requested, then explicitly deferred
by the user because their exact model aliases were unavailable on our Tailscale
proxy. Add them in a recorded later update once approved routes are available.
All our release inference goes through Tailscale; public CLI users retain the
OpenRouter/native-key defaults.

Fit one combined authoring-model leaderboard, including comparisons between new
generators. Put this leaderboard at the top of the report, with uncertainty,
coverage, generation conditions and a clear AI-judged label. Preserve the judge
validation study below it. Missing models or judgments remain explicit; they do
not receive invented scores. Prepare a public aggregate report separately from
private lesson plans, student context, saved requests and credentials.

The authorized total inference cap for preparing this release is $1,000.
Record attempts, usage, cost estimates, failures and cache reuse. Access failures
must be resolved through approved credentials or routes; do not substitute model
identities or change account policies to finish the table.

## v1.1: decision models as judges

Test **Jev and other decision models for judging lesson plans**. This is a
judge-method experiment, separate from adding lesson-plan generators. Do not
change the frozen v1 judge or rewrite its results while exploring alternatives.

1. Verify the exact Jev model/version and available decision-model alternatives,
   supported API contracts, access, settings and pricing. Name the tested
   systems precisely; do not assume a text-generation interface or output scale.
2. Add small judge adapters that map the models' actual outputs onto the existing
   A/B/tie/insufficient-evidence contract. Validate direction, missingness,
   uncertainty semantics and provenance. A provider error is never a tie.
3. Pilot criterion-specific decisions and whole-plan decisions on identical
   lesson-plan pairs. Check whether the model can use the complete student
   pretest and rubric, and whether decision granularity changes the ranking.
4. Compare the decision-model judges with the frozen v1 judge using human-rated
   comparisons. Prefer fresh held-out students for a confirmatory result.
   Experiments on already exposed study cohorts remain labeled exploratory.
5. Evaluate authoring-model rank agreement (Kendall tau-b and Spearman),
   model-order agreement, per-criterion agreement, coverage/abstention, A/B
   position sensitivity, repeated-call stability, latency and cost. Report
   student-cluster uncertainty and the support behind every estimate.
6. Freeze the v1.1 selection rule before opening its held-out results. Include
   grading-density experiments and any judge panels only when they answer a
   stated cost or accuracy question. Keep comparisons paired on the same inputs.

Completion means a reproducible adapter, offline contract tests, a bounded live
pilot and an explicit human-validation report. A new judge becomes an optional
versioned configuration first. Regrading a public leaderboard requires a
separate recorded version and an explanation of changes from v1.

## Further work

- Repeat generator runs to measure generation variability, beyond resampling
  students with fixed saved plans.
- Extend the validated workflow to GRE Verbal with its own rubric/context
  preparation and human-validation evidence.
- Provide checksum-verified benchmark distribution and public aggregate exports
  so external users can reproduce runs from a fresh checkout.

These items are future work, not requirements to expand the first release.
