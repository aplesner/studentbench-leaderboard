# Paired uncertainty for judge comparisons

The extension's primary contrast is the one-example Terra-high judge minus
the byte-verified cached zero-example Terra-high judge on the same 120 validation
pairs. Provider and panel contrasts against the same Terra control are secondary
exploratory comparisons. This protocol was implemented before inspecting new
judgment outcomes and does not select an extension winner.

`studentbench_leaderboard.paired_deltas.run_paired_deltas(study)` runs offline
after the two main analyses and writes `study/paired_deltas.json`. It requires
all executable calls to have completed or have a verified terminal-failure
declaration after at least three distinct failed API attempts; documented
unavailable variants remain separate. Missing outcomes are never imputed.
It verifies the same pair cohort and human reference across schedules,
reconstructs the same normalized votes and fixed panel recipes, and checks every
point metric against the completed main analysis.

The explicit `include_provider=False` option writes the completed example phase
without opening provider outcomes or requiring that phase to finish. The output
marks `complete_phases`, `pending_phases`, and an example-complete/provider-pending
status. A later default call adds provider contrasts; it uses exactly the same
example subject resamples and produces the same example intervals. This execution
option does not change the statistical protocol.

The terminal-failure execution amendment uses the same shared evidence validator
as final verification. Each phase validates only its own declarations, preserving
the example-only path's separation from provider outcomes. Terminal jobs remain
missing in the existing rows/coverage and do not shrink the scheduled human
reference. The output exposes the declaration file hash and details; there is no
change to candidates, baseline, subject draws, model universe, ridge or metrics.

Each of 500 seeded draws samples the full scheduled human-reference subject
cohort with replacement. That exact subject multiplicity applies to the human,
cached Terra, and every candidate's complete set of within-subject comparisons.
Repeated human ratings and rubric families are averaged before resampling;
multiplying the resulting pair weights retains every sampled cluster copy.
The cohort never shrinks to each candidate's successful subjects. Missing AI
comparisons remain missing, and new unreviewed pairs cannot enter this analysis.

All fits use the same authoring-model universe and Bradley–Terry ridge 1.
Each candidate's alignment to the resampled human leaderboard is compared with
Terra's alignment to that same reference. The reported raw difference is always
candidate minus baseline. Larger differences favor the candidate for Kendall
tau-b, Spearman rho, and model-order agreement; smaller differences favor it for
rank MAE. Each metric reports a 95% percentile interval and the fractions of
valid resamples with positive, negative, or zero differences. The separate
`proportion_improved` follows the metric's direction. These proportions are
descriptive resampling frequencies, **not p-values**.

Disconnected or unconverged human/baseline/candidate fits do not produce ranking
deltas. Their counts remain visible for each comparison; metric-specific valid
counts also exclude undefined correlations from entirely tied ranks. Interval
and sign summaries condition on the resulting usable draws. Absent panels have
unavailable metrics and zero usable draws, not zero disagreement.

This uncertainty conditions on the fixed observed model generations and one
deliberately selected human-reviewed development example. It does not resample
LLM generations, examples, backend changes, or human reviewers globally across
subjects. The cached baseline and new variant were generated at different times.
Provider/panel contrasts are multiple exploratory comparisons without adjustment
for multiplicity. The validation cohort was reused after the original test
was inspected, so these intervals do not supply a fresh confirmatory test.
