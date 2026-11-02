# Evaluating rankings of lesson-plan generation models

The target is the ordering of **generation model presets**, including their generation reasoning settings. Individual lesson plans supply comparisons; they are not the ranked entities. A judge can disagree with individual human labels and still recover the same aggregate model ordering.

## Observation and weighting

`ranking.aggregate_pairs` accepts dictionaries with `subject`, `pair_id`, `model_a`, `model_b`, and `value`, where 1 means A wins, 0.5 means a tie, and 0 means B wins. A `None` value is an abstention and is omitted. Signed preferences map to these three values; preference magnitude does not enter the primary estimator. Optional `criterion` identifies the exact question variant and rating scale. Optional `criterion_family` groups those variants into a stable rubric item such as q01–q08.

For records produced by `compare_plans` or the CLI, `ranking.judgment_rows(dataset, records)` validates their SSOT identities and creates these criterion rows. Pass one judge version at a time. The adapter prefers `preference_probabilities` when present, so balanced-order or panel disagreements can contribute fractional win shares even when the strict-majority categorical score is inconclusive. An explicit missing probability remains missing; it never falls back to a categorical score. A fractional value of 0.5 can therefore mean averaged opposing votes as well as explicit ties. The source is retained on each criterion row.

Aggregation first averages repeated human judgments within an exact criterion, then averages exact variants within a family, and finally averages the available families within each subject/plan pair. Every resulting pair has weight 1. Thus a pair with three human reviewers receives the same total weight as a pair with one reviewer, and a criterion with repeated labels cannot dominate the composite. Averaging each human's signed *direction* before averaging across reviewers preserves disagreement; taking the sign of the mean raw preference would instead turn some divided panels into unanimous wins.

Canonical model ordering normalizes reversed slots. The caller must provide a stable unordered plan-pair ID and exact criterion identities before matching human and AI labels. Abstention coverage and criterion coverage must be reported separately, because silently averaging only easy criteria could change the target. Equal family weighting is an explicit policy choice, not a discovered fact about lesson quality. Per-family and per-variant analyses are useful sensitivity checks.

## Pairwise model leaderboard

`fit_bradley_terry` consumes the aggregated pair rows. It minimizes fractional binomial cross-entropy with `P(A wins) = sigmoid(ability_A - ability_B)` and an L2 penalty `ridge / 2 * sum(ability²)`. The primary ridge value is fixed at 1.0. A tie contributes half a win to each side; this is a fractional-win Bradley–Terry model, not a model with a separately estimated tie propensity. It uses NumPy Newton updates with backtracking and reports convergence.

This estimator adjusts for which opponents each model faced. Raw mean win rates can change simply because one model faced stronger opponents. Ability is centered within each connected comparison component. If the graph has multiple components or a requested model has no comparisons, **global ranks are undefined**: the output sets them to `null`, lists components and missing-model coverage, and only supplies within-component probabilities and ranks. Ridge makes the optimization finite under complete separation; it does not identify comparisons between disconnected components.

Scores within 1e-8 are numerical ties and receive midpoint ranks. Always supply the same fixed `models` universe to human and AI fits, including models absent in a sparse sample. Otherwise dropping difficult-to-rank models would make apparent alignment improve. `compare_rankings` rejects differing model universes, disconnected graphs, and nonconverged fits.

One-dimensional abilities assume that relative quality can be usefully summarized across subjects, rubric families, and opponents. They do not eliminate preference cycles, model-by-student interactions, or a biased comparison schedule. The subject-balanced sensitivity sets each pair's weight to `N / (S * n_s)`, where `N` is the fit's pair count, `S` its subject count, and `n_s` the subject's pair count. Each subject therefore receives equal total weight `N / S`, and total fit weight remains `N`. This normalization is applied separately to the human and AI fits. The primary estimator gives equal weight to pairs. Report this distinction when increasing density, since students with many candidate plans can otherwise gain more influence.

## Alignment metrics

The primary metric is **Kendall tau-b**, which handles rank ties. Secondary metrics are Spearman correlation, the fraction of unordered model pairs with exactly matching rank directions, top-1 set overlap, mean absolute rank displacement, normalized rank displacement, and mean/RMS differences in Bradley–Terry-implied win probabilities. Top-1 overlap is the Jaccard overlap of winner sets, so tied winners are not chosen arbitrarily. Probability errors weight every model pair equally and can reveal disagreement in strength even when ordering agrees. Undefined correlations for constant ranks remain `null`.

These are model-level metrics. Per-label confusion matrices and score/explanation contradictions remain useful diagnostics, but should not replace leaderboard alignment in judge selection.

## Uncertainty and human reliability

`bootstrap_alignment` resamples **subjects**, assigning the same sampled subject multiplicity to all human and AI comparisons for that subject. Each input should already contain one aggregate observation per plan pair. Repeated criteria and repeated plans from one student are not independent bootstrap observations. Human and AI density may differ; subjects absent from either side are excluded explicitly. The model universe and ridge remain fixed across resamples.

The output contains percentile confidence intervals for alignment, per-model rank intervals, and frequencies of appearing in the top tied set. It reports the number of connected, converged resamples. Intervals computed from valid resamples condition on connectivity; when many sparse resamples are disconnected, those intervals alone understate instability. Subjects are the primary clustering unit, but this bootstrap does not fully account for dependence from the same human reviewer rating many students or from the same LLM/prompt producing many predictions. Human reliability analysis should additionally use disjoint reviewer groups and report the matched-pair coverage of each comparison.

The GRE Quant metadata contain 549 unique reviewed pairs, 246 subjects, and 39 human reviewers. There are 197 pairs with one review, 127 with two, 215 with three, and 10 with four. Human–human comparisons are therefore possible on a subset of the data. Agreement between a smaller human panel and a larger independent panel is a reliability estimate for those sampling conditions, not a hard upper bound on AI quality. Do not compare a subset of human votes against a reference that contains those same votes without labeling the resulting overlap.

## Frozen study protocol and density

The initial 100-pair run exposed 81 subjects. Those subjects belong to development. The remaining 165 subjects are split by a frozen seeded subject hash into validation (82) and test (83), with no shared plans or duplicate complete pretests across splits. Keep all comparisons and augmented pairs for one subject together. Model identities are allowed in every split because the task measures rankings of the same generation model presets on new subjects.

The frozen study screens candidates on 60 validation pairs, expands selected candidates to 120 validation pairs, and freezes the winning judge before test labels are used to assess it. Kendall tau-b drives selection, followed by win-probability MAE and measured usage for ties. The test is consumed once; any changes motivated by its results need a fresh test set. Earlier exploration on the exposed subjects remains development analysis.

Density studies should distinguish (1) more repeated votes on the same pair, (2) more distinct human-reviewed pairs, and (3) additional AI comparisons for pairs humans never reviewed. Use nested seeded subsets and repeated draws so a higher density includes the lower-density sample. Report model coverage, graph connectivity, subject coverage, and comparison count at each density. For a clean sampling-density contrast, retain the same eligible subjects and opponent schedule where feasible; separately report changes from adding previously unreviewed opponents. Compare both AI-on-human-pairs and AI-on-augmented-pairs against the same held-out human reference, and include the subject-balanced weighting sensitivity. A denser AI graph can reduce sampling noise but cannot by itself establish that judgments on unreviewed pairs are correct.

The ridge penalty stays at 1.0 at every density. It therefore has greater relative influence when fewer comparisons are available. Density changes in fitted win-probability error combine sampling effects with this change in shrinkage; they should not be interpreted purely as changes in judge calibration. The fixed estimator is part of the evaluated pipeline.

An unreviewed pair inherits the rubric context of an existing review for the same subject. The q08 wording changed in the source study, so additional pairs can also change the mix of formatting/weakness-identification and test-taking-strategy questions within that family. Dense-versus-reviewed results can reflect that rubric mix as well as more graph edges. The predeclared test sensitivity excluding q08 helps assess this limitation; it is not used to retune the selected judge.

## Related methods

The ability model follows the paired-comparison formulation of
[Bradley and Terry (1952)](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091).
Modern LLM preference leaderboards also use pairwise model-ranking and uncertainty
analysis, as described in [Chiang et al., Chatbot Arena (2024)](https://arxiv.org/abs/2403.04132).
Our fixed ridge, fractional rubric composite, subject-cluster resampling, and
matched tutor analyses are study-specific choices; these papers do not validate
this particular SSOT benchmark or its resulting ranks.
