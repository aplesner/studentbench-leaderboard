# Human references and model leaderboard reliability

The reference is the direct pairwise human preference, not the difference of two
individual plan ratings. Its sign becomes a fractional win: A wins = 1, tie =
0.5, B wins = 0. Authoring **model presets**, including the author's configured
reasoning effort, are the ranked entities. A unique subject and unordered plan
pair is one match. Reviewers are averaged within exact rubric variant and
rating scale, then variants within rubric family, then the eight families
equally within the pair. Repeated reviewers and the eight rubric votes do not
become independent matches.

The equal-family composite is an explicit quality policy. The changed wording
of Quant question 8 and the two question 7 rating scales remain distinguishable
in exact-unit human matching and in separate criterion leaderboards. Combining
these into a composite does not establish that their interpretations are equal.

`human_analysis.py` makes no API calls. It accepts an explicit subject set. Its
CLI reads the frozen manifest and defaults to validation; it refuses `--split
test` until `selection_locked` is true. Development contains all subjects exposed
in the earlier 100-pair judge run. Validation and test are subject-disjoint.
The structural audit found no Quant plan shared across subjects, no repeated
nonblank linked SSOT student ID, and no duplicate complete pretest fingerprint.
Anonymous human tutor identities are intentionally allowed across these subject
splits: the test concerns unseen student contexts, not unseen human tutors.

## Human agreement measures

The direct label-agreement statistic considers all distinct reviewers of the
same pair, exact rubric variant, and scale. It averages agreement **per unit**,
then across units so heavily reviewed units receive no extra weight. Its
three-way confusion matrix is symmetric; neither tutor is privileged as truth.
Its five-label agreement additionally preserves preference strength. Correlated
criteria and repeat tutors mean the individual comparisons are not independent.

For model-level agreement, each of 200 repetitions randomly divides anonymous
reviewer identities into two globally disjoint sets. Both sets are restricted
to the same exact pair/criterion units. Each half then produces a regularized
Bradley–Terry model ranking. Reported metrics include Kendall tau-b, Spearman,
model-pair order agreement, winner overlap, rank error, and the difference
between model-pair win probabilities. The 2.5th–97.5th percentiles across random
partitions describe **partition sensitivity**, not a confidence interval or a
hard upper bound for judge performance. Half-panels contain fewer reviewers than
the full reference; different half-panels can also have different task coverage.

A separately fixed reviewer partition receives a paired **subject cluster
bootstrap** with 500 resamples. This retains all correlated plan comparisons
and criteria from each sampled student context. The bootstrap provides model
rank intervals, top-rank frequencies, and alignment intervals. It does not
capture all dependence from the same human tutor reviewing several contexts.
Disconnected graph resamples contribute no global rank statistics; valid counts
are reported. Low-density graph disconnection is itself a meaningful failure
to establish a full model leaderboard.

The complete human reference also receives its own 500 subject-cluster
resamples, providing model-rank and ability-score intervals and the frequency
with which each model is ranked first. These intervals describe a reference
based on every available human review in the selected subject split, rather
than a smaller matched half-panel.

## Density experiments

For each of 200 fixed random seeds, nested samples retain 10%, 25%, 50%, 75%, and
100% of subjects or unique pairs. Another experiment caps the number of human
reviewers per pair at 1, 2, 3, and 4 while retaining all criteria from a selected
review. Counts are actual observed coverage: setting a cap does not create
reviews that do not exist. Within each replicate, a larger sample contains the
smaller sample. The independent reviewer experiment in this dataset may saturate
well below four reviewers per pair.

Two reference conditions answer different questions:

- **Self-overlap full reference:** a sample is compared with all human votes,
  including the sample. This measures internal stability and is optimistic for
  independent alignment. Full density agrees with itself by construction.
- **Independent reviewer reference:** a fixed global tutor split provides a
  sampling pool and a disjoint reference. Exact units are matched first. The
  reference then stays fixed while the pool is sampled. The curves condition on
  that one tutor partition; they avoid direct vote overlap but still vary the
  observed comparison schedule when subjects or pairs are removed.

These human experiments characterize the information available to build a
reference leaderboard. AI densification additionally scores currently unreviewed
plan pairs. Such pairs cannot support a direct label-accuracy claim; their value
is measured by whether the resulting model ranking better matches the held-out
human ranking on the same subject population.

Reproduce the validation analysis with:

```sh
uv run python -m studentbench_leaderboard.human_analysis \
  --manifest results/ranking-study-20260910/manifest.json \
  --split validation --output results/ranking-study-20260910/human
```

The JSON records the dataset fingerprint, subjects, random seeds, reviewer
partition, model universe, all sampling replicates, uncertainty, and coverage.
It remains local under the ignored `results/` directory.
