The report-revision diagnostics use saved judgments only. Run from the repository
root after the original study and example extension analyses exist:

```sh
uv run python experiments/report-revision/audit_statistics.py
uv run python experiments/report-revision/human_comparator.py
```

Outputs stay in ignored `results/report-revision-20260911/`. The first command
records split and generator-pair counts, matched low/high reasoning contrasts
with 500 paired student-bootstrap draws, and an exact replay of the example
interval. The second compares cached selected Terra and a human reviewer group
against the same other human-group reference on identical scored criterion
support. It counterbalances reference sides within each of the 200 existing
reviewer partitions; their percentile ranges describe dependent partition
sensitivity, not independent confidence intervals.

The local `human_comparator_protocol.json` records the prescribed diagnostic.
Both scripts verify the SSOT fingerprint and saved metric provenance and assert
that protected study artifacts remain unchanged. These are post-study
diagnostics; they do not change judge selection or make model calls.
