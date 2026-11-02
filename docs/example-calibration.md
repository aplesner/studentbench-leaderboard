# One-example human calibration ablation

The original ranking-study rounds used **no human-reviewed prompt examples**.
The extension introduces one fixed human-reviewed development comparison as a
new, versioned user-context block. It compares that variant against the cached
zero-example enhanced Terra-high judge on the same 120 validation pairs.
This is an ablation of adding an example, not a retrospective description of
the original judge.

`experiments/ranking-extension/prepare_example.py` selects the example using
development data alone. It does not inspect new validation model outcomes or
validation/test human annotations. The policy requires at least two different
reviewers, identical A/B presentation across reviews, the same eight exact
rubric variants/scales, and at least one A-favored and one B-favored criterion.
The presentation restriction keeps human explanations verbatim without an
unreliable natural-language A/B rewrite. Among eligible pairs, the script sorts
by reviewer count, mean per-criterion majority share, unanimous-criterion count,
then a fixed content-hash tie-break.

Of 260 development pairs, 14 satisfy the complete policy. The chosen comparison
has two reviewers and seven of eight direction agreements. They disagree on
practice alignment, disagree about some preference strengths and quality
ratings, and sometimes give different rationales. All of those observations
remain visible. This is a deliberately chosen illustration of tradeoffs,
not a representative sample or error-free gold answer. Human reasoning may
reward proxies or apply a different criterion; the prompt explicitly tells
the judge to assess the evidence independently.

The payload contains exactly one full comparison:

- `example_input`: both lesson plans, 27 pretest response items, and the eight
  rubric questions using the same human-display projection and verified context
  supplement as the enhanced judge.
- `human_reviews`: anonymous `Reviewer 1` and `Reviewer 2`, with every criterion's
  original separate A/B quality scores and labels, direct signed preference
  and label, and full human explanation.
- `human_quality_scales` and `human_preference_scale`: original SSOT labels.
  The sign convention explicitly says positive favors example A and negative
  favors example B; null is missing and is distinct from a tie. Separate quality
  scores are not silently turned into a pairwise preference.
- Instructions identifying this as a completed illustrative example rather
  than the current comparison, and explaining that no ideal assistant output
  or synthesized evidence has been supplied.

Reviewer, subject, lesson-plan, and author identifiers stay in a separate ignored
provenance file. Source prose would receive only deterministic identifier
redaction if needed; no such redactions were necessary for the chosen example.
The artifact records every human review separately and performs no majority
collapse. The original displayed answer remains unchanged; hidden
`answer_original` fields are removed by the established display projection.
The pretest supplement retains its original provenance and missing-figure notes.

Local artifacts are `results/ranking-extension-20260911/example.json` and
`example_provenance.json`. The example is 61,615 bytes with SHA-256
`070f62e1d16fa3e84e06ecd35664e328499d1eb2378c6a759cafe9c6489e8c9e`.
The provenance records the selection rule, candidate exclusions, source IDs,
source and script hashes, and exact input configuration. Re-running preparation
reproduces the files; it refuses to overwrite them with different content.

To embed the example in a `string.Template` user prompt, escape literal dollar
signs in the complete example JSON as `$$`, then append the one current
`$payload` placeholder. Keep the resulting template under a new judge version
and retain both its hash and the example hash. Dollar signs inside question
text and math must survive expansion unchanged.

The example contains no validation/test student or lesson-plan content. Adding
it nevertheless remains adaptive development on the existing validation set.
The original test was already opened after the first study's selection lock;
it should not be relabeled as a fresh confirmatory test for a judge chosen using
this extension. A new held-out cohort is needed for a fresh confirmatory claim.

Run preparation locally with:

```sh
uv run python experiments/ranking-extension/prepare_example.py
```
