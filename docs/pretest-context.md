# Optional pretest context supplement

The frozen Quant question-bank CSV contains question stems, answer choices,
answer keys, and skill metadata. It omits separate quantitative-comparison
quantities, shared chart data, and figure assets. The archived supplemental
judge-task inputs retain the same omissions. The leaderboard adapter did not
create them.

All 54 unique assessment questions occur in the previously exposed 81-subject
development sample. In that sample, 13 of 16 quantitative-comparison items lack
both quantities in the baseline payload; three stemless questions already
include their quantities as an exported fallback. Six chart-dependent questions
lack the shared data. Two additional geometry questions refer to diagrams. The
baseline omissions affect 536 quantitative-comparison response rows and 243
chart response rows among 2,187 development pretest rows.

The neighboring original application repository contains
`data/questions_gre_quant.json`. It provides the exact quantities, shared text,
and SVG assets. Its 54 question IDs, options, correct answers, formats, skills,
subskills, and difficulties match the immutable CSV. Stems also match exactly
after applying the documented quantity fallback for the three stemless items.
This is evidence for a compatible original-application supplement; the omitted
fields cannot be verified against the CSV itself, and a historical deployment
replay was not performed.

`PretestContextBank.load(ssot_root, source_bank=..., expected_source_sha256=...)`
first verifies the SSOT bank against its release manifest and the optional
application bank against an explicitly pinned SHA-256. It then verifies the
exact item universe and all shared fields. `enrich_pretest` validates each
response's question context before adding any fields. It retains the recorded
student answer and correctness, strips fields outside the judge input allowlist,
and never changes source files or lesson plans.

The text supplement restores all 13 missing sets of quantities and the shared
data for all six chart questions. It does not render the eight SVG figures or
replace them with invented descriptions. Every affected item explicitly states
that the figure is unavailable; the six chart questions receive their original
shared text, while the two geometry questions retain their original prose.
Judges must not infer unstated geometric information. A future visual-input
variant can render the frozen SVG assets as a separately versioned experiment.

The pinned local snapshot is
`results/ranking-study-20260910/context/quant_application_bank.json`, with SHA-256
`70bc1f1d4d85b93d6f6b23f035a49ac912d4aa5f29787a9510f68072d0a7f336`.
Its SSOT comparison bank has SHA-256
`641bc178334710625fabee525ea3754ccdeba0b24083ee75a4d5453dea7b4db6`.
Coverage and provenance are recorded in `context/coverage.json`. Raw assessment
content remains in ignored local results.

This adapter is opt-in. Existing `make_task` behavior and completed round-one
inputs remain unchanged. Enriched study variants carry their source path/hash
and context provenance, while request identities include the enriched prompt.
They must be reported as new input variants, not silently substituted into old
judge runs. No human preference labels guide the context reconstruction.
