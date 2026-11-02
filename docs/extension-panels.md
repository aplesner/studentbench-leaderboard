# Prespecified cached-call panels

The validation extension adds two offline composite judges over the nine
high-effort OpenRouter variants. Their recipes were fixed before inspecting
provider-validation outcomes in `experiments/ranking-extension/derived_judges.toml`
and copied unchanged into `results/ranking-extension-20260911/derived_judges.toml`.
The frozen file has SHA-256
`08a81948ba01951ef6c0638636ba2a8be85dfc8161bd6d5fb4f85e6793454418`.
The adjacent `derived_judges_manifest.json` records the freeze and enforces the
file fingerprint. Original judge definitions and original study results remain
unchanged.

`judge_10` / `or_panel_nine_high` assigns equal weight to all nine members.
`judge_11` / `or_panel_family_balanced_high` assigns equal total weight to the
six provider families. These are two prespecified weighting policies, not
weights optimized against human preferences.

| Provider family | Members | Equal-member weight per member | Equal-family weight per member |
|---|---:|---:|---:|
| Google | 1 | 1/9 | 1/6 |
| Anthropic | 2 | 1/9 | 1/12 |
| DeepSeek | 2 | 1/9 | 1/12 |
| Moonshot | 1 | 1/9 | 1/6 |
| Z.AI | 2 | 1/9 | 1/12 |
| Meta | 1 | 1/9 | 1/6 |

For each exact pair/criterion, each member's already-normalized preference
becomes a win share: A = 1, explicit tie = 0.5, B = 0. The panel takes the
weighted arithmetic mean. Preference strength is ignored, as in the primary
model-ranking estimator. An average of 0.5 can reflect opposite member votes;
it must not be described as unanimous agreement that plans are equal.

Every member is required. A missing cached call, missing criterion, or member
abstention makes that panel criterion missing; the remaining weights are not
renormalized. Complete-call coverage and scored-criterion coverage are reported
separately. No panel is added unless every member variant/draw was explicitly
scheduled for every pair in that analysis round. The analysis never invents
new calls or fills failed members with another model.

The optional TOML loader validates declared variants, nonnegative integer draw
indices, unique members, distinct derived IDs/names, and strictly positive
finite weights. The analysis records the original definition, its semantic
hash, the exact file hash, member configuration hash, raw and normalized weights,
and a derived judge version. Existing behavior is unchanged when the optional
file does not exist.

Both panels reuse the same completed member calls, so evaluating their two
weighting rules incurs **no additional API cost**. Each panel's reported call
count and estimated cost describe the calls needed to obtain that recipe's
verdicts; they must not be added to the underlying single-judge costs when
calculating actual study spending.

The panels belong in the `provider_validation` analysis and report alongside
their individual members. They remain exploratory validation results in an
extension of the original study. Family balancing does not guarantee independent
errors, eliminate shared prompt bias, or establish a superior leaderboard.

The Muse preflight then returned HTTP 404 because the existing OpenRouter
workspace guardrail excluded its only endpoint. Account policy was preserved.
Before inspecting any new judgment outcomes, a separate availability amendment
was frozen at `2026-09-11T06:46:14.404943+00:00` in
`derived_judges_available.toml`, with SHA-256
`560bc860c08f334ae27d320d2d2c9c1f1eb17b3d00c43ccd9c3b67f0373d05e8`.
Its own `derived_judges_available_manifest.json` records the original recipe
hash, availability record hash, and 404 evidence. The original nine-model and
six-family recipes remain unchanged and cannot score while Muse is missing.

The two additional recipes remove only that unavailable member:

- `judge_12` / `or_panel_eight_available_high`: equal weight of 1/8 for each
  of the eight available models.
- `judge_13` / `or_panel_five_families_available_high`: equal family mass of
  1/5; Google and Moonshot each have one member of weight 1/5, while each
  Anthropic, DeepSeek, and Z.AI member has weight 1/10.

Every remaining member is still required. This amendment changes the named
estimand from nine models/six families to eight models/five families; it is not
silent missing-member renormalization and does not demonstrate anything about
Muse's judgment quality. The analysis reads the original full provider schedule
so the unscorable original recipes and Muse's missing coverage remain visible.
It supports exactly these two optional recipe filenames, checks their separate
freeze hashes, and rejects name or judge-ID collisions across the files.
