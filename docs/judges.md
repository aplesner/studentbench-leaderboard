# Configurable two-plan judges

Use one public interface to compare two lesson-plan JSON objects:

```python
from studentbench_leaderboard import ComparisonContext, compare_plans

result = compare_plans(
    plan_a,
    plan_b,
    "judge_1",
    context=ComparisonContext(
        instrument="gre_quant",
        subject=review_subject_id,
        pretest=pretest_responses,
        rubric=rubric,
        plan_ids=(plan_a_id, plan_b_id),
    ),
)
```

The context supplies the shared student's pretest and the criteria being
evaluated. Each rubric entry has `id`, `question`, `scale_id`, and
`rating_labels`, as returned by `Dataset.rubric(grading_id)`. For SSOT data,
use the fields of `make_task(dataset, grading_id)` and pass
`dataset_sha256=dataset.fingerprint` in the context so the result can be matched
by the existing human-agreement report. For external plans, the function
fingerprints its supplied inputs. Callers supply two plans written for the
same pretest; the CLI validates this relationship against SSOT plan metadata.

The function resolves the configuration, makes the model calls, and returns
one result dictionary. It does not write files. The CLI uses the same
configured-judge implementation and journals complete results automatically.

## Available judges

| Public ID | Strategy | Calls per comparison | Behavior |
| --- | --- | ---: | --- |
| `judge_1` | `single` | 1 | One model compares A and B |
| `judge_2` | `order_check` | 2 | The same model compares AB and BA; the result records changes after correcting the orientation |
| `judge_3` | `repeated` | 5 | The same model votes five times, with reproducible random AB/BA presentation |
| `judge_4` | `panel` | 3 | One AB vote each from the configured example GPT, Claude, and Gemini models |
| `judge_5` | `single` | 1 | Evidence and explicit winner, Luna high effort |
| `judge_6` | `single` | 1 | Separate absolute ratings, Luna high effort |
| `judge_7` | `balanced_soft` | 2 | Luna high effort in AB and BA; mean win share for model ranking |
| `judge_8` | `panel_soft` | 4 | Luna, Terra, Sol, and Astra at low effort; mean win share for model ranking |

The `single` strategy means a single **two-plan comparison call**. The older
CLI `--mode single --model ...` still provides isolated-plan quality ratings.

```bash
# List IDs, resolved versions, models, and expected call counts (no SSOT needed).
uv run studentbench-leaderboard judges

# Inspect both orders before running the comparison.
uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 --judge judge_2 \
  --dry-run --output results/judge-2-preview.json

# Run the simplest configured judge; results default to results/judgments.jsonl.
uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 --judge judge_1

# Report each judge version separately against the human references.
uv run studentbench-leaderboard report results/judgments.jsonl \
  --output results/judge-report.json
```

Preview output includes every actual prompt, model, reasoning setting, and
presentation order, with the total number of planned model calls. The live
command also prints the resolved judge version and pending call count before
calling a provider. `--limit N` selects N comparison tasks, so `judge_3` can
make up to `5 * N` model calls.

## Configuration and subversions

Definitions live in
[`src/studentbench_leaderboard/judge_configs/`](../src/studentbench_leaderboard/judge_configs/).
TOML uses Python's standard-library parser. Each file has the same name as its
public judge ID. For example, `judge_1.toml` contains:

```toml
schema_version = 1
judge_id = "judge_1"
strategy = "single"

[prompts]
system = "prompts/pairwise_system.txt"
user = "prompts/pairwise_user.txt"

[[members]]
name = "member_1"
model = "openai/gpt-5.6-luna"
reasoning_effort = "low"
```

Paths are relative to the configuration directory. The user template contains
exactly one `$payload` placeholder, filled with both plans, the shared pretest,
and rubric definitions. Plan content is inserted as data; it is never
interpreted as another template. Configuration rejects unknown fields and
invalid strategies or member counts before making model calls.

The public name remains `judge_1`. Internally the implementation derives:

```text
judge_1_<prompt-content-hash>_<model-settings-hash>
```

Each suffix is 12 hexadecimal characters. Changes to the system prompt or
user template change the first suffix. Changes to a member model or its
reasoning effort change the second. File locations, descriptions, and TOML
comments do not change the effective version. Versions are content identifiers,
not sequential release numbers. Full configuration fingerprints and the exact
resolved configuration are also stored in results.

Keep member slot names stable when changing a slot's model. Changing the
strategy, number of repetitions, seed, member slots, aggregation rule, or
strategy implementation requires a **new public judge ID**. Copy a TOML to a
new filename and change its `judge_id`. The CLI rejects a changed strategy
under an existing ID when that ID is already present in its results journal;
reports also reject conflicting strategy definitions. With an empty journal
or direct API calls, preserving IDs across history is the configuration
author's responsibility.

Prompt files are shared by the supplied examples. Editing a shared file
updates every judge that references it. To customize just one judge, copy the
prompt file and change that judge's path. Saved results retain their old
resolved configurations and prompts.

Use `--judges-dir PATH` on `judge` or `judges`, or `judges_dir=Path(...)` in the
Python API, to load a separate directory of definitions. Copy the supplied
definitions and prompts there as a starting point. Credentials stay in the
provider environment, outside TOML. The example panel is editable; additional
LiteLLM provider/model identifiers can be used for GLM, Kimi, DeepSeek, or
other models. Adding/removing panel slots requires a new ID; replacing a
slot's model creates a subversion.

`--judge` and the older ad hoc `--model` option are mutually exclusive. For
configured judges, set reasoning effort in TOML so it is captured in the
resolved version. The legacy path remains available for one-off experiments.

## Voting and order diagnostics

Every member returns the existing per-criterion −2..+2 preference and a
concise explanation. A swapped-order response is negated before aggregation,
so all votes and final scores refer to the caller's original A and B.

Voting is separate for every rubric criterion:

- Map positive scores to A, zero to tie, negative scores to B, and `null` to
  abstention.
- Require more than half of **all scheduled calls** to choose A, B, or tie.
  Abstentions remain in the denominator. A 2–2–1 split in five calls has no
  majority, and two scored A votes with three abstentions are also inconclusive.
- Return `null` when there is no majority. Return zero only for a majority of
  explicit tie votes. The raw votes distinguish inconclusive aggregation from
  individual model abstentions.
- For an A/B majority, preserve the conservative median preference strength
  among voters on that side. With winning strengths 1 and 2, use 1; with
  strengths 1, 2, and 2, use 2. Restore the winning direction's sign.

With `judge_2`, both orientations must agree on the winner/tie category.
`diagnostics` includes every normalized score, vote counts, and separate
`score_changed` and `direction_changed` flags. Different strengths can thus be
flagged even when the favored plan stays the same. When fewer than two calls
produce scores, change flags are `null` because consistency is unassessed.

Repeated judging derives presentation order from the configured seed and
input identities; previews and execution use the same schedule. The seed
controls ordering, not the provider's generation randomness. Independent
random choices can present the same order several times. Use `judge_2` when
both orientations must be represented. Panel judging presents AB to every
member and gives each member one equal vote.

## Results and failure behavior

Configured results use schema version 2 and retain `judge_id`, resolved
`judge_version`, full configuration, input fingerprints, and request identity.
`calls[]` holds each member's actual messages, model/settings, plan order,
raw response, parsed and orientation-corrected scores, usage, timestamps,
and latency. Top-level `ratings` holds the aggregate. Its `raw_response` is
the serialized aggregate; individual provider responses are inside `calls[]`.

The existing report reads legacy version-1 and configured version-2 records.
It groups configured results by resolved version, so editing a prompt or model
cannot silently merge different judges' outputs. Each aggregate is one judge
observation; internal panel/repetition votes are not independent observations
in the human-agreement report.

Completed requests resume from the JSONL journal. A prompt/model/effort change
gets a different request ID and runs again. Provider or parse failures abort
the comparison; a failed comparison has no aggregate record and is retried
from its first call on the next run. Completed earlier comparisons remain
saved. There are no automatic provider fallbacks or partial-panel voting.

## Evidence-based experimental judges

`judge_5` asks for evidence from each plan, a criterion-specific explanation, an
explicit winner (`A`, `B`, `tie`, `insufficient`), and preference strength. The
adapter computes the signed score from that enum. `judge_6` instead asks for an
absolute 1–5 rating of each plan and derives the preference from their difference.
Both preserve the complete raw response and explanations in each call record.
The bundled examples use Luna with high reasoning effort; the ranking-study
TOML directories cover the other requested models and settings. These examples
are not implicitly selected as the study winner.

Optional `response_style` is `score_reason` (default), `evidence_winner`, or
`absolute_ratings`. Response-style changes alter the strategy contract and need
a different judge ID. Optional `plan_projection = "human_display"` removes
archival `answer_original` fields while preserving the displayed answer and
solution; this input change produces a prompt subversion. The SSOT is unchanged.
For verified richer pretest inputs, construct `ComparisonContext.pretest` using
`PretestContextBank.enrich_pretest`; the study runner configures its pinned
supplement through `context_input.toml` beside the judge TOML.

The study additionally records its frozen soft balanced-order and panel
compositions in `experiments/ranking-study/derived_judges.toml`. Those study
variants compose cached calls in `study_analysis`. The public API and CLI now
also support the `balanced_soft` and `panel_soft` strategy names through
`judge_7` and `judge_8`.

Explanations remain verbatim in the presentation order of their individual call.
Only numeric scores are normalized to the original input order. When auditing
an explanation's A/B references, use that call's `order`, `plan_ids` or
`presented_plan_ids`, and raw `ratings`; do not compare its words directly with
an order-normalized score without mapping the plan slots.

## Soft outputs for model ranking

`balanced_soft` makes exactly one AB call and one BA call with one configured
model. `panel_soft` makes one AB call per configured model, with equal member
weights. Both first map each normalized response to A-win share: A = 1, tie =
0.5, B = 0. Their mean is returned in the separate top-level
`preference_probabilities` map. This is an empirical vote share, not a
calibrated confidence estimate. If any member abstains for a criterion, its
probability is `null`.

The existing `ratings[criterion].score` remains the strict-majority categorical
result. Opposed AB/BA votes therefore produce probability `0.5` and categorical
score `null`: they contribute evenly split evidence to a model leaderboard
without claiming that the models explicitly judged a tie. A panel can also
have a categorical majority while its probability is `null` because one member
abstained. `diagnostics` preserves the normalized scores, win shares, majority,
and reason for a missing probability. Legacy judges and their version hashes
retain their existing behavior.

Use the same comparison interface and the ranking adapter:

```python
from studentbench_leaderboard import ComparisonContext, compare_plans
from studentbench_leaderboard.judge import make_task
from studentbench_leaderboard.ranking import (
    judgment_rows, aggregate_pairs, fit_bradley_terry,
)

task = make_task(dataset, grading_id)
context = ComparisonContext(
    instrument=task.instrument,
    subject=task.subject,
    pretest=task.pretest,
    rubric=task.rubric,
    plan_ids=tuple(task.plan_ids),
    dataset_sha256=dataset.fingerprint,
)
result = compare_plans(*task.plans, "judge_7", context=context)
rows = judgment_rows(dataset, [result])  # Supply one judge version at a time.
pairs = aggregate_pairs(rows)
models = sorted({plan["ai_model_preset_id"] for plan in dataset.plans.values()})
leaderboard = fit_bradley_terry(pairs, models=models)
```

Collect comparisons across the model universe before expecting global ranks;
the one-comparison example usually leaves its graph disconnected.
`judgment_rows` validates the SSOT fingerprint, subject, actual plan IDs,
optional model identities, and criterion metadata. It uses the probability
map when present, including its explicit missing values, and otherwise maps
legacy scores to win shares. It rejects mixed judge versions and duplicate
requests. Missing observations remain visible for coverage accounting and
are excluded by `aggregate_pairs`.

The existing `report` command still measures categorical-score agreement.
Use the ranking adapter above for model leaderboards that retain fractional
preferences; its use does not change the meaning of the older label metrics.

The new bundled judges use `human_display` plan projection and explicit-winner
responses. Richer pretest context remains explicit: when using a verified
`PretestContextBank`, pass `bank.enrich_pretest(task.pretest)` as the context's
`pretest`. The public API does not load external question banks automatically.
To reproduce a study result, retain its exact frozen prompts, models, settings,
and context supplement; the bundled examples are independently versioned
configurations, not an implicit assertion that a particular study winner was
deployed.
