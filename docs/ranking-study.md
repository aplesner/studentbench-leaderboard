# Model leaderboard validation study

This study evaluates whether human verdicts and LLM verdicts produce similar
**authoring-model rankings**, rather than optimizing agreement on every lesson-plan
comparison. The current experiment uses GRE Quant and the 12 authoring model
presets in its reviewed-plan catalog. Generation reasoning settings remain part
of a preset's identity. Verbal is not pooled with Quant.

The local frozen experiment is `results/ranking-study-20260910/`; the generated
standalone report is `report.html`. All raw data, responses, human labels, and
outputs remain local and ignored by Git. Source configuration lives in
`experiments/ranking-study/`.

## Data separation

The previous 100-pair run exposed 81 pretest subjects, all assigned to development.
The remaining 165 subjects are deterministically divided into 82 validation and
83 test subjects. Structural checks found no shared lesson-plan IDs, repeated
nonblank student IDs, or identical full pretests across these subjects. Tutors
can occur across subject splits; this assesses generalization to new students,
not to unseen tutors. Human-human analyses separately split entire tutor IDs.

| Split | Subjects | Human-reviewed pairs | All available plan pairs |
| --- | ---: | ---: | ---: |
| Development | 81 | 260 | 755 |
| Validation | 82 | 134 | 244 |
| Test | 83 | 155 | 293 |

The catalog includes only two distinct authoring presets for the same student.
The frozen manifest records every pair, seed, source hashes, and selected tasks.
Test evaluation requires a selection lock. Once test results have been inspected,
further tuning needs another untouched holdout or should be labeled exploratory.

## Estimand and uncertainty

Map A/tie/B votes to 1/.5/0, ignoring the original preference magnitude. Average
reviews within an exact rubric question and scale, then average variants within
a criterion family, then give q01–q08 equal weight within each pair. Each pair
has one unit of weight in the primary fractional-win Bradley–Terry fit; a fixed
L2 penalty of 1 keeps estimates finite. Equal criterion weighting is an explicit
policy choice, not an empirically established measure of educational utility.
Per-criterion rankings and alternative weightings are sensitivity checks.

Kendall tau-b across the 12 models is the primary selection metric. Spearman,
model-pair order agreement, top-model overlap, rank error, and fitted
win-probability discrepancy supplement it. Missing or disconnected model graphs
receive no global ranking. Confidence intervals resample entire student/pretest
subjects, preserving dependent criteria, repeated plans, and reviewers within a
subject. Tutors spanning subjects remain a dependence limitation.

AI fit and human fit use the same model universe. Dense AI rankings may include
unreviewed edges, but the human reference remains fixed; there are no fabricated
human labels for those edges. Density experiments report graph connectivity,
subject coverage, and both pair-weighted and subject-balanced sensitivities.

See [ranking methods](ranking-methods.md), [human validation](human-validation.md),
and [explanation audit](explanation-audit.md) for exact assumptions and limitations.

## Judge configurations

`judge_5` returns evidence for both plans, a criterion-specific explanation, an
explicit A/B/tie/insufficient winner, and strength. Code converts this response
to the signed score, avoiding LLM sign encoding. It does not show irrelevant
1–5 absolute quality labels in the pairwise prompt. Model and reasoning settings
are subversions configured in separate TOML directories.

The first screen uses Luna, Terra, Sol, and Astra at low/high effort, the original
Luna-low signed-score judge, and `judge_6`, which rates both plans against the
absolute 1–5 rubric before deriving a preference. The study can also evaluate
fixed panels, strict votes, and balanced AB/BA averages from stored member calls.
A soft average of disagreeing votes is a win share, not a claim that the plans
are equivalent. Model panels share the same provider family in this experiment;
the older development panel includes Claude and Gemini.

A separate `plan_projection = "human_display"` prompt subversion removes only
archival `answer_original` fields. It keeps the displayed answer and explanation
unchanged; the SSOT is never repaired in place. The original application altered
some keys before displaying them to humans, so rankings evaluate delivered plans,
including upstream postprocessing. Excluding answer accuracy is a sensitivity,
not grounds to rewrite human labels or restore original keys silently.

## Reproduction

From the project directory with the local SSOT and environment installed:

```sh
uv run python experiments/ranking-study/prepare.py
uv run python -m studentbench_leaderboard.study run results/ranking-study-20260910/round1_jobs.json
uv run python -m studentbench_leaderboard.human_analysis \
  --manifest results/ranking-study-20260910/manifest.json \
  --split validation --output results/ranking-study-20260910/human
```

The `study run` command makes paid external model calls; it caches each
variant/pair/draw, verifies frozen config and dataset identity, and reuses
completed calls after reparsing saved raw responses. Model failures and malformed
responses are explicit errors. Analysis and report rendering make no API calls.
All calls record prompts, JSON schemas, settings, response IDs, usage, source
fingerprints, elapsed time, and estimated cost. Costs use published token rates,
not a reconciled provider invoice.

The requested identifiers and low/high reasoning settings were checked against
the official model pages: [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra),
[Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol), and
[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra). The pinned
LiteLLM capability table predates Astra, so its documented reasoning parameter
is explicitly allowed instead of silently discarded.

For the richer-context second round, `prepare_enhanced.py` verifies and freezes a
local application-bank supplement, then creates the 240 fixed prototype jobs:

```sh
uv run python experiments/ranking-study/prepare_enhanced.py
uv run python -m studentbench_leaderboard.study run results/ranking-study-20260910/round2_enhanced_jobs.json
uv run python -m studentbench_leaderboard.study run results/ranking-study-20260910/round2_balanced_jobs.json
uv run python experiments/ranking-study/finish.py
```

`finish.py` waits for those explicitly scheduled calls, screens the ten original
single-call variants, advances the best single judge to 120 validation pairs,
and compares it with the two input-improved prototypes and the fixed balanced
Luna prototype. Eligibility requires at least 90% scored criteria and complete
execution. Selection maximizes Kendall tau-b, then minimizes probability error
and observed token cost. It writes `selection.json` and locks the manifest before
executing held-out jobs. The selected strategy is evaluated on 155 reviewed test
pairs with a legacy Luna-low control, and on all 293 available test pairs.
The report is regenerated after each completed analysis phase.

The supplemental question bank restores quantities and shared chart text only
after exact shared-field matching. It is not part of the immutable SSOT release,
and remaining source figures are explicitly unavailable in these text requests.
The round-two input changes are a bundled intervention; they do not separately
identify the effects of each input repair. See [pretest context](pretest-context.md).

The holdout is enforced by the analysis and selection code, not by separate
filesystem permissions: `Dataset` loads/validates the complete local CSVs, while
all tuning and validation statistics filter to development/validation subjects.
Test verdicts do not contribute to tuning or selection. The final test is then
opened under the recorded selection lock.

`finish.py` expects the three initial batch commands to be running or complete.
If a batch fails after its bounded retries, rerun that batch command to fill
only its missing cache entries; the controller waits for completion. Use one
controller per study directory. No failed or inconclusive judgment is filled
with a fabricated label.

## Using the selected configuration

The locked selection is the single-call `enhanced_terra_high` variant of
`judge_5`. Its frozen TOML, prompts, and verified input supplement are in
`results/ranking-study-20260910/configs/enhanced_terra_high/`. The bundled
`judge_5` example has different model settings; use the frozen directory when
you want this selected configuration.

The shared API accepts that configuration and enriched context:

```python
from pathlib import Path
import tomllib

from studentbench_leaderboard import ComparisonContext, compare_plans
from studentbench_leaderboard.data import Dataset
from studentbench_leaderboard.judge import make_task
from studentbench_leaderboard.pretest_context import PretestContextBank

config = Path("results/ranking-study-20260910/configs/enhanced_terra_high")
inputs = tomllib.loads((config / "context_input.toml").read_text())
dataset = Dataset(Path("studentbench_data_ssot"), "gre_quant")
bank = PretestContextBank.load(
    dataset.root,
    source_bank=config / inputs["source_bank"],
    expected_source_sha256=inputs["source_sha256"],
)
task = bank.enrich_task(make_task(dataset, "ai_vs_ai_quant_grading_0001"))
result = compare_plans(
    *task.plans, "judge_5", judges_dir=config,
    context=ComparisonContext(
        instrument=task.instrument, subject=task.subject,
        pretest=task.pretest, rubric=task.rubric,
        plan_ids=tuple(task.plan_ids), dataset_sha256=dataset.fingerprint,
    ),
)
```

This makes one paid call and returns its explanations and provenance. It uses
the supplied plan order. The experiment runner additionally applies its frozen
seeded presentation order and journals/reuses calls; use its existing job files
to reproduce the study exactly. Further judge tuning must use a new untouched
holdout now that this study's test results are available.
