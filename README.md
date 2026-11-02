# StudentBench lesson-plan leaderboard

Evaluate lesson plans with an LLM judge **J**, then check its judgments against
the human expert ratings in the frozen StudentBench SSOT.

The initial implementation is a Python CLI and a small library. It supports
single-plan quality ratings (1–5), pairwise preferences (−2 to +2), descriptive
plan/model leaderboards, and agreement reports on matching human reviews. A
judge can abstain with `null`; a pairwise score of zero means a tie. The ranking
study evaluates model-level alignment on separate validation and test subjects.
No model calls are needed for setup or the tests.

## Setup

```bash
uv sync --locked
uv run studentbench-leaderboard inspect --verify
uv run studentbench-leaderboard --instrument gre_verbal inspect --verify
uv run python -m unittest discover -s tests -v
```

Python 3.12+ is required. The existing `uv.lock` pins the environment. Tests use
synthetic fixtures and need neither the SSOT nor credentials.

## Local data

The data currently lives in `studentbench_data_ssot/`, next to the code. Its
compressed archive is `studentbench_ssot.tar.gz` (161,299,372 bytes, about 161 MB).
Both are ignored by git, along with `data/`, `results/`, databases, and `.env`
files. Keep generated prompts and judgments in `results/`, as they can contain
lesson-plan text and assessment items.

For another local location, set `STUDENTBENCH_SSOT` or pass
`--ssot /path/to/studentbench_data_ssot` **before the subcommand**. This code only
reads the SSOT and refuses to write outputs inside it.

The first data adapter reads the primary human AI-vs-AI review study:

| Instrument | Review sessions | Rubric grades | Plans | Pretest subjects |
| --- | ---: | ---: | ---: | ---: |
| GRE Quant | 1,136 | 9,088 | 795 | 246 |
| GRE Verbal | 892 | 8,028 | 438 | 135 |

The adapter validates joins and score labels, fingerprints the six files it
uses, and can verify them against `release_manifest.json`. `--verify` checks
these six files per instrument, rather than all 37,928 files in the release.
See [the data notes](docs/data.md) for the snapshot and join contract.

## Inspect and score plans

For configurable two-plan judging, start with a named judge:

```bash
uv run studentbench-leaderboard judges
uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 --judge judge_5 --dry-run
uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 --judge judge_5
```

[`judge_5.toml`](src/studentbench_leaderboard/judge_configs/judge_5.toml) defines
an explicit-winner judge with evidence and explanations. `judge_6` derives a
preference from per-plan ratings; `judge_7` averages AB/BA votes; `judge_8` uses
a model panel. The original signed-score judges remain available as
`judge_1`–`judge_4`. Prompt/model/reasoning changes automatically get new
subversions under the same public ID. Other strategy changes require a new ID.
The bundled examples are separate from the frozen study selection.

The Python API is `compare_plans(plan_a, plan_b, "judge_5", context=...)`.
See [configuring judges](docs/judges.md) for TOML configuration, shared pretest
and rubric context, exact voting rules, and result provenance. Definitions and
prompt assets are included in the Python package; `--judges-dir PATH` selects
a separate configuration directory.

List review IDs and their plan IDs:

```bash
uv run studentbench-leaderboard reviews --limit 5
```

A review ID supplies the student's pretest, exact rubric wording, and scale
versions. By default the judge compares that review's A and B plans. Preview
the complete request locally:

```bash
uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 \
  --dry-run --output results/pair-preview.json

uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 --mode single --slot a \
  --dry-run --output results/single-preview.json
```

You can instead pass `--plans PLAN_ID` or `--plans PLAN_A_ID PLAN_B_ID` with
`--grading-id`. The number of IDs determines the mode. Every selected plan must
belong to the review's pretest subject. This initial CLI accepts plans from the
SSOT review catalog; the library's `Task` type also accepts supplied plan JSON,
pretest context, and a rubric.

For ad hoc judging, select a LiteLLM `provider/model` identifier and configure
that provider's credentials in the environment. For named judges, models and
reasoning effort come from TOML. There is no implicit default judge, fallback
provider, or automatic retry. The provider must support the
structured response format. See the official
[LiteLLM structured-output documentation](https://docs.litellm.ai/docs/completion/json_mode).

```bash
export JUDGE_MODEL='provider/model-id'  # replace with your chosen model
uv run studentbench-leaderboard judge \
  --grading-id ai_vs_ai_quant_grading_0001 \
  --model "$JUDGE_MODEL"
```

Judgments default to `results/judgments.jsonl`. Use `--output PATH` to choose
another file. Re-running a completed request reuses its saved result. With
`--dry-run`, omitting `--output` prints the prompt preview to stdout.

With `--model`, `--reasoning-effort` passes an explicit setting to the provider. `--swap`
reverses A/B and records that orientation; use both orientations to investigate
position bias. Malformed responses fail validation and are never scored as
zero. A failed call does not append a judgment; earlier completed calls remain
saved. Use one writer per output file.

For a bounded batch, replace `--grading-id` with `--limit N`. This selects the
first N distinct requests in stable review-ID order, so it is a smoke-test
sample, not a representative benchmark sample. Single mode selects slot A by
default; run `--slot b` as well to cover both sides. Running the same command
against the same results file reuses completed requests. Hashes include the
data, prompt, plan order, rubric, and judge configuration. Prompt previews are
JSON documents; actual results are JSONL. Use separate paths for each.

## Leaderboards and human agreement

The model-ranking study now compares human and AI **authoring-model
leaderboards**, with subject-separated development/validation/test data,
Bradley–Terry ratings, uncertainty, human–human agreement, and sampling-density
experiments. It evaluates Luna, Terra, Sol, and Astra at low/high reasoning
effort, evidence-based winners, per-plan ratings, balanced presentation, and
panels. The reproducible protocol and commands are in
[the ranking-study guide](docs/ranking-study.md).

The [provider and example extension](docs/ranking-extension.md) adds the nine
requested OpenRouter model configurations, a controlled Terra comparison with
one human-graded development example, and panels with equal model or provider
family weights. It reuses the original validation cohort after test exposure;
its results are exploratory. The report includes a judge catalog with exact
prompts, settings, aggregation rules and example provenance.

The local HTML artifact is `results/ranking-study-20260910/report.html`; it
contains interactive model rankings, per-criterion comparisons, human and AI
density curves, and the explanation audit. Raw calls and generated outputs stay
under ignored `results/`. The older commands below retain their descriptive
win-share semantics; the study uses the separate `ranking` and `study_analysis`
modules for opponent-adjusted model rankings.

Build a reference leaderboard without making any model calls:

```bash
uv run studentbench-leaderboard human-leaderboard \
  --mode single --output results/human-single.json
uv run studentbench-leaderboard human-leaderboard \
  --mode pairwise --output results/human-pairwise.json
```

Compare saved judge results against the human study:

```bash
uv run studentbench-leaderboard report results/judgments.jsonl \
  --output results/judge-report.json
```

Use `--level plan` for individual lesson-plan rankings; the default groups by
authoring model preset. Rankings remain separate by instrument, question
variant, rating scale, judging mode, and judge configuration. There is no
implicit overall score across different rubric criteria.

Single-plan rankings use mean ratings. Pairwise rankings use win share with
half credit for ties, after averaging repeated judgments within each matched
pair. Reports include sample counts, abstentions, mean absolute error,
Spearman correlation, and pairwise direction agreement against the mean human
preference. Undefined correlations are `null`.

These are descriptive estimates. Pairwise win shares depend on the comparison
schedule; they are not Bradley–Terry ratings. Human single-plan reference scores
were elicited while reviewing two plans side by side. They are useful checks
for an isolated-plan judge, but the elicitation contexts differ. Repeated
reviewers and plans are not independent observations. See
[the evaluation design](docs/design.md) before drawing conclusions about judge
quality or rank separation.

## Project boundaries and next steps

The [integration plan](docs/integration-plan.md) describes the proposed two-command
workflow for generating a new model's plans and evaluating them against existing
models, using our LiteLLM proxy over Tailscale or explicit direct provider access.
Those generation and evaluation commands are not implemented yet.

`../studentbench/` contains the original application, instruments, and data
collection logic. `../studentbench-analysis/` contains the existing research
analysis and earlier single/pairwise LLM judging experiments. This project
builds on their conventions while reading the frozen SSOT directly; it does
not connect to the production database or import the neighboring projects at
runtime.

Further calibration should use a fresh held-out student cohort for confirmation,
repeat model generations and vary the development examples. The SSOT also contains an AI-vs-human
supplement and historical LLM judge calls; neither is pooled into this initial
adapter. Bulk data storage is deliberately deferred. An access-controlled
GCS bucket with immutable release objects and checksum-verified downloads is
one future option; no bucket or upload has been configured.
