# StudentBench lesson-plan leaderboard

Evaluate lesson plans with an LLM judge **J**, then check its judgments against
the human expert ratings in the frozen StudentBench SSOT.

The initial implementation is a Python CLI and a small library. It supports
single-plan quality ratings (1–5), pairwise preferences (−2 to +2), descriptive
plan/model leaderboards, and agreement reports on matching human reviews. A
judge can abstain with `null`; a pairwise score of zero means a tie. The ranking
study evaluates model-level alignment on separate validation and test subjects.
No model calls are needed for setup or the tests.

## First public leaderboard

The November 2, 2026 release ranks 17 generator systems: the original 12
baselines and five new models. It covers 83 frozen benchmark contexts, 415 new
plans, 2,313 comparisons and 18,504 rubric answers, with no missing or abstained
answers. All 500 student-bootstrap fits were valid. The ranking is AI-judged;
the historical and new generation pipelines differ, as described in the report.

View the [public leaderboard](reports/2026-11-02/leaderboard.html), or download
its [CSV](reports/2026-11-02/leaderboard.csv) and
[aggregate JSON](reports/2026-11-02/leaderboard.json). The detailed original
judge-validation research report remains local at
`results/ranking-study-20260910/report.html`. Release inference used the Tailscale
proxy; the CLI's `auto` routing described below remains unchanged. The ledger
estimate was $410.79 against the $1,000 cap, including retained reservations for
failed or unknown-cost calls; it is not an invoice. GPT-6.1 Terra and MiMo 2.6
Flash/Pro were deferred at the user's request. For methods and future work,
see the [release notes](experiments/public-release/README.md) and
[roadmap](docs/roadmap.md).

## Generate and evaluate a new model

After preparing a benchmark once, the new-model workflow is:

```bash
uv run studentbench-leaderboard generate \
  --model provider/model-id --output results/new-model
uv run studentbench-leaderboard evaluate results/new-model
```

Replace `provider/model-id` with an exact available model ID. The first command
generates plans for the benchmark students; the second judges those plans
against existing plans for the same students and writes `evaluation.json`,
`leaderboard.csv`, and `report.html` in `results/new-model/evaluation/`.
The generation model and the benchmark's selected judge are separate.

For a joint leaderboard with several new generators, run `evaluate-many` with
their generation directories and an output directory. This fits one model graph
and includes new-to-new plan comparisons:

```bash
uv run studentbench-leaderboard evaluate-many \
  results/model-a results/model-b --output results/combined
```

Compatible judgments are reused from `results/judge-cache` across candidate
models; override this with `--cache-dir PATH` or `STUDENTBENCH_JUDGE_CACHE`.
Changing the judge route can require fresh baseline judgments once; subsequent
models reuse those results when their requests match.

The default `auto` connection uses OpenRouter. For GPT models, an available
`OPENAI_API_KEY` selects direct OpenAI access; for Claude models, an available
`ANTHROPIC_API_KEY` selects direct Anthropic access. Otherwise configure
`OPENROUTER_API_KEY`. Explicit `openrouter/` model prefixes or `--profile
openrouter` force OpenRouter. `--profile openai`, `--profile anthropic`, and
`--profile direct` select native access explicitly. Keys are read from the
environment; see [.env.example](.env.example) and
[connection.example.toml](connection.example.toml). The application does not
automatically load `.env` files.

```bash
# Inspect requests and counts without inference or credentials.
uv run studentbench-leaderboard generate \
  --model provider/model-id --output results/new-model --dry-run
uv run studentbench-leaderboard evaluate results/new-model --dry-run

# Check connection/authentication without making an inference request.
uv run studentbench-leaderboard doctor --model provider/model-id
```

Re-running a command reuses validated completed requests. Use `--limit N` for a
generation subset, `--comparisons-per-student N` for a smaller evaluation
schedule, and `--max-new-calls N` to bound new inference. `--workers` and
`--timeout` control execution. Parameters defining a saved run must match when
resuming. Incomplete runs retain their completed work and report missing
outcomes. A provider or Tailscale failure never changes the connection profile
automatically.

The internal LiteLLM proxy is an explicit option:

```bash
export LITELLM_PROXY_API_BASE='http://your-tailnet-host:4000'
export LITELLM_PROXY_API_KEY='your-proxy-key'
uv run studentbench-leaderboard --profile tailscale generate \
  --model your-proxy-model-alias --output results/new-model
uv run studentbench-leaderboard --profile tailscale evaluate results/new-model
```

Use the actual proxy URL and aliases configured by its administrator. Tailscale
provides connectivity; this code does not install or configure it. A custom
connection TOML can be passed with `--config PATH`. Its optional per-profile
`models` mapping translates the frozen judge's canonical ID into your proxy's
confirmed alias, and `revision` records deployment changes in cache identities.

## Prepare the benchmark once

The portable GRE Quant bundle freezes the original 83-student test cohort,
baseline plans, generation prompt, selected `enhanced_terra_high` judge, enriched
pretest context, and baseline judgment provenance. Preparation uses the local
SSOT and the frozen study artifacts described below:

```bash
uv run studentbench-leaderboard benchmark prepare \
  --study results/ranking-study-20260910 --output data/gre-quant-v1
uv run studentbench-leaderboard benchmark inspect data/gre-quant-v1
```

Alternatively, obtain a prepared bundle from the data owner and verify it with
`benchmark inspect`. Once prepared, generation and evaluation need only that
bundle and credentials, without the historical results directories, neighboring
repositories, or raw SSOT. The default path is `data/gre-quant-v1`; use
`--benchmark PATH` or `STUDENTBENCH_BENCHMARK` to select another bundle.

Student data and bundles remain outside Git. A fresh checkout needs the data
owner's source assets or a prepared bundle. The generation renderer is versioned
and its provenance records differences from the historical application pipeline.
AI candidate rankings and historical human reference results are labeled
separately; newly generated plans have no human grades.

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

For ad hoc judging, select a model identifier and configure credentials for the
resolved connection profile. For named judges, models and
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

The [roadmap](docs/roadmap.md) separates the first public leaderboard from v1.1,
which will test Jev and other decision models as lesson-plan judges.

Our own first-release runs use the Tailscale-only
[release runner](experiments/public-release/README.md), while the public CLI
retains its provider-key defaults.

The [integration plan](docs/integration-plan.md) records the implementation design
for the two-command workflow and its verification requirements. OpenRouter with
automatic native-key selection is the public default; the LiteLLM proxy over
Tailscale is an explicit internal connection option.

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
