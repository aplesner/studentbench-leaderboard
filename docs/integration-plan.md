# Integration plan: generate plans and evaluate a new model

Status: implemented with offline verification, November 2, 2026. The original
judge research was committed as `7f5b347`; this document records the integration
design and acceptance checks. Implementation used Terra-based agents and
independent Sol-based code reviews. Live provider and Tailscale smoke tests
require the corresponding credentials and endpoint configuration.

## Outcome

After one-time data and connection setup, adding a generator requires two
commands:

```sh
uv run studentbench-leaderboard generate \
  --model provider/model-id --output results/new-model
uv run studentbench-leaderboard evaluate results/new-model
```

`generate` creates one lesson plan per frozen benchmark student context.
`evaluate` compares those plans with existing plans for the same students and
writes a model leaderboard, uncertainty, rubric breakdowns, coverage, and an HTML
report. Both commands resume validated completed work. Evaluation also reuses
compatible saved baseline judgments, so a new model does not require regrading
every old comparison.

The initial benchmark is GRE Quant. The generation model and judge model are
separate settings. Replace `provider/model-id` with an exact available model ID.
Internal proxy aliases are used only with an explicit proxy profile. The selected
judge is
the frozen `enhanced_terra_high` configuration, including its input enrichment,
not the differently configured bundled `judge_5` example.

## Connection design

Use a small shared client around the existing LiteLLM SDK for generation and
judging. The default profile is `auto`: use an available OpenAI key for GPT models
or an Anthropic key for Claude models; otherwise use OpenRouter. Canonical
`openai/` and `anthropic/` model IDs follow the same key selection. An explicit
`openrouter/` prefix or `openrouter` profile forces OpenRouter; explicit native
profiles force the named provider. Runtime failures do not change routes.

The `tailscale` profile is an explicit internal option, calling the existing
LiteLLM proxy at a supplied tailnet URL. Tailscale supplies network connectivity;
the application makes ordinary HTTP requests to that URL. It does not manage
Tailscale login, serve a gateway, or change the proxy's routing policies.

An explicit `direct` profile uses LiteLLM's provider adapters and the
user's provider credentials. Both profiles run the same benchmark, generation
schema, judge, scheduler, and reporting code. A direct run should work on a
machine without Tailscale or proxy credentials:

```sh
uv run studentbench-leaderboard --profile direct generate \
  --model openai/gpt-model-id --output results/new-model
uv run studentbench-leaderboard --profile direct evaluate results/new-model
```

The example configuration is:

```toml
default_profile = "auto"

[profiles.auto]
transport = "auto"

[profiles.tailscale]
transport = "litellm_proxy"
api_base_env = "LITELLM_PROXY_API_BASE"
api_key_env = "LITELLM_PROXY_API_KEY"

[profiles.direct]
transport = "direct"
```

The proxy adapter uses `litellm_proxy/<alias>` and passes the configured API base
and key per request. The direct adapter uses an explicit provider/model ID.
Resolve benchmark judge identities to profile-specific routes through a small
model mapping; adding a generator through `--model` must not require editing
Python. `--profile` selects a profile; otherwise the TOML's `default_profile`
applies, falling back to `auto`. Endpoint and credentials use the configured
environment-variable names. `--benchmark` or `STUDENTBENCH_BENCHMARK` selects
the bundle, defaulting to `data/gre-quant-v1`. Document the resolved configuration
with credentials redacted. Keep actual credentials and internal hostnames in
environment variables or ignored local configuration.

Avoid process-wide LiteLLM proxy flags. Direct mode must detect or override
ambient proxy routing so its name accurately describes where requests go. An
unreachable tailnet, missing credentials, unavailable alias, or unsupported
request setting produces a useful error. Switching to direct access requires an
explicit profile choice. Generation and evaluation may use different profiles;
each records the route it actually used.

Preserve requested and returned model identity, alias mapping/revision, supported
settings, and transport provenance. Benchmark routes should pin the intended
upstream model and record the proxy's effective routing configuration where
available. Undocumented proxy-side substitution cannot be treated as equivalent
to the frozen judge. Cache matching is conservative across routes and model
revisions; credentials are excluded from request identities and artifacts.

This design follows the [LiteLLM proxy provider interface](https://docs.litellm.ai/docs/providers/litellm_proxy)
and [direct SDK interface](https://docs.litellm.ai/docs/python_sdk). A tailnet
hostname may use [Tailscale MagicDNS](https://tailscale.com/docs/features/magicdns).
Tests check the pinned SDK through mocks at its final provider transports.

## Implementation sequence

| Milestone | Implementation | Completion check |
| --- | --- | --- |
| 1. Connection profiles | Add `llm_client.py` and profile loading in `config.py`; inject the shared client into generation and configurable judges. Add credential-free examples and `doctor`. | Mocked requests prove proxy/direct dispatch, explicit parameter handling, credential redaction, and clear Tailscale failures. `doctor` checks connectivity/authentication without inference. |
| 2. Frozen benchmark and plan catalog | Add `benchmark.py` and `plan_catalog.py`; separate read-only SSOT baseline plans from new run artifacts. Prepare a checksum-verified GRE Quant bundle and promote the selected judge/input configuration. | A prepared bundle works without sibling repositories or historical experiment directories. New plans can be registered without changing the SSOT fingerprint or files. |
| 3. Generation command | Add `generation.py`, versioned prompt rendering and a plan schema; wire `generate` into `cli.py`. Persist exact requests, raw output, validated plans, errors, and usage. | A synthetic model produces the expected student-matched plans; interrupted runs resume; changed inputs/settings invalidate reuse; invalid plans stay explicit failures. |
| 4. Evaluation command | Add `evaluation.py` with deterministic candidate/baseline scheduling, a compatible judgment cache and progress accounting. Generalize ranking/report adapters to the combined plan catalog. | A new model enters a connected leaderboard; every comparison uses one student's context; compatible old judgments are reused and incompatible judgments are rejected. |
| 5. Release workflow | Produce evaluation JSON/CSV/HTML; document setup, profiles and the two commands; verify the installed package from a clean checkout. | One offline end-to-end fixture covers both profiles. A bounded live proxy smoke run and a separately configured direct smoke run establish real transport compatibility. |

Keep implementation milestones in separate reviewable commits. Keep the original
study runners, frozen judge versions, historical request hashes, and report reproduction
behavior intact. Inject transport without rewriting historical artifacts.

## Benchmark and generation decisions

Start with a versioned GRE Quant benchmark covering the original study's 83 test
students and the 12 baseline generator presets. These students are already
exposed research data: the new leaderboard is an evaluation benchmark, not a new
untouched confirmation of judge quality. Record subject membership, available
baseline plans, prompt/context versions, rubric, judge settings, ranking policy,
and source checksums in the bundle manifest.

The released `gre.quant.plan.lean.v3` prompt is the starting point. Its source
exists in `studentbench_data_ssot/5_prompts/`. The new `raw-pretest.v1` renderer
strips generation-forbidden skill/difficulty metadata and uses the verified
question supplement. The schema uses fixed option fields for structured output;
validation removes unused option entries. The initial duration is 60 minutes,
matching the released baseline plans. Raw responses and normalized plans are
saved separately. This validation policy does not run the historical answer
verifier. Bundle provenance explicitly records that historical pipeline
equivalence has not been established. Original input rendering and answer edits
therefore remain limitations when interpreting generator-system quality.

Generation inputs contain only the benchmark's permitted student context. Human
grades, reviewer identities, baseline verdicts, and posttest outcomes are excluded.
Save model/settings, request/schema/prompt/context hashes, response identity,
usage, duration, parser version, and any documented transformation. The initial
release generates one draw per student and reserves draw identity in the request
contract for future repeated-generation experiments.

The bundle must package or reproducibly prepare the selected judge's verified
question-bank supplement and baseline assets. Bulk student data and generated
plans remain outside Git. Initially accept the local SSOT archive plus explicit
paths to required supplements, verify checksums, and give actionable setup
errors. Hosting authenticated downloads is a later storage decision.

## Comparison schedule and leaderboard

For the initial release, compare the candidate with every available baseline
plan for the same benchmark student. Save the exact schedule before dispatch;
show counts and estimated cost in `--dry-run`. This is a transparent first policy,
not a claim that maximum density is optimal. Expose a deterministic comparisons
per student limit and seed for smaller runs. Freeze each selected schedule and
report its coverage; choose future defaults using declared cost/stability checks.

Retain the compatible baseline-to-baseline judgments as reference edges and add
candidate edges using the same selected judge, rubric, context and orientation
rules. Reuse requires reconstructing matching requests and validating results;
the original research cache needs an explicit import adapter and provenance
ledger. Use reproducible A/B shuffling and anonymous plans for new requests.
The CLI also shares validated provider calls through `results/judge-cache`
(`--cache-dir` or `STUDENTBENCH_JUDGE_CACHE` overrides it), so baseline judgments
recomputed for a different route can be reused by later candidate models.

Reuse `fit_bradley_terry` and criterion aggregation from `ranking.py` with
student-cluster bootstrap intervals. `PlanCatalog` supplies a validated
Dataset-compatible catalog; evaluation builds model/criterion rows directly from
the frozen bundle and generated plans. Refit the combined
leaderboard and report the candidate's rank interval and pairwise win estimates.
Report equal-pair results with equal-student sensitivity so extra comparisons
for one student have visible consequences. Disconnected graphs and inadequate
model coverage do not produce a global ranking.

Label AI model rankings and historical human reference rankings separately.
New candidate comparisons have no human grades; historical human agreement
validates the chosen judge on its original data and cannot become a claimed
human grade for the candidate. Do not combine old human votes and new AI votes
into one unlabeled leaderboard.

## Execution and acceptance

Both commands support `--dry-run`, bounded concurrency, timeouts, bounded retries
for transient errors, completion counts, and an explicit provider-attempt limit
that includes retries. Dry runs show planned requests and counts; cost is marked
unknown when no price information is available. Usage is retained with each
response. Save each
completed result atomically; enforce one writer per run or use a run lock.
Failures and abstentions stay missing, and partial reports show the scheduled
denominator. Fully cached reruns make no external calls. Refuse outputs inside
the SSOT and keep secrets out of saved errors and reports.

The feature is complete when a new user can install with the lockfile, prepare
the documented inputs, configure either connection profile, run the two commands
for a previously unseen model, interrupt/resume them, and obtain a reproducible
leaderboard and report. All existing tests must still pass, and offline fixtures
must exercise actual CLI-to-generation-to-judging-to-ranking behavior with a
new model absent from the original SSOT.

Before the first live run, configure the selected provider's credentials. For
the optional proxy, also supply its URL/port/API prefix, generator/judge aliases,
and upstream mapping/revision. No internal address or working MiMo route is
assumed. A frozen bundle has been prepared and checked locally. The automated
acceptance suite uses synthetic model responses and makes no paid inference
calls; live transport smoke tests remain environment-specific verification.

## Verification completed

- All 231 offline tests pass, including the original research tests and new
  command, routing, resume, retry-budget, coverage, cache, and filesystem checks.
- Sol reviews covered routing, benchmark/generation, and evaluation. Reproduced
  findings received regression tests and were reviewed again after fixes.
- The real bundle verifies 83 students, 238 baseline plans, 12 generator presets,
  and 293 saved baseline judgments. A native-route dry run reuses all 293;
  OpenRouter schedules fresh baseline judgments and shares them across later runs.
- The built wheel imports from outside the checkout and includes the commands
  and packaged judge assets. No provider or proxy inference was dispatched.
