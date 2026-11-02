# Validation extension runner

The September 11 extension compares explicitly scheduled judges on the original
validation subjects. The original test results have already been inspected, so
this phase is a validation extension, not another independent test. Its runner
is `studentbench_leaderboard.extension`; the original study runner and frozen
artifacts are separate.

```sh
uv run python -m studentbench_leaderboard.extension \
  --study results/ranking-extension-20260911 \
  run results/ranking-extension-20260911/provider_jobs.json \
  --check-only --max-new-calls 1080

```

For a newly prepared study whose variants explicitly declare their output-limit
choice, use `--workers 128 --per-model-limit 128`, or raise both to 512 when the
schedule and provider capacity support it. The original schedule includes
unavailable and terminal requests; its verification command above is not an
instruction to retry those requests.

`run_jobs(study, jobs, workers=128, max_new_calls=1200, timeout=420,
max_attempts=3, *, check_only=False, per_model_limit=128)` is the equivalent Python interface.
`--check-only` reconstructs every request and reparses every cache without
creating an API client. It may create or extend the local configuration inventory.
The runner checks the SSOT fingerprint, actual plan/author identities, canonical
pair IDs, frozen judge configuration and input supplement, and validation
membership before dispatching new calls. Duplicate jobs are rejected.

The OpenRouter route strips exactly the `openrouter/` adapter prefix and sends
one complete model ID, such as `anthropic/claude-opus-5`. It never sends a
different-model fallback list. It permits compatible hosting endpoints for that
same model using `provider.allow_fallbacks=true` and
`provider.require_parameters=true`. The requested model must exist in the saved
`provider_catalog.json`; accepted returned model identifiers are the exact ID
or that entry's recorded canonical slug. The returned hosting provider is saved
when OpenRouter supplies it. No provider name is inferred when absent.
[OpenRouter API reference](https://openrouter.ai/docs/api_reference/overview)

The original nine OpenRouter candidates requested `reasoning.effort=high` and a
32,768-token completion ceiling. This was a fixed experiment budget chosen to
bound reasoning plus answer generation, not a provider requirement or a measured
optimal budget. Two GLM pairs remained missing after repeatedly exhausting that
cap. Their frozen requests and outcomes remain unchanged.

New OpenRouter variants must explicitly set `max_tokens` in `variants.json`:
a positive integer sends that ceiling, while JSON `null` omits the parameter
entirely and leaves the output budget to the provider. Omission does not mean
unlimited output; the provider's defaults and model limits still apply. The
runner adds no replacement cap. Native OpenAI calls already omit this parameter.
For example, a variant can include `"max_tokens": null` alongside its explicit
model ID and reasoning effort. The chosen body is recorded and hashed in both
requested and effective transport metadata.

For compatibility, old metadata without a `max_tokens` field reconstructs the
original 32,768-token request exactly. Existing caches are reusable byte for byte.
A missing OpenRouter job with that legacy implicit choice is rejected before any
API call; new variants must make the choice explicit. `--check-only` can still
reconstruct old missing jobs for verification and emits a
`legacy_cap_reconstruction` notice with the inherited cap and
`execution_allowed=false`. Changing a frozen variant's cap is rejected by its
inventory. Use a new variant to evaluate another output budget.

The catalog must advertise the requested effort. Effort names
do not specify equal reasoning budgets across providers: providers translate
these settings differently. There is no temperature, hidden example injection,
response-healing plugin, or alternate-model routing added by this runner.
[OpenRouter reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

Every candidate starts with strict JSON schema output. Only an explicit
unsupported-format API error can trigger schema → JSON object → prompt-only JSON
compatibility changes, within the same three-attempt limit. JSON object and
prompt-only requests include the unchanged semantic schema in their actual
system message. Each transition, exact request, effective format and request
hash is saved. Local semantic parsing remains strict in every mode: incomplete
responses, refusals, wrong rubric IDs, inconsistent winner/strength and invalid
JSON are errors. They are not repaired or turned into ties. This preserves the
semantic result contract while making reduced provider enforcement visible.
[OpenRouter structured output documentation](https://openrouter.ai/docs/guides/features/structured-outputs)

Repeated provider format failures led to the explicit serialization adapter
`extension-provider-serialization.v1`. New OpenRouter outputs, regardless of
provider, may have one complete outer JSON or unlabelled code fence with no
surrounding prose. The JSON must then be either the exact requested keyed object,
or exactly `{"criteria": [...]}`. Array entries must all use the same single
`id` or `criterion_id` field, contain each exact requested ID once, and have only
the required evidence/reason/winner/strength fields. Mapping the array to keys
changes no evidence or verdict. The unchanged strict semantic parser validates
the result. Native calls and already-saved strict preflight successes retain
their original strict parsing rules.

Every newly parsed provider record saves the parser contract/version/hash,
raw and canonical response hashes/text, transformations and an evaluation ID
that includes the parser contract. Adapted outputs explicitly set
`provider_schema_compliant=false`; successful local adaptation is not described
as provider schema enforcement. Missing fields, approximate IDs, duplicate IDs,
mixed identifier styles, extra fields, contradictory strengths and surrounding
prose remain failures.

Version `extension-provider-serialization.v2` extends the lossless wrapper rule
to a bare array or an object containing exactly one named array, such as
`evaluations`, `items` or `results`. It records the wrapper's exact name in the
transformation metadata. No extra root metadata is discarded. Exact criterion
count/IDs, uniform identifier fields, evidence fields and semantic validation
remain the same. New provider calls and recovery of missing jobs use v2. Existing
v1 caches are reparsed with their saved v1 contract; v1 recovered calls also retain
the earliest-compatible selection under v1, even if an earlier attempt would
become compatible under v2.

Native OpenAI calls use the existing `LiteLLMJudge` adapter, including its strict
schema and reasoning effort. That adapter does not send a completion-token cap;
native provenance therefore does not claim the OpenRouter 32,768-token ceiling.
The one-example intervention is entirely part of its frozen user template. Its
example is neither selected nor inserted by the runner.

Each runner process now defaults to 128 workers and 128 concurrent calls per
exact configured model. Both `--workers` and `--per-model-limit` accept integers
from 1 to 512. For example, `--workers 128 --per-model-limit 128` can admit all
120 jobs of one model immediately; `--workers 512 --per-model-limit 512` supports
more than 500 concurrent requests when enough jobs exist. There is no hidden
eight-call throttle. The effective model cap is the smaller of the two settings,
and each start event records the requested and effective values. These limits
apply per process, not across independently launched runners.

Concurrency is an execution setting: it changes no judge
configuration, request payload, presentation order, retry policy or cache identity.
Future invocations use round-robin model queues and reserve model capacity before
submitting a worker task. A slow model's waiting backlog therefore consumes no
worker threads and cannot starve other models. Earlier running processes retain
the scheduling settings with which they started. The optional cap does not
change or restart an existing process.

Historically, the September 11 batches began with 24 workers and eight calls per
model in each process. The final DeepSeek Flash resume used 24 workers and a
24-call model cap. Higher admission limits reduce local queue time; they do not
make an individual hosted response generate faster. In the completed Flash
records, median duration after admission was 650.1 seconds, including any retries,
and median completion usage was 13,529 tokens. That duration excludes local
dispatcher waiting. Long reasoning outputs and hosting latency therefore
remained significant even after the scheduling bottleneck was removed. Provider
rate limits can also restrict actual throughput; the saved attempts expose any
429 responses instead of treating configured concurrency as achieved throughput.

The SDK performs no automatic retries. The runner permits at most three total
attempts per scheduled missing job, including format transitions and transient
timeouts, connection failures, rate limits and server errors. `max_new_calls`
limits missing jobs, not total HTTP attempts or dollars. A failed result can be
retried by invoking the runner again; each invocation saves distinct error
files. Run one process at a time per study directory to avoid concurrent writers
issuing the same missing job.

The 420-second SDK timeout is a transport setting, not a guaranteed total wall
deadline: continued response activity can keep a connection alive. A future
hard wall-clock cancellation policy would need a separate execution contract,
an explicit interrupted-attempt record and unknown billing until the provider
reports usage. It should not silently resubmit a possibly running paid request.
The current experiment does not change or terminate ongoing calls to impose
such a new policy.

The extension cache keeps the analysis-compatible fields `judge`, `messages`,
`response_schema`, `raw_response`, `ratings`, `normalized_ratings`, `usage`,
`estimated_usd`, `response_model` and `response_id`. New fields preserve
`requested_messages`, requested/effective formats and transport contracts, the
full OpenRouter response payload, returned provider, cost provenance, run ID and
runner source identity. Neither credentials nor request headers are saved.
Errors preserve returned invalid text/payloads when available, so coverage losses
remain auditable. Error messages redact environment credential values.

Copied native controls require `imported_calls.json` with this shape:

```json
{
  "source_study": "/absolute/path/to/original-study",
  "calls": {
    "enhanced_terra_high--PAIR_ID--0.json": {
      "sha256": "SHA256_OF_ORIGINAL_FILE_BYTES",
      "request_id": "ORIGINAL_REQUEST_ID"
    }
  }
}
```

The runner verifies the imported byte hash, original request identity, frozen
prompts/schema, dataset and plan identities, and reparsed/normalized scores.
It leaves imported records byte-for-byte unchanged. If a ledger-listed control
is missing, it requires restoration of that cache instead of issuing a new call.
New caches must reproduce their requested/effective transport contracts and
parsed results before reuse. `extension_inventory.json` freezes the manifest,
provider catalog, variants and configuration files, permitting only additions
of previously absent variants/files.

OpenRouter's returned `usage.cost` is used when present. Otherwise the runner
uses the saved catalog's nominal input, cached-input and output prices, explicitly
marking them as estimates; time, endpoint and context-price overrides may differ.
Completion tokens already include reasoning and are counted once. Native costs
use the original study's fixed token rates. Successful-call totals and failed
attempts are separate: each error artifact retains returned usage and its
reported/estimated cost, or an explicit unknown cost when unavailable. This
includes paid responses rejected by strict parsing and HTTP-200 error envelopes.
Combined amounts are recorded/estimated subtotals excluding unknown charges.
Top-level and per-choice error envelopes use their explicit error code for
transient retry classification, including a choice-level 502 connection failure.
Usage from failed or timed-out attempts may be unavailable. Reused controls retain
their original response dates and original cost estimates, rather than appearing
as new extension spending.

For bring-your-own-key calls, `usage.cost` can be only the OpenRouter charge.
`economic_cost(record)` separately returns `economic_usd`, `router_charge_usd`,
`upstream_inference_usd`, `is_byok`, `basis`, `is_estimate` and `complete`. When
BYOK is explicitly reported, economic cost adds the returned gateway charge and
`usage.cost_details.upstream_inference_cost`; missing components remain unknown.
For ordinary OpenRouter billing, the upstream amount is not added again. This
avoids treating a zero gateway fee as free inference, while preserving the
original cached `estimated_usd` and request identities.
[OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting),
[BYOK billing](https://openrouter.ai/docs/guides/overview/auth/byok)

After all available jobs finish, run the offline verifier:

```sh
uv run python experiments/ranking-extension/verify.py \
  --study results/ranking-extension-20260911
```

It deduplicates the provider/example schedules and requires all results except
explicit provider-availability exclusions or verified individual terminal
failures. Availability exclusions require reasons and hashed API-error evidence.
The verifier reconstructs/reparses every cache with
`check_only=True` while additionally patching both API adapters to raise on any
attempted call. `verification.json` contains call, input and source hashes;
requested/returned model and provider/format provenance; imported/new counts;
and separate successful/failed-attempt economic cost summaries. Unknown error
costs remain unknown. The verifier neither changes guardrails nor changes cache
records.

Saved format failures can be recovered explicitly without another API call:

```sh
uv run python -m studentbench_leaderboard.extension_recovery \
  --study results/ranking-extension-20260911 \
  results/ranking-extension-20260911/preflight_jobs.json

# After reviewing the preview, publish only the recoverable missing jobs:
uv run python -m studentbench_leaderboard.extension_recovery \
  --study results/ranking-extension-20260911 \
  results/ranking-extension-20260911/preflight_jobs.json --write
```

The recovery tool requires the complete saved provider payload and exact
reconstructed original strict-schema request. It chooses the earliest compatible
attempt deterministically by observation timestamp and relative filename. It
copies the source text, usage, request, run ID and observation time into a new
cache and preserves `recovered_from_error: {file, sha256}`. Original source errors
and existing successes are unchanged; unavailable original duration and runner
hash remain null. Reuse revalidates the source hash, copied fields and earliest
compatible choice. The recovery source attempt is the same paid response as the
new success, so the verifier counts it once; duplicate paid retries remain
additional costs. Native routes and explicit availability exclusions are
ineligible for recovery.

After bounded retries, an individual job can remain terminally missing rather
than increasing the frozen output cap or fabricating a judgment. Declarations
are explicit and are never created by the runner or verifier:

```json
{
  "schema_version": 1,
  "jobs": [{
    "variant": "or_example_high",
    "pair_id": "EXACT_SCHEDULED_PAIR_ID",
    "draw": 0,
    "status": "terminal_failure",
    "reason": "Three completed attempts exhausted the frozen completion cap",
    "declared_at": "2026-09-11T12:00:00+00:00",
    "evidence": [
      {"file": "errors/FIRST_ATTEMPT.json", "sha256": "FIRST_BYTE_HASH"},
      {"file": "errors/SECOND_ATTEMPT.json", "sha256": "SECOND_BYTE_HASH"},
      {"file": "errors/THIRD_ATTEMPT.json", "sha256": "THIRD_BYTE_HASH"}
    ]
  }]
}
```

Save declarations as `terminal_failures.json` only after the requisite attempts
exist. The shared `load_terminal_failures` loader requires a missing original
scheduled job and at least three distinct failed API attempts. It verifies each
saved error's byte hash, job identity, frozen request, timestamp and run/attempt
identity; repeated copies of one returned response do not count as new attempts.
Local errors without evidence of an API attempt do not qualify. Recoverable
complete responses must be recovered instead of discarded. A success cannot be
declared terminal, and whole-provider availability exclusions remain separate.

The verifier, finalization script and paired-delta execution gate share this
validation. The original schedule, full human-reference cohort and missing-call
coverage remain unchanged. Terminal jobs contribute no AI rating, tie or
abstention; disconnected fits and insufficient metrics remain unavailable.
Summary and paired artifacts expose declaration hashes, details and counts.
The verifier also inventories the actual derived-judge TOMLs/manifests, paired
protocol/amendments, execution audit and original-preservation check.

The offline tests in `tests/test_extension.py` exercise actual SDK argument
construction with a mocked client, model identity, swapped-order normalization,
format compatibility and cache provenance, invalid-output rejection, bounded
retries, native calls, import byte preservation, validation scope and cost
accounting. They perform no network requests.
