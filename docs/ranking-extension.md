# Provider and human-example validation extension

The September 11 extension compares the requested OpenRouter judges on the same
120 human-reviewed validation pairs used in the original second round. These
pairs cover 82 student contexts and 12 lesson-authoring model presets. The target
is alignment between model leaderboards, rather than accuracy on individual
plan verdicts.

The original test results had already been inspected when this extension was
requested. The extension is exploratory validation. It does not replace the
original selection or constitute a new held-out test. The original report is
archived as `report-before-extension-20260911.html`; the current `report.html`
includes the later results in a separate section.

## Conditions

All nine requested provider IDs have a separate frozen TOML configuration under
`experiments/ranking-extension/configs`. The executed configurations, verified
pretest bank, raw requests and responses remain under the ignored local
`results/ranking-extension-20260911` directory. Each provider receives the same
system prompt, rubric, enriched pretest context and seeded A/B order, with
explicit high reasoning effort. A high setting does not imply the same amount
of computation across providers. The OpenRouter output limit is 32,768 tokens.

The native Terra-high and Luna-high controls reuse 120 original judgments each;
their bytes and request identities are checked before reuse. The one-example
condition changes only Terra's user template, placing one development comparison
with two real human reviewers' gradings before the current task. The example
preserves reviewer disagreement and contains no author or reviewer identities.
Its selection, full input and hash are saved locally. See
[example-calibration.md](example-calibration.md).

The source configuration for the example intentionally does not contain the
human grading text or a runnable replacement for it. Run `prepare_example.py`
and `prepare.py` with the local SSOT and original artifacts to materialize the
complete, frozen configuration under `results`. Run that prepared configuration.

Muse's only available endpoint failed the account's existing zero-data-retention
guardrail during preflight. The account policy remains unchanged. The original
nine-model and six-family panel recipes remain frozen; a separate availability
amendment defines eight-model and five-family panels, excluding Muse solely
because of that recorded access failure. No judgment performance informed panel
membership. See [extension-panels.md](extension-panels.md).

## Execution and analysis

The runner resumes validated caches and retains failed attempts separately. It
does not silently substitute model IDs, change effort settings, or treat missing
judgments as ties. See [extension-runner.md](extension-runner.md) for transport,
retry and cost details.

```sh
uv run python experiments/ranking-extension/prepare_example.py
uv run python experiments/ranking-extension/prepare.py
uv run python -m studentbench_leaderboard.extension \
  --study results/ranking-extension-20260911 run \
  results/ranking-extension-20260911/provider_available_jobs.json \
  --workers 24 --max-new-calls 960
uv run python -m studentbench_leaderboard.extension \
  --study results/ranking-extension-20260911 run \
  results/ranking-extension-20260911/example_jobs.json \
  --workers 8 --max-new-calls 120
uv run python experiments/ranking-extension/finish.py
uv run python experiments/ranking-extension/verify.py
```

Preparation requires the original study, public provider-catalog snapshot and
availability amendment already described in the local artifacts. The commands
above document this extension, rather than creating a new cohort automatically.
The full original schedule remains the analysis input, so unavailable conditions
are visible. The filtered schedule controls which API calls can execute.

`finish.py` reparses cached outputs before fitting Bradley–Terry leaderboards,
exports CSVs, computes student-context bootstrap intervals and sampling-density
curves, then rebuilds the HTML report. Paired bootstrap comparisons reuse the
same sampled students for human, control and candidate rankings. Their intervals
describe sampling uncertainty conditional on the saved model outputs and chosen
example; they do not cover repeated generation or alternative examples.

OpenRouter's gateway charge can be zero for enterprise keys while upstream
inference still incurs a charge. The cost summary distinguishes router charges,
reported upstream costs, native token estimates, imported controls and saved
failed attempts. Unknown failed-attempt charges remain unknown. Panel costs are
the attributed sum of their members; aggregating already saved judgments incurs
no additional API calls.

## Reading the original ablations

- **Weighting** compares equal weight per plan pair with equal total weight per
  student context. It checks whether students with more available plans dominate
  the leaderboard. The small original dense-sampling gain disappears with equal
  student weights.
- **Reference** changes which human comparisons define the reference ranking:
  the scheduled human-reviewed criteria, the whole split, or exactly the criteria
  the judge scored. These expose cohort and missingness sensitivity.
- **Rubric** removes answer accuracy, test-taking strategies, or both from both
  human and AI rankings. These are diagnostic changes to the ranking target,
  rather than improvements to the frozen judge. The original test was not used
  to choose a new rubric.

Original rounds contained no human-graded examples. The separate Terra example
condition is the first such experiment. The original four-model panels combine
GPT-5.6 Luna, Terra and Sol with GPT-6 Astra, either all at low effort or all at
high effort; they were not cross-provider panels.
