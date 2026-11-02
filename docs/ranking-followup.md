# Qwen and small panels

The follow-up retains the same 120 validation plan pairs and 12 lesson-plan
generator models. It copies 480 completed judgments from the prior extension:
Terra, DeepSeek Flash, GLM Flash and Kimi. Three frozen panel recipes use equal
votes with one model per provider family: Terra/DeepSeek Flash/GLM Flash;
Terra/DeepSeek Flash/Kimi; and Terra/DeepSeek Flash/GLM Flash/Qwen. A missing
member leaves the panel judgment missing. It is never replaced with a tie or
silently omitted from the panel.

Panel membership was chosen after observing prior validation results. This is
exploratory development on reused validation subjects, not another independent
test and not a replacement for the original locked Terra judge. The Qwen variant
requests the exact `qwen/qwen3.8-max-0902` model with high reasoning and explicitly
omits `max_tokens`; provider defaults and model limits still apply. Copied
judgments retain their original reasoning settings and output budgets, so this
does not isolate a model-only effect of Qwen.

The Qwen preflight returned HTTP 404 because its only endpoint was excluded by
the existing workspace zero-data-retention guardrail. No Qwen verdict was
produced and no bulk requests were dispatched. `availability.json` preserves the
reason and the error artifact's SHA256. All 600 scheduled jobs remain present:
480 copied successes and 120 unavailable Qwen outcomes. Both triplets are
evaluable; the quartet requiring Qwen remains unavailable. Planned concurrency
was 128 total and per model, while the actual preflight concurrency was one.
`execution.json` records this distinction.

To prepare the historical follow-up, provide the same local provider catalog JSON
used for that study. The preparer copies its bytes into the frozen study and
rejects changed existing inputs. It does not fetch a catalog:

```sh
uv run python experiments/ranking-followup/prepare.py \
  --provider-catalog /path/to/provider-catalog.json
```

Run the finalizer and independent verifier from the repository root:

```sh
uv run python experiments/ranking-followup/finish.py \
  --study results/ranking-followup-20260911 --bootstrap 500

uv run python experiments/ranking-followup/verify.py \
  --study results/ranking-followup-20260911
```

Both commands are offline. They patch both API adapters to fail if a call is
attempted. `prepared_inputs.json` pins the manifests, prompts, schedules, panel
recipes and both reuse ledgers. Every copied source and destination file must
match `reused_calls.json`. The 480 copied requests are reconstructed in the prior
study using its original catalog, configuration and recovery-error provenance.
Qwen requests are reconstructed separately in the new study. Reused provider
costs are never recalculated against the newer catalog. Unexpected call files,
undeclared missing outcomes, changed prepared inputs or a changed original
selection cause verification to fail.

`focused_validation_analysis.json` fits the same Bradley–Terry model with ridge
1, equal pair weights, equally weighted rubric families and 500 subject
bootstraps. It includes human and AI model leaderboards, rank intervals,
per-family comparisons, subject-balanced sensitivity and nested density
subsampling. `paired_deltas.json` uses the same full human-reference subject
resample for Terra and every candidate, refitting both model leaderboards on
each draw. Its seven candidate-minus-Terra entries retain unavailable candidates
with undefined metrics. Confidence intervals condition on the observed judgments
and connected fits; they do not adjust for choosing panels from earlier results
or for multiple exploratory comparisons.

Exports are `exports/focused_validation_judges.csv` and
`exports/focused_validation_model_leaderboards.csv`. `summary.json` distinguishes
new Qwen attempts from all 480 reused controls. Underlying member costs can
appear in several panel rows, but are counted once in experiment spending;
panels themselves make no API calls. The rejected Qwen attempt has no returned
usage, so its billing remains unknown. A zero known-cost subtotal does not mean
the attempt was free. `verification.json` contains source/input/call/output
inventories and separate validation results for the old and new catalogs.

The finalizer writes only this follow-up directory. It does not rewrite earlier
analyses or render earlier reports. Source snapshots and final report publication
remain separate steps.
