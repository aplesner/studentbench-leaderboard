# First public leaderboard run

The completed November 2, 2026 release adds five generator systems to the 12 frozen
GRE Quant baselines. It covers 83 student contexts, 415 new plans, 2,313
comparisons including new-to-new pairs, and 18,504 rubric answers. No rubric
answers were missing or abstained, and 500 of 500 student-bootstrap fits were valid. GPT-6.1 Terra and MiMo v2.6
Flash/Pro are explicitly deferred at the user's request while their proxy aliases
are unavailable; their initial access checks remain in the audit history.

**All our authoring and judging calls use the existing Tailscale LiteLLM proxy.**
This release-specific choice does not change the public CLI's automatic
OpenRouter/native-key routing. The runner rejects non-Tailscale profiles.
The total authorized inference estimate cap was $1,000. The ledger estimate was
$410.79, including retained reservations for failed or unknown-cost calls; it is
not a reconciled invoice. The 272 offline tests passed and Sol reviews cleared.

The versioned public snapshot is available as
[HTML](../../reports/2026-11-02/leaderboard.html),
[CSV](../../reports/2026-11-02/leaderboard.csv), and
[JSON](../../reports/2026-11-02/leaderboard.json). The full detailed historical
judge-validation report remains local at `results/ranking-study-20260910/report.html`.
The next judge-method work is Jev and other decision judges in v1.1; it does not
change this release's frozen judge or ranking.

Local, ignored inputs are:

- `data/gre-quant-v1/`: the verified portable benchmark.
- `results/public-release-20261102/protocol.json`: frozen model IDs, settings,
  data paths and authorized cap.
- `results/public-release-20261102/connection.toml`: Tailscale profile and
  administrator-confirmed alias mappings; credentials remain in environment
  variables (`LITELLM_BASE_URL` and `LITELLM_API_KEY` for this environment).
- `results/live-model-smoke-20261102/openrouter-catalog.json`: the frozen
  catalog used for token-price estimates, not for inference routing.

Run from the repository root after installing the locked environment:

```sh
uv run python experiments/public-release/run.py smoke --models MODEL_ID --max-new-calls 1
uv run python experiments/public-release/run.py generate --workers 8 --max-new-calls 1000
uv run python experiments/public-release/run.py evaluate --workers 8 --max-new-calls 5000
uv run python experiments/public-release/run.py summary
```

The runner resumes validated generations and judgments. Use `--models` to select
specific enabled protocol IDs for generation; joint evaluation requires all
enabled models. Deferred entries remain in the protocol with `enabled = false`.
Availability failures never change the model or route. Missing proxy aliases
must be supplied by the proxy administrator before those runs can proceed.
Generated content that fails validation remains a failed attempt, with the raw
response and usage retained for diagnosis.

A fresh run initializes an empty shared ledger before its first call. The
original early compatibility calls were seeded from their complete legacy smoke
traces; a missing ledger alongside historical artifacts requires restoring the
ledger or complete cost records before continuing. Legacy smoke migration accepts
only the original smoke artifacts; later evaluations, pilots, or other extra
files require the original ledger.

Every dispatch reserves a conservative catalog-based amount in a shared locked
ledger. The input rate includes the highest prompt/cache-write rate across
catalog tiers. Completed calls with usage release unused reservations; failed
or unknown-cost calls retain theirs. Judge requests record an 8,192-token cap;
generation defaults to the protocol's 32,768-token cap, with a 65,536-token
cap for the GLM 5.3 main cohort via its per-model override. Each cohort uses a
consistent cap; the earlier 32,768-token GLM pilot is excluded from the main run.
Evaluation uses the protocol’s `judge_timeout` (420 seconds for this release),
falling back to `generation_timeout` if the judge setting is absent; the timeout
is part of the saved judgment request identity.
The ledger records its pricing
revision and catalog hash. These are estimates, not a reconciled proxy invoice.

The study's `report_leaderboard.json` points to the combined aggregate artifact.
`render_report` puts it above the existing judge-validation study;
`render_public_report` creates an aggregate-only standalone report with the
validation summary below it. Pending/blocked systems stay visible without
invented ranks. Source data, private calls, credentials and internal endpoints
must not be copied into public outputs. See [the report guide](../../docs/study-report.md)
and [the roadmap](../../docs/roadmap.md).
