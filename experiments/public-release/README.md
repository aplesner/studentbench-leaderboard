# First public leaderboard run

The November 2, 2026 release adds five enabled generator systems to the 12 frozen
GRE Quant baselines. The protocol uses 83 student contexts, 415 new plans and up
to 2,313 comparisons, including new-to-new pairs. GPT-6.1 Terra and MiMo v2.6
Flash/Pro are explicitly deferred at the user's request while their proxy aliases
are unavailable; their initial access checks remain in the audit history.

**All our authoring and judging calls use the existing Tailscale LiteLLM proxy.**
This release-specific choice does not change the public CLI's automatic
OpenRouter/native-key routing. The runner rejects non-Tailscale profiles.
The total authorized inference estimate cap is $1,000.

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

Every dispatch reserves a conservative catalog-based amount in a shared locked
ledger. The input rate includes the highest prompt/cache-write rate across
catalog tiers. Completed calls with usage release unused reservations; failed
or unknown-cost calls retain theirs. Judge requests record an 8,192-token cap;
generation uses the protocol's 32,768-token cap. The ledger records its pricing
revision and catalog hash. These are estimates, not a reconciled proxy invoice.

The study's `report_leaderboard.json` points to the combined aggregate artifact.
`render_report` puts it above the existing judge-validation study;
`render_public_report` creates an aggregate-only standalone report with the
validation summary below it. Pending/blocked systems stay visible without
invented ranks. Source data, private calls, credentials and internal endpoints
must not be copied into public outputs. See [the report guide](../../docs/study-report.md)
and [the roadmap](../../docs/roadmap.md).
