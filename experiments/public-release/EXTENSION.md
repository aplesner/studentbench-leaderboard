# Extending the first release

`extend.py` creates a separate, resumable extension; it reads the first release
and preserves its protocol, plans, original budget ledger, and judgment cache.
It checks the original schedule and judge route, copies the original judgment
cache, and freezes a byte-level inventory of the original judgments. New calls
use the team proxy and a separate $1,000 estimated-cost ledger in the extension
directory. Catalog estimates are not reconciled proxy invoices.

Prepare a local JSON roster with explicit confirmed proxy model IDs and catalog
price IDs. `enabled` defaults to true. Disabled entries are retained in the
roster but are not run; select the complete active set before initializing the
extension because the roster is frozen on first use.

```json
{
  "models": [
    {
      "id": "example",
      "label": "Example model",
      "model": "confirmed/proxy-model-id",
      "price_id": "confirmed/catalog-model-id",
      "enabled": true,
      "generation_settings": {"max_tokens": 32768},
      "reasoning_effort": null
    }
  ]
}
```

Provide a frozen local catalog containing prices for every active model and the
original judge. A local connection TOML must resolve the original judge to the
same route as the first release; supply credentials securely through the
configured environment variables. The output directory must be outside the
first release, benchmark, and SSOT. `summary`, `generate`, and `evaluate` default
to planning without inference; initialization creates the extension manifest,
a seed ledger, and the cache copy.

```sh
uv run python experiments/public-release/extend.py generate \
  --config /path/to/extension-protocol.json \
  --catalog /path/to/frozen-catalog.json \
  --connection /path/to/connection.toml \
  --extension-dir /path/to/extension-output \
  --models example --max-new-calls 1
```

Add `--execute` to authorize the planned calls; it requires an explicit positive
`--max-new-calls`. Omit `--models` to generate all enabled models. Generate the
complete 83-student cohort for every active model before `evaluate`; joint
evaluation always includes all enabled new models and the five v1 candidates.
The preflight reports the sum of maximum reservations for the requested batch.
If that exceeds the remaining cap, use smaller batches; actual usage can release
unused reservations, while failures without usage retain them. Concurrent
processes share the same locked extension ledger.

With six new complete models, the planned cohort contains 498 generated plans
and 5,163 incremental pairs, in addition to 2,313 original pairs: 7,476 total.
With seven, those totals are 581 plans, 6,314 incremental pairs, and 8,627 total.
These counts assume the original 83 subjects and plan coverage. The new
leaderboard must be verified independently before any later publication; it
does not change the published first release.
