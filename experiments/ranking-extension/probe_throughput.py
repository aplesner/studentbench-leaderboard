"""One isolated routing diagnostic; never used to select verdicts or rank judges."""
import copy
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from studentbench_leaderboard.extension import openrouter_complete, economic_cost
from studentbench_leaderboard.study import write_json

root = Path("results/ranking-extension-20260911")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fireworks-only", action="store_true")
args = parser.parse_args()
profile = "fireworks" if args.fireworks_only else "throughput"
output = root / f"diagnostics/deepseek_flash_{profile}_probe.json"
if output.exists():
    raise SystemExit("The diagnostic is already recorded; no repeat call made.")
first = json.loads((root / "manifest.json").read_text())["round2_validation_pairs"][0]
source = json.loads((root / "calls" / f"or_deepseek41_flash_high--{first}--0.json").read_text())
contract = copy.deepcopy(source["transport"]["requested"])
if args.fireworks_only:
    contract["body"]["provider"].update(only=["fireworks"], allow_fallbacks=False)
else:
    contract["body"]["provider"]["sort"] = "throughput"
catalog = json.loads((root / "provider_catalog.json").read_text())["data"]
model = next(m for m in catalog if m["id"] == contract["body"]["model"])
record = {"purpose": "Latency diagnostic only; excludes rankings and performance selection",
          "source_pair_id": first, "transport": contract,
          "created_at": datetime.now(timezone.utc).isoformat()}
write_json(output.with_suffix(".request.json"), record)
started = time.monotonic()
try:
    raw, metadata = openrouter_complete(contract, model)
    record.update(raw_response=raw, **metadata)
except Exception as exc:
    record.update(error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None))
record["duration_seconds"] = time.monotonic() - started
record["cost"] = economic_cost(record)
write_json(output, record)
print(json.dumps({k: record.get(k) for k in ("duration_seconds", "response_provider", "error_type", "cost")}))
