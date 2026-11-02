"""Offline extension cache/provenance verification; never executes an API call."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from unittest.mock import patch

from studentbench_leaderboard.extension import (
    DEFAULT_STUDY, _failed_attempt_billing, economic_cost, run_jobs, sha256,
)
from studentbench_leaderboard.study import call_path, write_json
from studentbench_leaderboard.extension_terminal import load_terminal_failures, terminal_failure_summary


def _load(path: Path):
    return json.loads(path.read_text())


def _key(job: dict) -> tuple[str, str, int]:
    return job["variant"], job["pair_id"], job.get("draw", 0)


def _no_api(*args, **kwargs):
    raise AssertionError("Verification is offline: an attempted API call is a verifier failure")


def _schedule(study: Path, name: str) -> list[dict]:
    jobs = _load(study / name)
    if not isinstance(jobs, list) or len({_key(job) for job in jobs}) != len(jobs):
        raise ValueError(f"Schedule must contain unique jobs: {name}")
    return jobs


def _exclusions(study: Path, variants: set[str]) -> dict:
    path = study / "availability.json"
    if not path.exists():
        return {}
    availability = _load(path)
    if availability.get("schema_version") != 1 or not isinstance(availability.get("exclusions"), dict):
        raise ValueError("Availability must declare schema_version=1 and an exclusions object")
    for name, entry in availability["exclusions"].items():
        if (name not in variants or not isinstance(entry, dict) or entry.get("status") != "unavailable" or
                not isinstance(entry.get("reason"), str) or not entry["reason"].strip()):
            raise ValueError("Every exclusion requires a scheduled variant and an explicit unavailable reason")
        evidence = (study / entry["evidence_file"]).resolve()
        if not evidence.is_relative_to(study.resolve()) or not evidence.is_file():
            raise ValueError("Exclusion evidence must be an existing file inside the extension")
        if sha256(evidence) != entry["evidence_sha256"]:
            raise ValueError("Exclusion evidence hash changed")
        recorded = _load(evidence)
        if recorded.get("variant") != name or recorded.get("status_code") != entry.get("http_status"):
            raise ValueError("Exclusion evidence does not match the declared variant/status")
    return availability["exclusions"]


def _money_summary(records: list[dict]) -> dict:
    costs = [economic_cost(record) for record in records]
    known = [row["economic_usd"] for row in costs if row["economic_usd"] is not None]
    return {"records": len(records), "known_economic_usd": sum(known),
            "economic_usd": sum(known) if len(known) == len(records) else None,
            "economic_cost_unknown_records": len(records) - len(known),
            "reported_router_charge_usd": sum(row["router_charge_usd"] for row in costs
                                               if row["router_charge_usd"] is not None),
            "reported_byok_upstream_usd": sum(row["upstream_inference_usd"] for row in costs
                                              if row["is_byok"] and row["upstream_inference_usd"] is not None),
            "byok_records": sum(row["is_byok"] is True for row in costs),
            "cost_bases": dict(Counter(row["basis"] for row in costs)),
            "estimated_cost_records": sum(row["is_estimate"] is True for row in costs)}


def _error_summary(study: Path, jobs: dict, completed: set, excluded: set, variants: dict,
                   providers: dict, catalog_sha: str, recovered_sources: dict, terminal: set) -> dict:
    entries, billable_records, incremental_records = [], [], []
    for path in sorted((study / "errors").glob("*.json")):
        error = _load(path)
        key = _key(error)
        if key not in jobs:
            raise ValueError(f"Error record does not belong to an explicit scheduled job: {path}")
        model = variants[key[0]]["model"]
        provider = providers.get(model.removeprefix("openrouter/")) if model.startswith("openrouter/") else None
        # Early preflight errors predate explicit billing summaries. Reconstruct
        # those summaries from retained usage without changing the raw error file.
        billing = _failed_attempt_billing(error, model, provider, catalog_sha)
        if "cost_provenance" in error:
            for field in ("usage", "estimated_usd", "cost_provenance"):
                if error.get(field) != billing[field]:
                    raise ValueError(f"Saved error billing disagrees with its returned usage: {path}")
        cost_record = {**error, **billing}
        billable_records.append(cost_record)
        relative = str(path.relative_to(study))
        recovered = relative in recovered_sources
        if recovered and recovered_sources[relative] != sha256(path):
            raise ValueError("Recovered billing source hash changed")
        if not recovered:
            incremental_records.append(cost_record)
        entries.append({"path": str(path.relative_to(study)), "sha256": sha256(path),
                        "variant": key[0], "pair_id": key[1], "draw": key[2],
                        "type": error.get("type"), "status_code": error.get("status_code"),
                        "resolved_by_completed_call": key in completed,
                        "declared_unavailable": key in excluded,
                        "declared_terminal_failure": key in terminal,
                        "counted_in_successful_recovery": recovered,
                        "billing": billing, "economic_cost": economic_cost(cost_record)})
    return {"attempts": len(entries),
            "resolved_attempts": sum(row["resolved_by_completed_call"] for row in entries),
            "declared_unavailable_attempts": sum(row["declared_unavailable"] for row in entries),
            "terminal_failure_attempts": sum(row["declared_terminal_failure"] for row in entries),
            "by_status": dict(Counter(str(row["status_code"]) for row in entries)),
            "by_type": dict(Counter(row["type"] for row in entries)),
            "recovered_attempts_counted_in_successes": sum(row["counted_in_successful_recovery"] for row in entries),
            "all_error_attempt_cost": _money_summary(billable_records),
            "cost": _money_summary(incremental_records), "records": entries}


def verify(study: Path = DEFAULT_STUDY) -> dict:
    """Require completed jobs or validated availability/terminal missing outcomes."""
    study = Path(study)
    provider_jobs = _schedule(study, "provider_jobs.json")
    example_jobs = _schedule(study, "example_jobs.json")
    union = {_key(job): {**job, "draw": job.get("draw", 0)} for job in provider_jobs + example_jobs}
    variants = {variant["variant"]: variant for variant in _load(study / "variants.json")}
    exclusions = _exclusions(study, {key[0] for key in union})
    with patch("studentbench_leaderboard.extension.openrouter_complete", side_effect=_no_api), \
            patch("studentbench_leaderboard.extension.LiteLLMJudge.complete", side_effect=_no_api):
        terminal = load_terminal_failures(study, list(union.values()))
    unavailable = {key for key in union if key[0] in exclusions}
    expected_missing = unavailable | set(terminal)
    present = {key for key, job in union.items() if call_path(study, job).is_file()}
    missing = set(union) - present
    if missing != expected_missing:
        raise ValueError(f"Missing jobs must equal explicit availability exclusions or verified terminal failures: "
                         f"unexpected_missing={len(missing - expected_missing)}, "
                         f"excluded_but_present={len(expected_missing - missing)}")
    allowed_paths = {call_path(study, job).resolve() for job in union.values()}
    unexpected = [path.name for path in (study / "calls").glob("*.json") if path.resolve() not in allowed_paths]
    if unexpected:
        raise ValueError(f"Unexpected call records outside the frozen schedules: {unexpected[:5]}")
    available_path = study / "provider_available_jobs.json"
    if available_path.exists():
        actual = {_key(job) for job in _schedule(study, available_path.name)}
        expected = {_key(job) for job in provider_jobs if job["variant"] not in exclusions}
        if actual != expected:
            raise ValueError("Available provider schedule differs from the explicit availability exclusion")
    # check_only already prevents dispatch. Patching both adapters additionally
    # makes an accidental API path fail before constructing a request/client.
    with patch("studentbench_leaderboard.extension.openrouter_complete", side_effect=_no_api), \
            patch("studentbench_leaderboard.extension.LiteLLMJudge.complete", side_effect=_no_api), \
            redirect_stdout(io.StringIO()):
        checked = run_jobs(study, list(union.values()), max_new_calls=len(union), check_only=True)
    if checked["pending"] != len(expected_missing) or checked["reused"] != len(present) or checked["api_attempts"]:
        raise ValueError("Offline cache validation count disagrees with the explicit schedule")
    imports = _load(study / "imported_calls.json")
    imported_names = set(imports["calls"])
    scheduled_names = {call_path(study, job).name for job in union.values()}
    if not imported_names <= scheduled_names:
        raise ValueError("Import ledger includes calls outside the scheduled union")
    providers = {entry["id"]: entry for entry in _load(study / "provider_catalog.json")["data"]}
    catalog_sha = sha256(study / "provider_catalog.json")
    inventory, imported_records, new_records, recovered_sources = {}, [], [], {}
    per_variant = defaultdict(list)
    for key in sorted(present):
        path = call_path(study, union[key])
        record = _load(path)
        imported = path.name in imported_names
        (imported_records if imported else new_records).append(record)
        per_variant[key[0]].append(record)
        if source := record.get("recovered_from_error"):
            if source["file"] in recovered_sources:
                raise ValueError("A saved error attempt was recovered more than once")
            recovered_sources[source["file"]] = source["sha256"]
        inventory[str(path.relative_to(study))] = {
            "sha256": sha256(path), "request_id": record["request_id"],
            "requested_request_id": record.get("requested_request_id"), "imported": imported,
            "created_at": record.get("created_at"), "response_model": record.get("response_model"),
            "response_provider": record.get("response_provider"),
            "requested_response_format": record.get("requested_response_format", "legacy-json-schema"),
            "effective_response_format": record.get("effective_response_format", "legacy-json-schema"),
            "parser_version": record.get("parser_version", "legacy-strict"),
            "parser_contract_sha256": record.get("parser_contract_sha256"),
            "provider_schema_compliant": record.get("provider_schema_compliant", True),
            "transformations": record.get("transformations", []),
            "recovered_from_error": record.get("recovered_from_error"),
            "economic_cost": economic_cost(record),
        }
    if len(imported_records) != len(imported_names):
        raise ValueError("An imported control is missing from the verified calls")
    variant_summary = []
    for name in sorted({key[0] for key in union}):
        records = per_variant[name]
        variant = variants[name]
        summary = {"variant": name, "requested_model": variant["model"],
                   "requested_reasoning_effort": variant.get("reasoning_effort"),
                   "scheduled": sum(key[0] == name for key in union), "completed": len(records),
                   "missing": sum(key[0] == name for key in missing),
                   "terminal_failures": sum(key[0] == name for key in terminal),
                   "exclusion": exclusions.get(name),
                   "response_models": dict(Counter(record.get("response_model", "not-reported") for record in records)),
                   "response_providers": dict(Counter(record.get("response_provider") or "not-reported" for record in records)),
                   "effective_response_formats": dict(Counter(record.get("effective_response_format", "legacy-json-schema")
                                                                for record in records)),
                   "compatibility_fallback_calls": sum(bool(record.get("transport", {}).get("compatibility_fallbacks"))
                                                       for record in records),
                   "parser_versions": dict(Counter(record.get("parser_version", "legacy-strict") for record in records)),
                   "adapted_serialization_calls": sum(bool(record.get("transformations")) for record in records),
                   "serialization_transformations": dict(Counter(value for record in records
                                                                   for value in record.get("transformations", []))),
                   "recovered_calls": sum("recovered_from_error" in record for record in records),
                   "cost": _money_summary(records)}
        variant_summary.append(summary)
    errors = _error_summary(study, union, present, unavailable, variants, providers, catalog_sha, recovered_sources, set(terminal))
    inputs = [study / name for name in ("manifest.json", "variants.json", "provider_catalog.json", "imported_calls.json",
              "availability.json", "provider_jobs.json", "provider_available_jobs.json", "example_jobs.json",
              "extension_inventory.json", "config_inventory.json", "report_metadata.json",
              "derived_judges.toml", "derived_judges_available.toml",
              "derived_judges_manifest.json", "derived_judges_available_manifest.json",
              "paired_deltas_protocol.json", "paired_deltas_phase_amendment.json",
              "paired_deltas_terminal_amendment.json", "execution_interruptions.json",
              "per_model_concurrency_amendment.json",
              "per_model_concurrency_activation.json", "deepseek_flash_resume_jobs.json",
              "planned_flash_restart.json",
              "terminal_failures.json", "original_preservation_check.json")]
    inputs.extend(path for path in (study / "configs").rglob("*") if path.is_file())
    input_inventory = {str(path.relative_to(study)): sha256(path) for path in sorted(set(inputs)) if path.is_file()}
    repository = Path(__file__).resolve().parents[2]
    sources = [path for directory in (repository / "src", repository / "tests", repository / "experiments/ranking-extension")
               for path in directory.rglob("*") if path.is_file() and path.suffix in (".py", ".toml", ".txt", ".json")
               and "__pycache__" not in path.parts]
    sources += [repository / "pyproject.toml", repository / "uv.lock", repository / "docs/extension-runner.md"]
    source_inventory = {str(path.relative_to(repository)): sha256(path) for path in sorted(set(sources)) if path.is_file()}
    known_new_error = errors["cost"]["known_economic_usd"]
    new_cost = _money_summary(new_records)
    status = ("passed-with-terminal-missing-outcomes" if terminal else
              "passed-with-declared-unavailable" if exclusions else "passed")
    result = {"schema_version": 1, "verified_at": datetime.now(timezone.utc).isoformat(),
              "study": str(study.resolve()), "status": status,
              "network_calls": 0, "cache_validation": checked,
              "counts": {"provider_schedule": len(provider_jobs), "example_schedule": len(example_jobs),
                         "unique_jobs": len(union), "completed_unique_calls": len(present),
                         "imported_calls": len(imported_records), "new_completed_calls": len(new_records),
                         "recovered_calls": len(recovered_sources),
                         "declared_unavailable_jobs": len(unavailable), "terminal_failed_jobs": len(terminal),
                         "missing_outcomes": len(missing)},
              "exclusions": exclusions, "terminal_failures": terminal_failure_summary(study, terminal),
              "variants": variant_summary,
              "cost": {"new_successful_calls": new_cost, "reused_controls": _money_summary(imported_records),
                       "failed_attempts": errors["cost"],
                       "known_new_calls_and_errors_economic_usd": new_cost["known_economic_usd"] + known_new_error,
                       "all_new_costs_complete": (new_cost["economic_cost_unknown_records"] == 0 and
                                                  errors["cost"]["economic_cost_unknown_records"] == 0),
                       "note": "Costs preserve returned billing and BYOK upstream separately; unknown failed-attempt cost is not zero. "
                               "Recovered source attempts are counted once in successful calls; duplicate paid retries remain additional costs. "
                               "Native/catalog token estimates are not invoice totals. Reused controls are not new extension spending."},
              "error_ledger": errors, "call_inventory": inventory, "input_inventory": input_inventory,
              "source_inventory": source_inventory,
              "limitations": ["This is an exposed validation extension, not an independent test.",
                              "The availability exclusion preserves the existing account guardrail; no model was substituted.",
                              "High reasoning effort is a provider-specific setting, not an equal compute budget.",
                              "Hashes describe local artifacts and do not constitute an immutable external audit log."]}
    write_json(study / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    result = verify(args.study)
    print(json.dumps({"status": result["status"], **result["counts"],
                      "known_new_calls_and_errors_economic_usd": result["cost"]["known_new_calls_and_errors_economic_usd"],
                      "cost_complete": result["cost"]["all_new_costs_complete"]}), flush=True)


if __name__ == "__main__":
    main()
