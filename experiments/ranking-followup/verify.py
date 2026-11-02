"""Verify follow-up caches offline, preserving each reused provider's catalog."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from studentbench_leaderboard import extension
from studentbench_leaderboard.data import digest
from studentbench_leaderboard.study import call_path, write_json
from studentbench_leaderboard.study_analysis import load_derived_judges

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("results/ranking-followup-20260911")
ORIGINAL = ROOT / "results/ranking-study-20260910"
QWEN = "qwen38_max_high"
REUSE = ("enhanced_terra_high", "or_deepseek41_flash_high", "or_glm53_flash_high", "or_kimi3_high")
PHASE = "focused_validation"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BASE = module(ROOT / "experiments/ranking-extension/verify.py", "followup_extension_verifier")


def load(path):
    return json.loads(Path(path).read_text())


def key(row):
    return row["variant"], row["pair_id"], row.get("draw", 0)


def prepared_inputs(study):
    inventory = load(study / "prepared_inputs.json")
    required = {"manifest.json", "variants.json", "provider_catalog.json", "reused_calls.json",
                "focused_jobs.json", "qwen_jobs.json", "derived_judges.toml",
                "derived_judges_manifest.json", "imported_calls.json", "config_inventory.json"}
    if not isinstance(inventory, dict) or not required <= inventory.keys():
        raise ValueError("Prepared input inventory is missing required frozen files")
    for name, expected in inventory.items():
        path = (study / name).resolve()
        if not path.is_relative_to(study.resolve()) or not path.is_file() or extension.sha256(path) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    return inventory


def schedules(study, manifest):
    focused = BASE._schedule(study, "focused_jobs.json")
    qwen = BASE._schedule(study, "qwen_jobs.json")
    pairs = manifest["round2_validation_pairs"]
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("Follow-up validation pair IDs must be unique and nonempty")
    expected = {(variant, pair, 0) for variant in (*REUSE, QWEN) for pair in pairs}
    if {key(job) for job in focused} != expected:
        raise ValueError("Focused schedule must retain every pair for all five judges")
    if {key(job) for job in qwen} != {job for job in expected if job[0] == QWEN}:
        raise ValueError("Qwen schedule differs from the focused schedule")
    return focused, qwen


def validate_reuse(study, jobs, manifest):
    ledger = load(study / "reused_calls.json")
    source = Path(ledger["source_study"]).resolve()
    if (extension.sha256(source / "provider_catalog.json") != ledger["source_catalog_sha256"] or
            extension.sha256(source / "manifest.json") != manifest["previous_manifest_sha256"]):
        raise ValueError("Reused source catalog or manifest changed")
    expected = {call_path(study, job).name: job for job in jobs if job["variant"] in REUSE}
    if set(ledger["calls"]) != set(expected):
        raise ValueError("Reuse ledger must contain exactly the four copied judge schedules")
    copied = []
    for filename, job in expected.items():
        entry = ledger["calls"][filename]
        original = Path(entry["source"]).resolve()
        target = study / "calls" / filename
        if original != (source / "calls" / filename).resolve():
            raise ValueError("Reused call source differs from its declared source study")
        if (key(entry) != key(job) or not original.is_file() or not target.is_file() or
                extension.sha256(original) != entry["sha256"] or extension.sha256(target) != entry["sha256"]):
            raise ValueError(f"Reused source/copy byte hash changed: {filename}")
        record = load(target)
        if key(record) != key(job) or record["request_id"] != entry["request_id"]:
            raise ValueError(f"Reused call identity changed: {filename}")
        copied.append(record)
    # Recovered calls refer to source-study errors; catalog pricing can also
    # differ. Never send these jobs to the new study's request validator.
    with patch.object(extension, "openrouter_complete", side_effect=BASE._no_api), \
            patch.object(extension.LiteLLMJudge, "complete", side_effect=BASE._no_api), \
            redirect_stdout(io.StringIO()):
        checked = extension.run_jobs(source, list(expected.values()), check_only=True, max_new_calls=0)
    if checked["reused"] != len(expected) or checked["pending"] or checked["api_attempts"]:
        raise ValueError("Reused source cache validation did not cover the complete copied schedule")
    return copied, ledger, checked


def output_checks(study, counts, jobs, variants):
    analysis = load(study / f"{PHASE}_analysis.json")
    summary = load(study / "summary.json")
    paired = load(study / "paired_deltas.json")
    definitions = load_derived_judges(study, variants)
    expected_judges = set(variants) | {d["variant"] for d in definitions}
    actual = {j["variant"]: j for j in analysis["judges"]}
    if (analysis["jobs_sha256"] != digest(jobs) or analysis["n_expected_calls"] != counts["scheduled_calls"] or
            analysis["n_completed_calls"] != counts["completed_calls"] or set(actual) != expected_judges):
        raise ValueError("Final analysis count, schedule or judge set differs from verified inputs")
    for field in ("scheduled_calls", "completed_calls", "new_completed_calls", "imported_controls",
                  "unavailable_calls", "missing_outcomes"):
        if summary["counts"][field] != counts[field]:
            raise ValueError(f"Final summary count differs: {field}")
    if set(paired["comparisons"]) != expected_judges - {"enhanced_terra_high"}:
        raise ValueError("Paired contrasts must retain every candidate versus Terra")
    if (paired["main_analysis_sha256"] != {PHASE: extension.sha256(study / f"{PHASE}_analysis.json")} or
            paired["jobs_sha256"] != {PHASE: digest(jobs)} or paired["pending_phases"]):
        raise ValueError("Paired contrasts differ from the final analysis")
    for name, comparison in paired["comparisons"].items():
        if comparison["coverage"]["candidate_pairs"] != actual[name]["coverage"]["scored_pairs"]:
            raise ValueError("Paired contrast coverage differs from final analysis")
    paths = [study / f"{PHASE}_analysis.json", study / "paired_deltas.json",
             *sorted((study / "exports").glob("*.csv"))]
    return {str(path.relative_to(study)): extension.sha256(path) for path in paths}


def verify(study=STUDY, *, require_outputs=True, write=True):
    study = Path(study)
    inputs = prepared_inputs(study)
    manifest = load(study / "manifest.json")
    if digest(load(ORIGINAL / "selection.json")) != manifest["original_selection_sha256"]:
        raise ValueError("Original held-out-test judge selection changed")
    jobs, qwen_jobs = schedules(study, manifest)
    variants_list = load(study / "variants.json")
    variants = {v["variant"]: v for v in variants_list}
    if len(variants_list) != 5 or set(variants) != {*REUSE, QWEN}:
        raise ValueError("Follow-up variants differ from the fixed five-member schedule")
    definitions = load_derived_judges(study, variants)
    if len(definitions) != 3:
        raise ValueError("Follow-up requires exactly three frozen panel recipes")
    for definition in definitions:
        families = [member["family"] for member in definition["weighted_members"]]
        if len(families) != len(set(families)):
            raise ValueError("Follow-up panels require distinct provider families")
    exclusions = BASE._exclusions(study, set(variants))
    if set(exclusions) - {QWEN}:
        raise ValueError("Copied complete controls cannot be excluded")
    if (study / "terminal_failures.json").exists():
        raise ValueError("Individual terminal outcomes were not declared for this follow-up")
    expected_paths = {call_path(study, job).resolve() for job in jobs}
    if any(path.resolve() not in expected_paths for path in (study / "calls").glob("*.json")):
        raise ValueError("Unexpected call record outside the focused schedule")
    present = {key(job) for job in jobs if call_path(study, job).is_file()}
    unavailable = {key(job) for job in jobs if job["variant"] in exclusions}
    if {key(job) for job in jobs} - present != unavailable:
        raise ValueError("Missing outcomes must equal the explicit availability exclusion")
    reused, ledger, previous_check = validate_reuse(study, jobs, manifest)
    with patch.object(extension, "openrouter_complete", side_effect=BASE._no_api), \
            patch.object(extension.LiteLLMJudge, "complete", side_effect=BASE._no_api), \
            redirect_stdout(io.StringIO()):
        qwen_check = extension.run_jobs(study, qwen_jobs, check_only=True, max_new_calls=len(unavailable))
        # Missing jobs also provide the exact original request for validating
        # availability/error provenance; no calls are dispatched by _prepare.
        pending, _, providers, catalog_sha = extension._prepare(study, qwen_jobs, 420)
    new = [load(call_path(study, job)) for job in qwen_jobs if key(job) in present]
    if (qwen_check["reused"] != len(new) or qwen_check["pending"] != len(unavailable) or
            qwen_check["api_attempts"]):
        raise ValueError("New Qwen cache validation disagrees with its explicit schedule")
    requested = {key(p.job): p for p in pending}
    cached_new = {key(record): record for record in new}
    for path in (study / "errors").glob("*.json"):
        error = load(path)
        if key(error) not in {key(job) for job in qwen_jobs}:
            raise ValueError("Only newly scheduled Qwen attempts belong in follow-up errors")
        if key(error) in requested:
            p = requested[key(error)]
            task, judge, variant = p.call_task, p.judge, p.variant
        else:
            record = cached_new[key(error)]
            task = SimpleNamespace(messages=lambda: record["requested_messages"],
                                   schema=lambda: record["response_schema"])
            judge, variant = record["judge"], variants[QWEN]
        contract = extension.request_contract(task, variant,
                                              response_format=error["transport"]["response_format"])
        if (error["transport"] != contract or
                error["request_id"] != extension._request_id(contract, judge, key(error)[2])):
            raise ValueError("Qwen error request differs from the frozen uncapped request")
    recovered = {r["recovered_from_error"]["file"]: r["recovered_from_error"]["sha256"]
                 for r in new if "recovered_from_error" in r}
    errors = BASE._error_summary(study, {key(j): j for j in qwen_jobs}, present, unavailable,
                                variants, providers, catalog_sha, recovered, set())
    execution_path = study / "execution.json"
    execution = load(execution_path) if execution_path.exists() else None
    if execution and execution.get("status") == "preflight_unavailable_no_bulk_dispatch":
        if (new or set(exclusions) != {QWEN} or errors["attempts"] != 1 or
                execution.get("new_api_attempts") != 1 or execution.get("new_successes") != 0 or
                execution.get("bulk_requests_dispatched") != 0 or
                load(study / errors["records"][0]["path"]).get("run_id") != execution.get("preflight_run_id")):
            raise ValueError("Preflight-only execution ledger disagrees with saved calls/errors")
    counts = {"scheduled_calls": len(jobs), "completed_calls": len(present),
              "new_completed_calls": len(new), "imported_controls": len(reused),
              "unavailable_calls": len(unavailable), "terminal_failed_calls": 0,
              "missing_outcomes": len(unavailable), "failed_attempts": errors["attempts"],
              "recovered_error_attempts": len(recovered)}
    extra_inputs = ("prepared_inputs.json", "availability.json", "execution.json", "extension_inventory.json",
                    "source_snapshot.json", "source_snapshot.zip", "report_revision_artifacts.json")
    inputs.update({name: extension.sha256(study / name) for name in extra_inputs if (study / name).exists()})
    source_files = set((ROOT / "src").rglob("*.py")) | set((ROOT / "tests").glob("*.py"))
    source_files |= set((ROOT / "experiments/ranking-followup").rglob("*"))
    source_files |= {ROOT / "experiments/ranking-extension/finish.py", ROOT / "experiments/ranking-extension/verify.py",
                     ROOT / "docs/ranking-followup.md", ROOT / "pyproject.toml", ROOT / "uv.lock"}
    source_files = {p for p in source_files if p.is_file() and "__pycache__" not in p.parts}
    call_inventory = {}
    for row in reused + new:
        path = call_path(study, row)
        call_inventory[path.name] = {"sha256": extension.sha256(path), "request_id": row["request_id"],
                                     "reused": path.name in ledger["calls"], "variant": row["variant"],
                                     "response_model": row.get("response_model"), "response_provider": row.get("response_provider"),
                                     "effective_response_format": row.get("effective_response_format", "original-schema")}
    new_cost, reused_cost = BASE._money_summary(new), BASE._money_summary(reused)
    result = {"schema_version": 1, "verified_at": datetime.now(timezone.utc).isoformat(),
              "status": "passed-with-declared-unavailable" if exclusions else "passed",
              "network_calls": 0, "counts": counts, "exclusions": exclusions,
              "cache_validation": {"reused_source_study": previous_check, "new_qwen_study": qwen_check},
              "reused_source_study": ledger["source_study"],
              "reused_source_catalog_sha256": ledger["source_catalog_sha256"],
              "original_selection_unchanged": True,
              "execution": execution,
              "input_inventory": inputs, "call_inventory": call_inventory,
              "source_inventory": {str(p.relative_to(ROOT)): extension.sha256(p) for p in sorted(source_files)},
              "error_ledger": errors,
              "cost": {"new_successful_calls": new_cost, "reused_controls": reused_cost,
                       "failed_attempts": errors["cost"],
                       "known_new_calls_and_errors_economic_usd": new_cost["known_economic_usd"] + errors["cost"]["known_economic_usd"],
                       "all_new_costs_complete": not (new_cost["economic_cost_unknown_records"] or errors["cost"]["economic_cost_unknown_records"])},
              "limitations": "Copied provider calls retain source catalog, request, recovery and billing provenance; they are validated only in that source study. Reused calls and derived panels add no new follow-up spending. Missing provider results never become ties. Unknown failed-attempt charges are not zero."}
    if require_outputs:
        result["output_inventory"] = output_checks(study, counts, jobs, variants)
    if write:
        write_json(study / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    result = verify(args.study)
    print(json.dumps({"status": result["status"], **result["counts"], "network_calls": 0}))


if __name__ == "__main__":
    main()
