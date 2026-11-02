"""Analyze the provider/example extension and export reviewable results offline.

Requires every executable job to be cached or explicitly terminal after bounded
failed attempts. Availability and terminal outcomes remain distinct and missing.
This script never selects a replacement for the original held-out-test judge.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path

from studentbench_leaderboard.extension import economic_cost, run_jobs
from studentbench_leaderboard.extension_terminal import job_key, load_terminal_failures, terminal_failure_summary
from studentbench_leaderboard.study import call_path, write_json
from studentbench_leaderboard.study_analysis import analyze_round
from studentbench_leaderboard.study_report import render_report

STUDY = Path("results/ranking-extension-20260911")
ORIGINAL = Path("results/ranking-study-20260910")


def load(path):
    return json.loads(path.read_text())


def unique(jobs):
    return list({(j["variant"], j["pair_id"], j.get("draw", 0)): j for j in jobs}.values())


def costs(records):
    rows = [economic_cost(r) for r in records]
    known = [r["economic_usd"] for r in rows if r["economic_usd"] is not None]
    return {
        "known_economic_usd": sum(known), "n_records": len(rows),
        "n_unknown": len(rows) - len(known),
        "n_estimated": sum(r["is_estimate"] is True for r in rows),
        "router_charge_usd": sum(r["router_charge_usd"] or 0 for r in rows),
        "byok_upstream_inference_usd": sum(r["upstream_inference_usd"] or 0 for r in rows if r["is_byok"]),
        "bases": dict(Counter(r["basis"] for r in rows)),
    }


def economic_total(records, interruptions=None):
    rows = [economic_cost(r) for r in records]
    interruptions = interruptions or {"complete": True, "known_economic_usd": 0, "n_unknown_events": 0}
    complete = all(r["complete"] for r in rows) and interruptions["complete"]
    known = sum(r["economic_usd"] or 0 for r in rows) + interruptions["known_economic_usd"]
    bases = {r["basis"] for r in rows}
    if interruptions["n_unknown_events"]:
        bases.add("unresolved-interrupted-request-billing")
    return {"economic_usd": known if complete else None, "known_economic_usd": known,
            "router_charge_usd": sum(r["router_charge_usd"] or 0 for r in rows),
            "upstream_inference_usd": sum(r["upstream_inference_usd"] or 0 for r in rows if r["is_byok"]),
            "is_byok": any(r["is_byok"] for r in rows),
            "basis": "; ".join(sorted(bases)),
            "interruption_events_with_unknown_billing": interruptions["n_unknown_events"],
            "is_estimate": any(r["is_estimate"] is True for r in rows), "complete": complete}


def serialization(records):
    provider = [r for r in records if r.get("transport", {}).get("effective", {}).get("adapter") == "openrouter-openai-sdk"]
    return {"completed_calls": len(records),
            "provider_schema_compliant_calls": sum(r.get("provider_schema_compliant", True) for r in provider),
            "adapted_calls": sum(bool(r.get("transformations")) for r in provider),
            "recovered_from_errors": sum("recovered_from_error" in r for r in provider),
            "parser_contracts": dict(Counter(r.get("parser_version", "original-strict-parser") for r in records)),
            "transformations": dict(Counter(t for r in provider for t in r.get("transformations", [])))}


def diagnostic_attempts(study):
    """Count request/result pairs once; an unsaved response has unknown cost."""
    directory = study / "diagnostics"
    requests = {p.name.removesuffix(".request.json"): p for p in directory.glob("*_probe.request.json")}
    results = {p.stem: p for p in directory.glob("*_probe.json")}
    records = []
    for name in sorted(set(requests) | set(results)):
        if name in results:
            records.append(load(results[name]))
        else:
            # A request log alone proves neither a returned result nor zero
            # billing. Do not invent token usage or copy its parent-call cost.
            records.append({**load(requests[name]), "usage": {}, "estimated_usd": None,
                            "cost_provenance": {"source": "unavailable", "is_estimate": None},
                            "diagnostic_execution_status": "request_without_saved_result"})
    return records, {
        "recorded_requests": len(requests), "saved_results": len(results), "attempts": len(records),
        "requests_without_saved_results": [str(requests[name].relative_to(study))
                                           for name in sorted(requests.keys() - results.keys())],
        "results_without_request_logs": [str(results[name].relative_to(study))
                                         for name in sorted(results.keys() - requests.keys())],
        "interpretation": "Routing diagnostics are excluded from ranking. A request without a saved result has unknown billing and is counted once as an unresolved diagnostic attempt, not as a zero-cost call.",
    }


def interruption_costs(study):
    """Keep actual stop events separate from unobserved request/charge counts."""
    path = study / "execution_interruptions.json"
    ledger = load(path) if path.exists() else {}
    entries = ledger.get("interruptions", [ledger] if "signal_sent" in ledger else [])
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Execution interruption ledger must contain event objects")
    events, known, unknown = [], 0.0, 0
    fields = ("confirmed_interrupted_api_request_count", "local_api_calls_lower_bound",
              "local_api_calls_upper_bound", "local_api_calls_upper_bound_by_model",
              "charges_usd", "billing_reconciliation_status")
    for entry in entries:
        if entry.get("signal_sent") is not True:
            continue  # Proposed or rejected actions create no interruption cost.
        inflight = entry.get("in_flight") or {}
        charge = inflight.get("charges_usd")
        reconciled = (inflight.get("billing_reconciliation_status") in {"resolved", "reconciled"}
                      and type(charge) in (int, float) and isfinite(charge) and charge >= 0)
        known += float(charge) if reconciled else 0
        unknown += not reconciled
        events.append({**{key: entry.get(key) for key in (
            "interruption_id", "target_pid", "target_run_id", "signal_sent_at", "status", "variant_ids", "model_ids")},
            "signal_sent": True, "in_flight": {key: inflight.get(key) for key in fields}})

    def total_count(field):
        values = [event["in_flight"].get(field) for event in events]
        return sum(values) if all(type(value) is int and value >= 0 for value in values) else None

    by_model = Counter()
    bounds_complete = True
    for event in events:
        bounds = event["in_flight"].get("local_api_calls_upper_bound_by_model")
        if not isinstance(bounds, dict) or any(type(value) is not int or value < 0 for value in bounds.values()):
            bounds_complete = False
        else:
            by_model.update(bounds)
    return {"n_events": len(events), "n_unknown_events": unknown,
            "ignored_unsent_events": len(entries) - len(events),
            "known_economic_usd": known, "economic_usd": known if not unknown else None,
            "complete": not unknown,
            "confirmed_interrupted_api_request_count": total_count("confirmed_interrupted_api_request_count"),
            "local_api_calls_lower_bound": total_count("local_api_calls_lower_bound"),
            "local_api_calls_upper_bound": total_count("local_api_calls_upper_bound"),
            "local_api_calls_upper_bound_by_model": dict(by_model) if bounds_complete else None,
            "events": events, "file": path.name if path.exists() else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "interpretation": "Only verified signal_sent=true events count. The old runner did not persist every unfinished dispatch, so request identities/counts and additional billing can remain unknown. Bounds describe concurrent local operations, not confirmed API dispatches, queued jobs, or all upstream work. Saved successes/errors are accounted separately; unresolved interruption charges are excluded from the recorded/estimated subtotal and prevent a complete total."}


def artifact_hashes(study):
    """Hash final outputs and their local protocol records after all writes."""
    names = ("paired_deltas.json", "example_audit.json", "example_audit.md", "example.json",
             "example_provenance.json", "derived_judges.toml", "derived_judges_manifest.json",
             "derived_judges_available.toml", "derived_judges_available_manifest.json",
             "paired_deltas_protocol.json", "paired_deltas_phase_amendment.json",
             "paired_deltas_terminal_amendment.json", "terminal_failures.json",
             "availability.json", "verification.json", "execution_interruptions.json",
             "per_model_concurrency_amendment.json",
             "per_model_concurrency_activation.json", "deepseek_flash_resume_jobs.json",
             "planned_flash_restart.json",
             "original_preservation_check.json", "source_snapshot.json", "source_snapshot.zip",
             "manifest.json", "variants.json", "provider_jobs.json", "example_jobs.json",
             "provider_available_jobs.json", "config_inventory.json", "extension_inventory.json",
             "report_metadata.json", "provider_catalog.json", "imported_calls.json")
    paths = set((study / "exports").glob("*.csv")) | set(study.glob("*_analysis.json"))
    paths |= set((study / "diagnostics").glob("*.json"))
    paths |= {study / name for name in names if (study / name).is_file()}
    return {str(path.relative_to(study)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def export_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def enrich_and_export(study, analysis, calls, terminal=None):
    terminal = terminal or {}
    availability = study / "availability.json"
    unavailable_variants = set(load(availability).get("exclusions", {})) if availability.exists() else set()
    name = analysis["name"]
    analysis["evaluation_role"] = "validation_extension_after_original_test_exposure"
    analysis["interpretation"] += (
        " This extension reuses the original validation cohort after the original test was inspected."
        " It is exploratory validation, not fresh held-out confirmation."
        " Costs include reported BYOK upstream inference where available; imported controls are not new spend."
    )
    comparisons, leaderboards = [], []
    for source, fit in (("human_scheduled", analysis["human"]), ("human_full_validation", analysis["human_full"])):
        leaderboards.extend({"source": source, **r} for r in fit["leaderboard"])
    for judge in analysis["judges"]:
        members = {tuple(m) for m in judge["members"]}
        records = [r for r in calls if (r["variant"], r.get("draw", 0)) in members]
        cost = costs(records)
        judge["cached_charge_or_estimate_usd"] = judge["estimated_usd"]
        judge["estimated_usd"] = cost["known_economic_usd"] if not cost["n_unknown"] else None
        judge["cost_breakdown"] = cost
        judge["cost_note"] = "Successful underlying calls only; panels reuse member calls and do not incur extra API calls."
        judge["terminal_failed_calls"] = sum((key[0], key[2]) in members for key in terminal)
        judge["unavailable_member_variants"] = sorted({variant for variant, _draw in members} & unavailable_variants)
        missing_calls = judge["coverage"]["missing_calls"]
        if judge["unavailable_member_variants"]:
            # Every soft-panel member is required. One unavailable member
            # prevents all aggregate criteria, regardless of other failures.
            judge["execution_status"] = "unavailable"
        elif missing_calls:
            if judge["terminal_failed_calls"] == missing_calls:
                judge["execution_status"] = "completed_with_terminal_failures" if judge["coverage"]["n_calls"] else "terminal_failure"
            else:
                judge["execution_status"] = "incomplete"
        else:
            judge["execution_status"] = "completed"
        comparisons.append({"variant": judge["variant"], "label": judge["label"], **judge["metrics"],
                            "criterion_coverage": judge["coverage"]["criterion_coverage"],
                            "missing_calls": judge["coverage"]["missing_calls"],
                            "terminal_failed_calls": judge["terminal_failed_calls"],
                            "execution_status": judge["execution_status"],
                            "unavailable_member_variants": "; ".join(judge["unavailable_member_variants"]),
                            "successful_inference_usd": judge["estimated_usd"]})
        intervals = (judge.get("bootstrap") or {}).get("rank_intervals", {}).get("judge", {})
        for row in judge["fit"]["leaderboard"]:
            interval = intervals.get(row["model"], {})
            leaderboards.append({"source": judge["variant"], **row,
                                 "rank_interval_low": interval.get("low"),
                                 "rank_interval_high": interval.get("high")})
    write_json(study / f"{name}_analysis.json", analysis)
    export_csv(study / "exports" / f"{name}_judges.csv", comparisons,
               ["variant", "label", "kendall_tau_b", "spearman", "pair_order_agreement", "rank_mae",
                "probability_mae", "criterion_coverage", "missing_calls", "terminal_failed_calls",
                "execution_status", "unavailable_member_variants", "successful_inference_usd"])
    export_csv(study / "exports" / f"{name}_model_leaderboards.csv", leaderboards,
               ["source", "model", "rank", "score", "n_comparisons", "n_subjects", "rank_interval_low", "rank_interval_high"])
    return analysis


def execution_completion(study, jobs):
    """Validate completion statuses without removing jobs or filling in ratings."""
    availability = study / "availability.json"
    excluded = load(availability)["exclusions"] if availability.exists() else {}
    terminal = load_terminal_failures(study, jobs)
    missing = [job for job in jobs if not call_path(study, job).is_file()]
    unavailable = {job_key(job) for job in jobs if job["variant"] in excluded}
    if {job_key(job) for job in missing} != unavailable | set(terminal):
        raise ValueError(f"Incomplete executable schedule: {len(missing)} missing, "
                         f"{len(unavailable)} unavailable, {len(terminal)} verified terminal failures")
    return excluded, terminal, missing, unavailable


def main(study=STUDY, bootstrap=500):
    provider_jobs, example_jobs = load(study / "provider_jobs.json"), load(study / "example_jobs.json")
    jobs = unique(provider_jobs + example_jobs)
    excluded, terminal, missing, unavailable = execution_completion(study, jobs)
    # This reparses every saved response and reconstructs inputs without dispatch.
    check = run_jobs(study, jobs, max_new_calls=len(missing), check_only=True)
    if check["reused"] != len(jobs) - len(missing) or check["pending"] != len(missing):
        raise ValueError("Cache verification did not match the explicit schedule")
    calls = [load(call_path(study, j)) for j in jobs if call_path(study, j).is_file()]
    analyses = []
    for name, schedule in (("provider_validation", "provider_jobs.json"), ("example_validation", "example_jobs.json")):
        print(json.dumps({"event": "analyzing", "round": name, "bootstrap": bootstrap}), flush=True)
        analysis = analyze_round(study, name, study / schedule, bootstrap=bootstrap, density=True)
        members = {(j["variant"], j["pair_id"], j.get("draw", 0)) for j in load(study / schedule)}
        scoped = [r for r in calls if (r["variant"], r["pair_id"], r.get("draw", 0)) in members]
        analyses.append(enrich_and_export(study, analysis, scoped,
                        {key: value for key, value in terminal.items() if key in members}))
    imported = load(study / "imported_calls.json")["calls"]
    new = [r for r in calls if call_path(study, r).name not in imported]
    error_files = list((study / "errors").glob("*.json"))
    errors = [load(p) for p in error_files]
    recovered = {r["recovered_from_error"]["file"] for r in calls if "recovered_from_error" in r}
    # A recovered success and its preserved source error describe one paid attempt.
    extra_errors = [load(p) for p in error_files if str(p.relative_to(study)) not in recovered]
    diagnostic_records, diagnostic_reconciliation = diagnostic_attempts(study)
    availability = {}
    for v in load(study / "variants.json"):
        name = v["variant"]
        records = [r for r in calls if r["variant"] == name]
        availability[name] = {
            "status": "unavailable" if name in excluded else "available",
            "completed_calls": len(records), **excluded.get(name, {}),
            "terminal_failed_calls": sum(key[0] == name for key in terminal),
            "response_formats": dict(Counter(r.get("effective_response_format", "original-schema") for r in records)),
            "response_providers": dict(Counter(r.get("response_provider") or "native OpenAI" for r in records)),
        }
    new_cost, failed_cost, diagnostic_cost = costs(new), costs(extra_errors), costs(diagnostic_records)
    interrupted = interruption_costs(study)
    summary = {
        "status": ("complete_with_terminal_missing_outcomes" if terminal else
                   "complete_with_provider_unavailable" if excluded else "complete"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "validation_extension_after_original_test_exposure",
        "availability": availability,
        "terminal_failures": terminal_failure_summary(study, terminal),
        "counts": {"scheduled_calls": len(jobs), "completed_calls": len(calls),
                   "new_completed_calls": len(new), "imported_controls": len(imported),
                   "unavailable_calls": len(unavailable), "terminal_failed_calls": len(terminal),
                   "missing_outcomes": len(missing), "failed_attempts": len(errors),
                   "recovered_error_attempts": len(recovered),
                   "routing_diagnostic_attempts": len(diagnostic_records),
                   "routing_diagnostic_unsaved_results": len(diagnostic_reconciliation["requests_without_saved_results"]),
                   "actual_execution_interruptions": interrupted["n_events"],
                   "n_interruption_events_with_unknown_billing": interrupted["n_unknown_events"]},
        "estimated_usd": new_cost["known_economic_usd"] + failed_cost["known_economic_usd"] + diagnostic_cost["known_economic_usd"] + interrupted["known_economic_usd"],
        "costs": {"new_successful_calls": new_cost, "failed_attempts": failed_cost,
                  "routing_diagnostics": diagnostic_cost,
                  "interrupted_in_flight_requests": interrupted,
                  "imported_controls": costs([r for r in calls if call_path(study, r).name in imported])},
        "costs_by_variant": {v["variant"]: economic_total([r for r in calls if r["variant"] == v["variant"]])
                             for v in load(study / "variants.json")},
        "costs_total": economic_total(new + extra_errors + diagnostic_records, interrupted),
        "execution_interruptions": interrupted,
        "routing_diagnostics": diagnostic_reconciliation,
        "serialization_by_variant": {v["variant"]: serialization([r for r in calls if r["variant"] == v["variant"]])
                                     for v in load(study / "variants.json")},
        "cost_note": "Recorded/estimated subtotal from new successful calls, saved failed calls, and diagnostics, excluding unknown charges; recovered successes and their source error are counted once. Unknown failed-attempt charges are excluded. Routing diagnostic requests without saved responses remain unknown-cost attempts. Only actual signal_sent=true interruptions can add unresolved in-flight billing; rejected proposals do not. In-flight bounds are not confirmed dispatched requests, and unresolved charges prevent a complete total. Native costs are token estimates. BYOK includes reported upstream inference plus router charges; a zero router fee does not mean free inference. Imported controls and derived panels add no new calls.",
        "verification": {"status": "request_reconstruction_passed", "file": "verification.json"},
        "original_selection_unchanged": load(study / "manifest.json")["original_selection_sha256"] == load(ORIGINAL / "manifest.json")["selection_sha256"],
    }
    from studentbench_leaderboard.paired_deltas import run_paired_deltas
    run_paired_deltas(study, n_bootstrap=bootstrap)
    summary["artifact_sha256"] = artifact_hashes(study)
    write_json(study / "summary.json", summary)
    render_report(ORIGINAL)
    print(json.dumps({"event": "complete", **summary["counts"], "known_new_inference_usd": summary["estimated_usd"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    main(args.study, args.bootstrap)
