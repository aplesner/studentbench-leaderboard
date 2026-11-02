"""Refresh execution counts and billing in the report without fitting rankings."""
from datetime import datetime, timezone

from finish import (
    ORIGINAL, STUDY, costs, diagnostic_attempts, economic_total,
    interruption_costs, load, serialization, unique,
)
from studentbench_leaderboard.study import call_path, write_json
from studentbench_leaderboard.study_report import render_report
from studentbench_leaderboard.extension_terminal import load_terminal_failures, terminal_failure_summary


def main(study=STUDY):
    jobs = unique(load(study / "provider_jobs.json") + load(study / "example_jobs.json"))
    # Count and account for saved outputs without inspecting their verdicts.
    calls = [load(call_path(study, job)) for job in jobs if call_path(study, job).is_file()]
    imported = load(study / "imported_calls.json")["calls"]
    new = [row for row in calls if call_path(study, row).name not in imported]
    recovered = {row["recovered_from_error"]["file"] for row in calls if "recovered_from_error" in row}
    error_paths = list((study / "errors").glob("*.json"))
    errors = [load(path) for path in error_paths if str(path.relative_to(study)) not in recovered]
    diagnostics, reconciliation = diagnostic_attempts(study)
    interruption = interruption_costs(study)
    terminal = load_terminal_failures(study, jobs)
    exclusions = load(study / "availability.json")["exclusions"]
    variants = load(study / "variants.json")
    new_cost, failed_cost, diagnostic_cost = costs(new), costs(errors), costs(diagnostics)
    total = economic_total(new + errors + diagnostics, interruption)
    known = sum(row["known_economic_usd"] for row in (new_cost, failed_cost, diagnostic_cost, interruption))
    summary = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "validation_extension_after_original_test_exposure",
        "counts": {"scheduled_calls": len(jobs), "completed_calls": len(calls),
                   "new_completed_calls": len(new), "imported_controls": len(imported),
                   "unavailable_calls": sum(job["variant"] in exclusions for job in jobs),
                   "terminal_failed_calls": len(terminal),
                   "missing_outcomes": len(jobs) - len(calls), "failed_attempts": len(error_paths),
                   "recovered_error_attempts": len(recovered),
                   "actual_execution_interruptions": interruption["n_events"],
                   "n_interruption_events_with_unknown_billing": interruption["n_unknown_events"]},
        "availability": {variant["variant"]: {
            "status": "unavailable" if variant["variant"] in exclusions else "available",
            "completed_calls": sum(row["variant"] == variant["variant"] for row in calls),
            "terminal_failed_calls": sum(key[0] == variant["variant"] for key in terminal),
            **exclusions.get(variant["variant"], {})} for variant in variants},
        "terminal_failures": terminal_failure_summary(study, terminal),
        "estimated_usd": known,
        "costs": {"new_successful_calls": new_cost, "failed_attempts": failed_cost,
                  "routing_diagnostics": diagnostic_cost, "interrupted_in_flight_requests": interruption},
        "costs_total": total, "execution_interruptions": interruption,
        "routing_diagnostics": reconciliation,
        "costs_by_variant": {variant["variant"]: economic_total([
            row for row in calls if row["variant"] == variant["variant"]]) for variant in variants},
        "serialization_by_variant": {variant["variant"]: serialization([
            row for row in calls if row["variant"] == variant["variant"]]) for variant in variants},
        "cost_note": "Running snapshot of saved outputs. Known costs exclude unsaved in-flight requests, "
                     "unknown failed-attempt charges and unresolved interruption billing. Recovered outputs "
                     "and their source errors count once. BYOK upstream inference is included when reported. "
                     "Imported controls and derived panels add no new calls. No partial provider rankings are fitted.",
    }
    write_json(study / "summary.json", summary)
    render_report(ORIGINAL)
    print({"status": "running", **summary["counts"], "known_new_inference_usd": known})


if __name__ == "__main__":
    main()
