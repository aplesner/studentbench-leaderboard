"""Analyze the frozen follow-up offline; never execute calls or render old reports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

from studentbench_leaderboard.data import Dataset
from studentbench_leaderboard.paired_deltas import BASELINE, _round_rows, paired_bootstrap_deltas
from studentbench_leaderboard.study import call_path, write_json
from studentbench_leaderboard.study_analysis import analyze_round


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ROOT = Path(__file__).resolve().parents[2]
VERIFY = module(Path(__file__).with_name("verify.py"), "followup_verifier_for_finish")
BASE = module(ROOT / "experiments/ranking-extension/finish.py", "followup_extension_finish_helpers")
STUDY, PHASE, QWEN = VERIFY.STUDY, VERIFY.PHASE, VERIFY.QWEN


def artifact_hashes(study):
    paths = {study / name for name in VERIFY.load(study / "prepared_inputs.json")}
    paths |= {study / name for name in ("prepared_inputs.json", "availability.json", "execution.json", "extension_inventory.json",
                                       "verification.json", "paired_deltas.json", "source_snapshot.json", "source_snapshot.zip",
                                       "report_revision_artifacts.json")}
    paths |= set(study.glob("*_analysis.json")) | set((study / "exports").glob("*.csv"))
    return {str(path.relative_to(study)): VERIFY.extension.sha256(path)
            for path in sorted(paths) if path.is_file()}


def enrich_and_export(study, analysis, records, exclusions):
    analysis["evaluation_role"] = "exploratory_followup_on_reused_validation"
    analysis["interpretation"] += (
        " Panel members were chosen after examining earlier validation results."
        " This is exploratory reuse of the same validation cohort, not fresh held-out confirmation."
        " Qwen uses provider-default output limits; copied controls retain their historical request budgets."
        " Costs attached to judges describe underlying successful calls; reused calls and panel aggregation add no new spending."
    )
    boards, comparisons = [], []
    for source, fit in (("human_scheduled", analysis["human"]), ("human_full_validation", analysis["human_full"])):
        boards.extend({"source": source, **row} for row in fit["leaderboard"])
    for judge in analysis["judges"]:
        members = {tuple(member) for member in judge["members"]}
        calls = [record for record in records if (record["variant"], record.get("draw", 0)) in members]
        new = [record for record in calls if record["variant"] == QWEN]
        reused = [record for record in calls if record["variant"] != QWEN]
        total = BASE.costs(calls)
        judge["cached_charge_or_estimate_usd"] = judge["estimated_usd"]
        judge["estimated_usd"] = total["known_economic_usd"] if not total["n_unknown"] else None
        judge["cost_breakdown"] = total
        judge["new_followup_cost"] = BASE.costs(new)
        judge["reused_call_cost"] = BASE.costs(reused)
        judge["cost_note"] = "Underlying successful calls only. Only Qwen records are new follow-up calls; panel aggregation makes no API calls. Member costs overlap across panels and must not be summed as experiment spending."
        judge["terminal_failed_calls"] = 0
        judge["unavailable_member_variants"] = sorted({name for name, _ in members} & set(exclusions))
        judge["execution_status"] = "unavailable" if judge["unavailable_member_variants"] else "completed"
        if judge["coverage"]["missing_calls"] and not judge["unavailable_member_variants"]:
            raise ValueError("A completed judge has undeclared missing calls")
        comparisons.append({"variant": judge["variant"], "label": judge["label"], **judge["metrics"],
                            "criterion_coverage": judge["coverage"]["criterion_coverage"],
                            "scored_pairs": judge["coverage"]["scored_pairs"],
                            "missing_calls": judge["coverage"]["missing_calls"],
                            "execution_status": judge["execution_status"],
                            "unavailable_member_variants": "; ".join(judge["unavailable_member_variants"]),
                            "successful_inference_usd": judge["estimated_usd"],
                            "new_successful_calls": len(new), "reused_calls": len(reused),
                            "known_new_successful_inference_usd": judge["new_followup_cost"]["known_economic_usd"]})
        intervals = (judge.get("bootstrap") or {}).get("rank_intervals", {}).get("judge", {})
        for row in judge["fit"]["leaderboard"]:
            interval = intervals.get(row["model"], {})
            boards.append({"source": judge["variant"], **row, "execution_status": judge["execution_status"],
                           "rank_interval_low": interval.get("low"), "rank_interval_high": interval.get("high")})
    write_json(study / f"{PHASE}_analysis.json", analysis)
    BASE.export_csv(study / "exports" / f"{PHASE}_judges.csv", comparisons,
                    ["variant", "label", "kendall_tau_b", "spearman", "pair_order_agreement", "rank_mae",
                     "probability_mae", "criterion_coverage", "scored_pairs", "missing_calls", "execution_status",
                     "unavailable_member_variants", "successful_inference_usd", "new_successful_calls",
                     "reused_calls", "known_new_successful_inference_usd"])
    BASE.export_csv(study / "exports" / f"{PHASE}_model_leaderboards.csv", boards,
                    ["source", "model", "rank", "score", "n_comparisons", "n_subjects", "rank_interval_low",
                     "rank_interval_high", "execution_status"])
    return analysis


def paired_deltas(study, analysis, bootstrap, exclusions):
    manifest = VERIFY.load(study / "manifest.json")
    variants = {variant["variant"]: variant for variant in VERIFY.load(study / "variants.json")}
    dataset = Dataset(Path("studentbench_data_ssot"), manifest["instrument"])
    if dataset.fingerprint != manifest["dataset_sha256"]:
        raise ValueError("Paired comparison SSOT differs from the follow-up manifest")
    phase = _round_rows(study, PHASE, "focused_jobs.json", dataset, manifest, variants, set(exclusions))
    candidates = {name: rows for name, rows in phase["judges"].items() if name != BASELINE}
    result = paired_bootstrap_deltas(phase["human"], phase["judges"][BASELINE], candidates,
                                    models=analysis["models"], n_bootstrap=bootstrap, seed=manifest["seed"])
    result["interpretation"] = result["interpretation"].replace(" and one selected development example", "")
    result["interpretation"] += " Small panels were selected using earlier validation outcomes; there is no independent follow-up test or new example intervention."
    result.update(evaluation_role=analysis["evaluation_role"], dataset_sha256=dataset.fingerprint,
                  complete_phases=[PHASE], pending_phases=[],
                  status="complete_with_provider_unavailable" if exclusions else "complete",
                  main_analysis_sha256={PHASE: VERIFY.extension.sha256(study / f"{PHASE}_analysis.json")},
                  jobs_sha256={PHASE: phase["jobs_sha256"]})
    actual = {judge["variant"]: judge for judge in analysis["judges"]}
    for name, comparison in result["comparisons"].items():
        comparison.update(role="exploratory_followup", source_analysis=PHASE,
                          criterion_coverage=phase["coverage"][name])
        for metric, values in comparison["metrics"].items():
            for field, variant in (("baseline", BASELINE), ("candidate", name)):
                expected, observed = actual[variant]["metrics"][metric], values[field]
                if ((expected is None) != (observed is None) or
                        expected is not None and abs(expected - observed) > 1e-10):
                    raise ValueError("Paired point metric differs from the main follow-up analysis")
    write_json(study / "paired_deltas.json", result)
    return result


def main(study=STUDY, bootstrap=500):
    study = Path(study)
    checked = VERIFY.verify(study, require_outputs=False, write=False)
    jobs = VERIFY.load(study / "focused_jobs.json")
    records = [VERIFY.load(call_path(study, job)) for job in jobs if call_path(study, job).exists()]
    new = [record for record in records if record["variant"] == QWEN]
    reused = [record for record in records if record["variant"] != QWEN]
    errors = [{**VERIFY.load(study / row["path"]), **row["billing"]}
              for row in checked["error_ledger"]["records"] if not row["counted_in_successful_recovery"]]
    print(json.dumps({"event": "analyzing", "phase": PHASE, "bootstrap": bootstrap,
                      **checked["counts"]}), flush=True)
    analysis = analyze_round(study, PHASE, study / "focused_jobs.json", bootstrap=bootstrap, density=True)
    analysis = enrich_and_export(study, analysis, records, checked["exclusions"])
    paired_deltas(study, analysis, bootstrap, checked["exclusions"])
    variants = VERIFY.load(study / "variants.json")
    availability = {}
    for variant in variants:
        name = variant["variant"]
        calls = [record for record in records if record["variant"] == name]
        availability[name] = {"status": "unavailable" if name in checked["exclusions"] else "available",
                              "completed_calls": len(calls), "terminal_failed_calls": 0,
                              "response_formats": dict(Counter(r.get("effective_response_format", "original-schema") for r in calls)),
                              "response_providers": dict(Counter(r.get("response_provider") or "native OpenAI" for r in calls)),
                              **checked["exclusions"].get(name, {})}
    new_cost, failed_cost = BASE.costs(new), BASE.costs(errors)
    summary = {"status": "complete_with_provider_unavailable" if checked["exclusions"] else "complete",
               "created_at": datetime.now(timezone.utc).isoformat(), "evaluation_role": analysis["evaluation_role"],
               "availability": availability, "counts": checked["counts"],
               "execution": checked["execution"],
               "estimated_usd": new_cost["known_economic_usd"] + failed_cost["known_economic_usd"],
               "costs": {"new_successful_calls": new_cost, "failed_attempts": failed_cost,
                         "imported_controls": BASE.costs(reused)},
               "costs_total": BASE.economic_total(new + errors),
               "costs_by_variant": {v["variant"]: BASE.economic_total([r for r in records if r["variant"] == v["variant"]]) for v in variants},
               "serialization_by_variant": {v["variant"]: BASE.serialization([r for r in records if r["variant"] == v["variant"]]) for v in variants},
               "cost_note": "Only newly attempted Qwen calls belong to follow-up spending. All 480 copied control calls retain their original cost/catalog provenance and add no new spending. Panels aggregate existing member verdicts and add no calls. Successful and non-recovered failed attempts are counted once; unknown failed-attempt charges remain unknown, not zero.",
               "verification": {"status": checked["status"], "file": "verification.json"},
               "original_selection_unchanged": checked["original_selection_unchanged"]}
    # The frozen previous manifest pins the prior selection; no selection file is
    # written by this pipeline. Verification checks that manifest byte identity.
    summary["artifact_sha256"] = artifact_hashes(study)
    write_json(study / "summary.json", summary)
    VERIFY.verify(study)
    summary["artifact_sha256"] = artifact_hashes(study)
    write_json(study / "summary.json", summary)
    print(json.dumps({"event": "complete", **summary["counts"], "known_new_inference_usd": summary["estimated_usd"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    main(args.study, args.bootstrap)
