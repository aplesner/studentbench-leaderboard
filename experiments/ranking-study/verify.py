"""Verify a completed study and export leaderboards without any provider calls.

This reads frozen inputs/cached responses directly. It does not invoke the
runner or freeze helper, rewrite configurations, or modify cached calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from math import isclose
from pathlib import Path
import tomllib
from unittest.mock import patch

from studentbench_leaderboard.comparison import ConfiguredJudge
from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.judge import LiteLLMJudge, make_task
from studentbench_leaderboard.judge_config import load_definition
from studentbench_leaderboard.pretest_context import PretestContextBank
from studentbench_leaderboard.structured_judge import parse_judge_response
from studentbench_leaderboard.study import DEFAULT_STUDY, call_path, catalog, estimated_cost, write_json


JOB_FILES = ("round1_jobs.json", "round2_jobs.json", "final_execution_jobs.json")


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def job_key(job: dict) -> tuple[str, str, int]:
    return job["variant"], job["pair_id"], job.get("draw", 0)


def expected_jobs(study: Path) -> tuple[dict, dict]:
    """Require unique jobs within stages; de-duplicate legitimate stage reuse."""
    jobs, stages = {}, {}
    for name in JOB_FILES:
        rows = load(study / name)
        keys = [job_key(row) for row in rows]
        require(len(keys) == len(set(keys)), f"Duplicate jobs in {name}")
        stages[name] = len(rows)
        for row in rows:
            key = job_key(row)
            if key in jobs:
                require(jobs[key] == row, f"Conflicting repeated job identity: {key}")
            jobs[key] = row
    require(bool(jobs), "Expected job ledger is empty")
    return jobs, stages


def check_completion(study: Path) -> tuple[dict, dict, dict]:
    """Perform completion and lock checks before loading any dataset labels."""
    summary, manifest = load(study / "analysis.json"), load(study / "manifest.json")
    require(summary.get("status") == "complete", "Final analysis is not complete")
    selection = load(study / "selection.json")
    require(manifest.get("selection_locked") is True, "Selection is not locked")
    require(manifest.get("selection") == selection, "Manifest and selection file differ")
    require(manifest.get("selection_sha256") == digest(selection), "Selection lock hash differs")
    require(summary.get("selected_judge") == selection, "Final summary changed the selected judge")
    require(load(study / "selection_rule.json") == selection["rule"], "Selection rule differs from its frozen record")
    return summary, manifest, selection


def check_cached_call(record: dict, job: dict, entry: dict, task, judge: ConfiguredJudge,
                      manifest: dict, context_provenance: dict | None = None) -> None:
    """Reconstruct request identity and every normalized score, without calls."""
    require(job_key(record) == job_key(job), "Cached call job identity differs")
    for key in ("subject", "plan_ids", "model_a", "model_b", "grading_id", "human_reviewed", "split"):
        require(record[key] == entry[key], f"Cached call catalog mismatch: {key}")
    require(record["dataset_sha256"] == manifest["dataset_sha256"], "Cached call SSOT fingerprint differs")
    require(record["criteria"] == task.metadata()["criteria"], "Cached call criterion metadata differs")
    require(record["judge"] == judge.configuration(), "Cached judge configuration differs")
    swapped = bool(int(digest([manifest["seed"], entry["pair_id"]]), 16) % 2) ^ bool(job.get("draw", 0) % 2)
    displayed = judge.call_task(task, swapped)
    require(record["order"] == ("BA" if swapped else "AB"), "Cached presentation order differs")
    require(record["presented_plan_ids"] == displayed.plan_ids, "Cached presented plan IDs differ")
    messages, schema = displayed.messages(), displayed.schema()
    require(record["messages"] == messages, "Cached prompt differs from its frozen inputs")
    require(record["response_schema"] == schema, "Cached response schema differs")
    request = digest({"messages": messages, "schema": schema, "judge": judge.configuration(),
                      "draw": job.get("draw", 0)})
    require(record["request_id"] == request, "Cached request hash differs")
    parsed = parse_judge_response(record["raw_response"], displayed)
    require(record["ratings"] == parsed, "Cached parsed response differs from the raw response")
    normalized = {key: {**value, "score": -value["score"] if swapped and value["score"] is not None
                        else value["score"]} for key, value in parsed.items()}
    require(record["normalized_ratings"] == normalized, "Cached normalized preferences differ")
    require(record.get("pretest_context_provenance") == context_provenance,
            "Cached pretest-context provenance differs")


def export_leaderboard(path: Path, fit: dict, intervals: dict | None, label: str) -> None:
    fields = ("source", "model", "rank", "score", "n_comparisons", "n_subjects", "connected",
              "rank_interval_low", "rank_interval_high", "bootstrap_valid_draws", "top1_frequency")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in fit["leaderboard"]:
            interval = (intervals or {}).get(row["model"], {})
            writer.writerow({"source": label, "model": row["model"], "rank": row["rank"],
                             "score": row["score"], "n_comparisons": row["n_comparisons"],
                             "n_subjects": row["n_subjects"], "connected": fit["connected"],
                             "rank_interval_low": interval.get("low"), "rank_interval_high": interval.get("high"),
                             "bootstrap_valid_draws": interval.get("n_valid"),
                             "top1_frequency": interval.get("top1_frequency")})
    temporary.replace(path)


def summarize_errors(study: Path, jobs: dict) -> dict:
    """Report saved failed attempts separately from the final successful ledger."""
    counts, grouped = Counter(), {}
    resolved = unresolved = 0
    for path in sorted((study / "errors").glob("*.json")):
        error = load(path)
        key = job_key(error)
        status = f"{error.get('type', 'unknown')} / {error.get('status_code', 'unknown')}"
        counts[status] += 1
        exists = call_path(study, error).is_file()
        resolved += exists
        unresolved += not exists
        item = grouped.setdefault(key, {"variant": key[0], "pair_id": key[1], "draw": key[2],
                                        "in_expected_ledger": key in jobs, "final_call_exists": exists,
                                        "saved_error_attempts": []})
        item["saved_error_attempts"].append({"file": str(path.relative_to(study)),
                                            "attempt": error.get("attempt"), "type": error.get("type"),
                                            "status_code": error.get("status_code")})
    return {"saved_error_attempts": resolved + unresolved, "resolved_attempts": resolved,
            "unresolved_attempts": unresolved, "type_status_counts": dict(counts),
            "jobs": list(grouped.values()),
            "interpretation": "A saved failed attempt is resolved when its final cached call exists and passed reconstruction. Saved error files are not a provider billing ledger or a guaranteed complete history of earlier reruns."}


def verify(study: Path = DEFAULT_STUDY, ssot: Path = Path("studentbench_data_ssot")) -> dict:
    study = Path(study).resolve()
    summary, manifest, selection = check_completion(study)
    jobs, stages = expected_jobs(study)
    expected_paths = {call_path(study, job) for job in jobs.values()}
    actual_paths = set((study / "calls").glob("*.json"))
    require(expected_paths == actual_paths,
            f"Call ledger mismatch: {len(expected_paths - actual_paths)} missing, {len(actual_paths - expected_paths)} unexpected")
    require(len(jobs) == summary["new_completed_calls"], "Final summary call count differs from the expected ledger")
    inventory = load(study / "config_inventory.json")
    actual_inventory = {str(path.relative_to(study)): digest(path.read_text())
                        for path in (study / "configs").rglob("*") if path.is_file()}
    require(inventory == actual_inventory, "Frozen configuration file inventory changed")
    require(selection["config_inventory_sha256"] == digest(inventory), "Selection configuration inventory hash differs")
    dataset = Dataset(ssot, manifest["instrument"])
    require(dataset.fingerprint == manifest["dataset_sha256"], "Current SSOT differs from the frozen study")
    dataset.verify_manifest()
    pairs = catalog(manifest)
    variant_rows = load(study / "variants.json")
    variants = {row["variant"]: row for row in variant_rows}
    require(len(variants) == len(variant_rows), "Duplicate variant definitions")
    definitions, banks = {}, {}
    for name, variant in variants.items():
        definition = load_definition(variant["judge_id"], study / "configs" / name)
        require(len(definition.members) == 1 and definition.members[0].model == variant["model"]
                and definition.members[0].reasoning_effort == variant["reasoning_effort"]
                and definition.response_style == variant["response_style"],
                f"Variant metadata differs from its configuration: {name}")
        definitions[name] = ConfiguredJudge(definition)
        if context := variant.get("pretest_context"):
            declared = tomllib.loads((study / "configs" / name / "context_input.toml").read_text())
            require(declared["source_bank"] == context["file"] and declared["source_sha256"] == context["sha256"]
                    and declared["projection"] == definition.plan_projection,
                    f"Variant context declaration differs: {name}")
            source = (study / "configs" / name / context["file"]).resolve()
            require(source.is_relative_to((study / "configs" / name).resolve()), "Context source escapes its frozen configuration")
            banks[name] = PretestContextBank.load(dataset.root, source_bank=source,
                                                  expected_source_sha256=context["sha256"])
    call_inventory, request_ids = {}, set()
    runner_hashes, response_models, split_counts, completed_attempts = Counter(), Counter(), Counter(), Counter()
    total_cost = 0.0
    token_totals = Counter()
    # Defense in depth: even an accidental future call through the known adapter fails locally.
    with patch.object(LiteLLMJudge, "complete", side_effect=AssertionError("Verifier must never call a model")):
        for key, job in sorted(jobs.items()):
            path = call_path(study, job)
            record = load(path)
            entry = pairs[job["pair_id"]]
            task = make_task(dataset, entry["grading_id"], entry["plan_ids"])
            bank = banks.get(job["variant"])
            if bank is not None:
                task = bank.enrich_task(task)
            check_cached_call(record, job, entry, task, definitions[job["variant"]], manifest,
                              bank.provenance if bank is not None else None)
            require(record["request_id"] not in request_ids, "Duplicate request hash in distinct ledger jobs")
            request_ids.add(record["request_id"])
            cost = estimated_cost(variants[job["variant"]]["model"], record.get("usage", {}))
            require(isclose(cost, record["estimated_usd"], rel_tol=1e-12, abs_tol=1e-12), "Cached token-cost estimate differs")
            total_cost += cost
            for token_type in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_totals[token_type] += record.get("usage", {}).get(token_type, 0) or 0
            runner_hashes[record["runner_sha256"]] += 1
            response_models[record.get("response_model", "unknown")] += 1
            split_counts[entry["split"]] += 1
            completed_attempts[record.get("attempts", 1)] += 1
            call_inventory[str(path.relative_to(study))] = sha256(path)
    require(isclose(total_cost, summary["estimated_usd"], rel_tol=1e-10, abs_tol=1e-10), "Final token-cost estimate differs from calls")
    require(dict(token_totals) == summary["tokens"], "Final token totals differ from calls")
    analyses = {}
    for name in ("round1", "round2", "test", "dense"):
        analysis = load(study / f"{name}_analysis.json")
        require(analysis.get("complete") is True, f"{name} analysis is incomplete")
        require(analysis["dataset_sha256"] == dataset.fingerprint, f"{name} analysis uses another SSOT")
        require(analysis["jobs_sha256"] == digest(load(study / f"{name}_jobs.json")), f"{name} job schedule differs from its analysis")
        analyses[name] = analysis
    require(analyses["round2"]["jobs_sha256"] == selection["validation_jobs_sha256"], "Selection used a different validation schedule")
    require(analyses["test"]["human"] == analyses["dense"]["human"], "Reviewed and dense results use different human references")
    chosen = {}
    for name in ("test", "dense"):
        matches = [row for row in analyses[name]["judges"] if row["variant"] == selection["variant"]]
        require(len(matches) == 1, f"Selected judge missing or duplicated in {name} analysis")
        chosen[name] = matches[0]
        require(chosen[name]["metrics"] == summary[f"{name}_metrics"], f"Summary {name} metrics differ from the selected result")
        require(chosen[name]["members"] == selection["members"] and chosen[name]["aggregation"] == selection["aggregation"],
                f"Selected strategy changed in {name} analysis")
    exports = study / "exports"
    exports.mkdir(exist_ok=True)
    files = {}
    reviewed_bootstrap = chosen["test"].get("bootstrap") or {}
    dense_bootstrap = chosen["dense"].get("bootstrap") or {}
    reviewed_intervals = reviewed_bootstrap.get("rank_intervals", {}) if not chosen["test"].get("bootstrap_cohort_warning") else {}
    dense_intervals = dense_bootstrap.get("rank_intervals", {}) if not chosen["dense"].get("bootstrap_cohort_warning") else {}
    for filename, fit, intervals, label in (
        ("human_test_leaderboard.csv", analyses["test"]["human"], reviewed_intervals.get("human"), "human_test_exact_task_criteria"),
        ("selected_ai_reviewed_leaderboard.csv", chosen["test"]["fit"], reviewed_intervals.get("judge"), selection["variant"] + ":reviewed"),
        ("selected_ai_dense_leaderboard.csv", chosen["dense"]["fit"], dense_intervals.get("judge"), selection["variant"] + ":dense"),
    ):
        path = exports / filename
        export_leaderboard(path, fit, intervals, label)
        files[str(path.relative_to(study))] = sha256(path)
    root = Path(__file__).resolve().parents[2]
    source_files = [path for directory in (root / "src/studentbench_leaderboard", root / "experiments/ranking-study")
                    for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".toml", ".txt", ".json"}]
    source_files += [root / "pyproject.toml", root / "uv.lock"]
    source_inventory = {str(path.relative_to(root)): sha256(path) for path in sorted(source_files)}
    artifact_names = ("manifest.json", "selection.json", "selection_rule.json", "variants.json", "config_inventory.json",
                      "analysis.json", "round1_analysis.json", "round2_analysis.json", "test_analysis.json", "dense_analysis.json",
                      *JOB_FILES, "test_jobs.json", "dense_jobs.json")
    result = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
              "network_calls": 0, "verification_method": "read_only_reconstruction_and_response_reparse",
              "selected_variant": selection["variant"], "selection_sha256": digest(selection),
              "dataset_sha256": dataset.fingerprint, "stage_job_counts": stages,
              "unique_expected_calls": len(jobs), "verified_calls": len(call_inventory),
              "unique_request_hashes": len(request_ids), "split_call_counts": dict(split_counts),
              "estimated_usd": total_cost, "tokens": dict(token_totals),
              "cost_scope": "Estimates use returned token usage from completed calls. Failed or timed-out attempts may have unreported usage and are excluded.",
              "error_ledger": summarize_errors(study, jobs),
              "completed_call_attempt_histogram": dict(sorted(completed_attempts.items())),
              "response_models": dict(response_models), "observed_runner_hashes": dict(runner_hashes),
              "source_inventory_note": "These are current source-file checksums. Observed runner hashes identify execution versions; this does not assert that every historical source version was archived. Every saved request and parsed result was reconstructed successfully with the frozen inputs.",
              "source_sha256": source_inventory, "call_sha256": call_inventory,
              "config_file_sha256": {name: sha256(study / name) for name in sorted(inventory)},
              "artifact_sha256": {name: sha256(study / name) for name in artifact_names},
              "export_sha256": files}
    write_json(study / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--ssot", type=Path, default=Path("studentbench_data_ssot"))
    args = parser.parse_args()
    result = verify(args.study, args.ssot)
    print(json.dumps({key: result[key] for key in ("status", "verified_calls", "selected_variant", "network_calls")}, indent=2))


if __name__ == "__main__":
    main()
