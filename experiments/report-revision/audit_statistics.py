"""Offline report audit. Reuses saved calls and writes only this audit directory."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.paired_deltas import _pair_rows, _round_rows, _sample
from studentbench_leaderboard.ranking import aggregate_pairs, compare_rankings, fit_bradley_terry
from studentbench_leaderboard.study import call_path, catalog
from studentbench_leaderboard.study_analysis import _call_votes, _human_rows, _key, criterion_family, materialize_judge

ORIGINAL = Path("results/ranking-study-20260910")
EXTENSION = Path("results/ranking-extension-20260911")
OUTPUT = Path("results/report-revision-20260911")
METRICS = ("kendall_tau_b", "spearman", "pair_order_agreement", "rank_mae")
SEED, REPEATS = 20260910, 500


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_summary(values):
    values = list(values)
    return {"n": len(values), "min": min(values) if values else None,
            "median": float(np.median(values)) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "max": max(values) if values else None}


def edge_distribution(rows, models):
    counts = Counter(tuple(sorted((r["model_a"], r["model_b"]))) for r in rows)
    possible = list(combinations(models, 2))
    values = [counts[edge] for edge in possible]
    return {"n_plan_pairs": len(rows), "n_possible_generator_pairs": len(possible),
            "n_observed_generator_pairs": sum(value > 0 for value in values),
            "all_generator_pairs": numeric_summary(values),
            "observed_generator_pairs": numeric_summary(value for value in values if value > 0),
            "missing_generator_pairs": [list(edge) for edge in possible if counts[edge] == 0],
            "count_histogram": dict(sorted(Counter(values).items())),
            "comparisons_by_generator_pair": [{"model_a": a, "model_b": b, "n_plan_pairs": counts[(a, b)]}
                                               for a, b in possible]}


def round1_phase(dataset, manifest, wanted):
    """Rebuild frozen single-judge definitions without loading legacy panel prose."""
    jobs = load(ORIGINAL / "round1_jobs.json")
    analysis = load(ORIGINAL / "round1_analysis.json")
    assert analysis["jobs_sha256"] == digest(jobs)
    pairs = catalog(manifest)
    ids = sorted({job["pair_id"] for job in jobs})
    units = {pair: [{"criterion": c["id"] + "|" + c["scale_id"],
                    "criterion_family": criterion_family(c["id"], dataset)}
                   for c in dataset.rubric(pairs[pair]["grading_id"])] for pair in ids}
    expected = {(pair, c["criterion"]) for pair in ids for c in units[pair]}
    human = aggregate_pairs([row for row in _human_rows(dataset, set(manifest["splits"]["validation"]), pairs)
                             if _key(row) in expected])
    calls, hashes = {}, {}
    for job in jobs:
        if job["variant"] not in wanted:
            continue
        path = call_path(ORIGINAL, job)
        record = load(path)
        hashes[str(path)] = sha(path)
        assert record["dataset_sha256"] == dataset.fingerprint
        assert all(record[k] == pairs[job["pair_id"]][k] for k in ("subject", "plan_ids", "model_a", "model_b"))
        assert set(_call_votes(record)) == {c["criterion"] for c in units[job["pair_id"]]}
        calls[(job["variant"], job["pair_id"], job.get("draw", 0))] = record
    rows, coverage = {}, {}
    for definition in analysis["judges"]:
        name = definition["variant"]
        if name not in wanted:
            continue
        values, counts = materialize_judge(definition, ids, pairs, units, calls)
        assert counts["scored_criteria"] == definition["coverage"]["scored_criteria"]
        rows[name], coverage[name] = aggregate_pairs(values), counts
    assert set(rows) == set(wanted)
    return {"human": human, "judges": rows, "coverage": coverage, "analysis": analysis,
            "pair_ids": ids, "input_call_sha256": hashes}


def paired_audit(phase, systems, contrasts, models):
    human = _pair_rows(phase["human"], label="human")
    rows = {name: _pair_rows(phase["judges"][name], label=name) for name in systems}
    subjects = sorted({row["subject"] for row in human})
    point_human = fit_bradley_terry(human, models=models)
    point_fits = {name: fit_bradley_terry(value, models=models) for name, value in rows.items()}
    point_metrics = {name: compare_rankings(point_human, value) for name, value in point_fits.items()}
    saved = {j["variant"]: j for j in phase["analysis"]["judges"]}
    for name in systems:
        for metric in METRICS:
            assert abs(point_metrics[name][metric] - saved[name]["metrics"][metric]) < 1e-12
    result = {"n_subjects": len(subjects), "n_pairs": len(human), "seed": SEED,
              "n_bootstrap": REPEATS, "confidence": .95, "ridge": 1.0,
              "human_pairs_sha256": digest(human), "subject_order_sha256": digest(subjects),
              "contrasts": {}}
    for label, baseline, candidate in contrasts:
        result["contrasts"][label] = {
            "baseline": baseline, "candidate": candidate,
            "n_disconnected_draws": 0, "n_nonconverged_draws": 0, "n_valid_fits": 0,
            "metrics": {metric: {"baseline": point_metrics[baseline][metric],
                                   "candidate": point_metrics[candidate][metric],
                                   "delta": point_metrics[candidate][metric] - point_metrics[baseline][metric],
                                   "draws": []} for metric in METRICS}}
    rng, resamples = np.random.default_rng(SEED), []
    for index in range(REPEATS):
        counts = Counter(rng.choice(subjects, size=len(subjects), replace=True).tolist())
        resamples.append([counts[s] for s in subjects])
        reference = fit_bradley_terry(_sample(human, counts), models=models)
        fitted = {name: fit_bradley_terry(_sample(value, counts), models=models) for name, value in rows.items()}
        metrics = {name: compare_rankings(reference, fit) for name, fit in fitted.items()}
        for label, baseline, candidate in contrasts:
            output = result["contrasts"][label]
            fits = [reference, fitted[baseline], fitted[candidate]]
            disconnected = any(not fit["connected"] for fit in fits)
            nonconverged = any(not fit["converged"] for fit in fits)
            output["n_disconnected_draws"] += disconnected
            output["n_nonconverged_draws"] += nonconverged
            if disconnected or nonconverged:
                continue
            output["n_valid_fits"] += 1
            for metric in METRICS:
                left, right = metrics[baseline][metric], metrics[candidate][metric]
                if left is not None and right is not None:
                    output["metrics"][metric]["draws"].append(right - left)
        if (index + 1) % 100 == 0:
            print(json.dumps({"event": "bootstrap", "pairs": len(human), "draws": index + 1}), flush=True)
    result["resample_counts_sha256"] = digest(resamples)
    alpha = (1 - .95) / 2
    for contrast in result["contrasts"].values():
        for metric, entry in contrast["metrics"].items():
            values = np.asarray(entry["draws"])
            ordered = np.sort(values)
            low, high = np.quantile(values, [alpha, 1 - alpha]).tolist()
            point = entry["delta"]
            equal = np.isclose(values, point, atol=1e-12, rtol=0)
            lo_position, hi_position = alpha * (len(values) - 1), (1 - alpha) * (len(values) - 1)
            entry.update(low=low, high=high, n_valid=len(values),
                         mean=float(np.mean(values)), median=float(np.median(values)),
                         min=float(np.min(values)), max=float(np.max(values)),
                         proportion_delta_gt_zero=float(np.mean(values > 0)),
                         proportion_delta_lt_zero=float(np.mean(values < 0)),
                         proportion_delta_equal_zero=float(np.mean(values == 0)),
                         n_draws_below_point=int(np.sum((values < point) & ~equal)),
                         n_draws_equal_point_tolerance_1e12=int(np.sum(equal)),
                         n_draws_above_point=int(np.sum((values > point) & ~equal)),
                         histogram_rounded_12_decimals=dict(sorted(Counter(round(float(v), 12) for v in values).items())),
                         percentile_method="NumPy default linear (Hyndman-Fan type 7)",
                         low_zero_based_position=lo_position, high_zero_based_position=hi_position,
                         low_bracketing_values=[float(ordered[int(np.floor(lo_position))]), float(ordered[int(np.ceil(lo_position))])],
                         high_bracketing_values=[float(ordered[int(np.floor(hi_position))]), float(ordered[int(np.ceil(hi_position))])],
                         lower_exactly_equals_point=low == point)
    return result


def main():
    # These immutable result files must retain their exact bytes after the audit.
    protected = [*ORIGINAL.glob("*_analysis.json"), ORIGINAL / "manifest.json", ORIGINAL / "selection.json",
                 EXTENSION / "paired_deltas.json", EXTENSION / "example_validation_analysis.json",
                 EXTENSION / "provider_validation_analysis.json"]
    original_hashes = {str(path): sha(path) for path in protected}
    manifest = load(ORIGINAL / "manifest.json")
    dataset = Dataset(Path("studentbench_data_ssot"), manifest["instrument"])
    assert dataset.fingerprint == manifest["dataset_sha256"]
    models = sorted({row["ai_model_preset_id"] for row in dataset.plans.values()})
    assert len(models) == 12
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "role": "post-study descriptive audit of saved outputs; no judge selection or model calls",
              "script_sha256": sha(__file__), "script_file": "experiments/report-revision/audit_statistics.py",
              "dataset_sha256": dataset.fingerprint,
              "original_selection_sha256": manifest["selection_sha256"], "source_artifact_sha256": original_hashes,
              "source_code_sha256": {str(path): sha(path) for path in map(Path, [
                  "src/studentbench_leaderboard/ranking.py", "src/studentbench_leaderboard/study_analysis.py",
                  "src/studentbench_leaderboard/paired_deltas.py", "src/studentbench_leaderboard/human_analysis.py"])}}
    output["splits"] = {split: {"subjects": len(manifest["splits"][split]),
                               "available_plan_pairs": len(rows),
                               "reviewed_plan_pairs": sum(r["human_reviewed"] for r in rows),
                               "unreviewed_plan_pairs": sum(not r["human_reviewed"] for r in rows),
                               "distinct_plans": len({plan for r in rows for plan in r["plan_ids"]})}
                        for split, rows in manifest["pair_catalog"].items()}
    output["schedule_counts"] = {}
    for name in ("round1", "round2", "test", "dense", "input_ablation"):
        analysis = load(ORIGINAL / f"{name}_analysis.json")
        output["schedule_counts"][name] = {key: analysis[key] for key in (
            "n_pairs", "n_subjects", "n_expected_calls", "n_completed_calls", "human_coverage")}
        output["schedule_counts"][name].update(human_reference_pairs=analysis["human"]["n_comparisons"],
            whole_split_human_pairs=analysis["human_full"]["n_comparisons"])
    test = manifest["pair_catalog"]["test"]
    output["generator_pair_distribution"] = {
        "reviewed_test": edge_distribution([r for r in test if r["human_reviewed"]], models),
        "dense_test": edge_distribution(test, models)}

    variants = {v["variant"]: v for v in load(ORIGINAL / "variants.json")}
    contrast_definitions = [(name, name + "_low", name + "_high") for name in ("luna", "terra", "sol", "astra")]
    system_names = [variant for _, low, high in contrast_definitions for variant in (low, high)]
    round1 = round1_phase(dataset, manifest, system_names)
    output["reasoning_effort"] = paired_audit(round1, system_names, contrast_definitions, models)
    output["reasoning_effort"]["input_call_sha256"] = round1["input_call_sha256"]
    output["reasoning_effort"]["contract_checks"] = {}
    for name, low, high in contrast_definitions:
        records = {v: [load(call_path(ORIGINAL, {"variant": v, "pair_id": pair, "draw": 0}))
                       for pair in round1["pair_ids"]] for v in (low, high)}
        checks = {"same_messages": 0, "same_schema": 0, "same_order": 0}
        for left, right in zip(records[low], records[high]):
            checks["same_messages"] += left["messages"] == right["messages"]
            checks["same_schema"] += left["response_schema"] == right["response_schema"]
            checks["same_order"] += left["order"] == right["order"]
        checks["pairs"] = len(round1["pair_ids"])
        checks["model"] = variants[low]["model"]
        assert variants[low]["model"] == variants[high]["model"]
        checks["requested_efforts"] = [variants[low]["reasoning_effort"], variants[high]["reasoning_effort"]]
        checks["criterion_coverage"] = {v: round1["coverage"][v] for v in (low, high)}
        output["reasoning_effort"]["contract_checks"][name] = checks

    extension_manifest = load(EXTENSION / "manifest.json")
    extension_variants = {v["variant"]: v for v in load(EXTENSION / "variants.json")}
    example = _round_rows(EXTENSION, "example_validation", "example_jobs.json", dataset,
                          extension_manifest, extension_variants, set())
    example_variants = ["enhanced_terra_high", "terra_high_one_example"]
    output["example_interval_audit"] = paired_audit(example, example_variants,
        [("one_example_minus_zero_example", *example_variants)], models)
    recorded = load(EXTENSION / "paired_deltas.json")
    actual = output["example_interval_audit"]
    assert actual["resample_counts_sha256"] == recorded["resample_counts_sha256"]
    for metric, entry in actual["contrasts"]["one_example_minus_zero_example"]["metrics"].items():
        previous = recorded["comparisons"]["terra_high_one_example"]["metrics"][metric]
        for key in ("baseline", "candidate", "delta", "low", "high", "n_valid"):
            assert entry[key] == previous[key], (metric, key, entry[key], previous[key])
    actual["exactly_reproduces_saved_example_intervals"] = True

    output["human_comparison_context"] = {}
    for split in ("validation", "test"):
        human = load(ORIGINAL / "human" / f"{split}.json")
        halves = human["reviewer_split_half"]
        output["human_comparison_context"][split] = {
            "full_coverage": human["coverage"], "direct_agreement": human["direct_human_agreement"],
            "reviewer_half_repeats": halves["repeats"], "reviewer_half_metrics": halves["summary"],
            "half_pair_counts_all_draws": numeric_summary(r["matched_pairs"] for r in halves["runs"]),
            "half_subject_counts_all_draws": numeric_summary(r["matched_subjects"] for r in halves["runs"]),
            "half_pair_counts_connected_draws": numeric_summary(r["matched_pairs"] for r in halves["runs"]
                                                               if r["left_connected"] and r["right_connected"]),
            "half_subject_counts_connected_draws": numeric_summary(r["matched_subjects"] for r in halves["runs"]
                                                                  if r["left_connected"] and r["right_connected"])}
    output["human_comparison_context"]["full_validation_vs_test"] = load(ORIGINAL / "human" / "validation_test_human_alignment.json")
    reviewers = {split: {s["human_reviewer_anon_id"] for s in dataset.sessions.values()
                        if s["review_subject_id"] in set(subjects)} for split, subjects in manifest["splits"].items()}
    output["reviewer_overlap"] = [{"split_a": a, "split_b": b, "n_shared_reviewers": len(reviewers[a] & reviewers[b])}
                                  for a, b in combinations(reviewers, 2)]
    assert all(sha(path) == expected for path, expected in original_hashes.items())
    output["protected_artifacts_byte_unchanged"] = True
    (OUTPUT / "statistics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "complete", "file": str(OUTPUT / "statistics.json"),
                      "sha256": sha(OUTPUT / "statistics.json")}), flush=True)


if __name__ == "__main__":
    main()
