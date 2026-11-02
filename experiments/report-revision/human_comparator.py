"""Matched-support comparison to the same independent human-group reference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.human_analysis import consensus_agreement, reviewer_halves, summarize_numbers
from studentbench_leaderboard.ranking import aggregate_pairs, compare_rankings, fit_bradley_terry
from studentbench_leaderboard.study import call_path, catalog
from studentbench_leaderboard.study_analysis import _human_rows, criterion_family, materialize_judge

STUDY = Path("results/ranking-study-20260910")
OUTPUT = Path("results/report-revision-20260911")
METRICS = ("kendall_tau_b", "spearman", "pair_order_agreement", "rank_mae")
load = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
unit = lambda row: (row["pair_id"], row["criterion"])


def scope(rows):
    return {"pairs": len({r["pair_id"] for r in rows}), "subjects": len({r["subject"] for r in rows}),
            "criterion_units": len({unit(r) for r in rows}), "votes": len(rows),
            "reviewers": len({r["reviewer"] for r in rows if "reviewer" in r}),
            "review_sessions": len({r["grading_id"] for r in rows if "grading_id" in r})}


def compare_partition(left, right, ai, models):
    """Return one counterbalanced result on identical scored exact units."""
    scored = {name: [row for row in rows if row["value"] is not None]
              for name, rows in (("left", left), ("right", right), ("ai", ai))}
    common = set.intersection(*({unit(row) for row in rows} for rows in scored.values()))
    rows = {name: [row for row in values if unit(row) in common] for name, values in scored.items()}
    if {row.get("reviewer") for row in rows["left"] if row.get("reviewer")} & {
            row.get("reviewer") for row in rows["right"] if row.get("reviewer")}:
        raise ValueError("Human reference/comparator reviewer groups must be disjoint")
    assert {unit(row) for row in rows["left"]} == {unit(row) for row in rows["right"]} == {unit(row) for row in rows["ai"]}
    fits = {name: fit_bradley_terry(aggregate_pairs(value), models=models) for name, value in rows.items()}
    record = {"support": {name: scope(value) for name, value in rows.items()},
              "connected": all(fit["connected"] for fit in fits.values()),
              "converged": all(fit["converged"] for fit in fits.values()),
              "orientations": [], "metrics": {}}
    for reference, comparator in (("left", "right"), ("right", "left")):
        ai_metrics = compare_rankings(fits[reference], fits["ai"])
        human_metrics = compare_rankings(fits[reference], fits[comparator])
        record["orientations"].append({"reference_group": reference, "human_comparator_group": comparator,
            "ai_vs_reference": {key: ai_metrics[key] for key in METRICS},
            "human_vs_reference": {key: human_metrics[key] for key in METRICS},
            "criterion_direction_agreement_ai": consensus_agreement(rows[reference], rows["ai"]),
            "criterion_direction_agreement_human": consensus_agreement(rows[reference], rows[comparator])})
    for metric in METRICS:
        orientations = record["orientations"]
        if all(o[key][metric] is not None for o in orientations for key in ("ai_vs_reference", "human_vs_reference")):
            ai_mean = sum(o["ai_vs_reference"][metric] for o in orientations) / 2
            human_mean = sum(o["human_vs_reference"][metric] for o in orientations) / 2
            record["metrics"][metric] = {"ai_vs_reference": ai_mean, "human_vs_reference": human_mean,
                                         "delta_ai_minus_human": ai_mean - human_mean}
    return record


def summarize_partitions(records):
    """One observation per original partition, never per reference orientation."""
    summaries = {}
    for metric in METRICS:
        usable = [r["metrics"][metric] for r in records if metric in r["metrics"]]
        deltas = np.asarray([r["delta_ai_minus_human"] for r in usable])
        summaries[metric] = {key: summarize_numbers([r[key] for r in usable]) for key in
                             ("ai_vs_reference", "human_vs_reference", "delta_ai_minus_human")}
        summaries[metric].update(proportion_delta_positive=float(np.mean(deltas > 1e-12)) if len(deltas) else None,
                                 proportion_delta_negative=float(np.mean(deltas < -1e-12)) if len(deltas) else None,
                                 proportion_delta_zero=float(np.mean(np.abs(deltas) <= 1e-12)) if len(deltas) else None,
                                 zero_tolerance=1e-12)
    return summaries


def main():
    manifest = load(STUDY / "manifest.json")
    analysis = load(STUDY / "test_analysis.json")
    human_analysis = load(STUDY / "human/test.json")
    protocol = OUTPUT / "human_comparator_protocol.json"
    dataset = Dataset(Path("studentbench_data_ssot"), manifest["instrument"])
    assert dataset.fingerprint == manifest["dataset_sha256"]
    assert manifest["selection_locked"] and manifest["selection"]["variant"] == "enhanced_terra_high"
    variant = manifest["selection"]["variant"]
    pairs = catalog(manifest)
    jobs = [j for j in load(STUDY / "test_jobs.json") if j["variant"] == variant]
    ids = sorted({j["pair_id"] for j in jobs})
    units = {pair: [{"criterion": c["id"] + "|" + c["scale_id"],
                    "criterion_family": criterion_family(c["id"], dataset)}
                   for c in dataset.rubric(pairs[pair]["grading_id"])] for pair in ids}
    paths = [call_path(STUDY, j) for j in jobs]
    before = {str(p): sha(p) for p in paths + [STUDY / "manifest.json", STUDY / "test_analysis.json", STUDY / "human/test.json"]}
    calls = {(j["variant"], j["pair_id"], j.get("draw", 0)): load(path) for j, path in zip(jobs, paths)}
    definition = next(j for j in analysis["judges"] if j["variant"] == variant)
    ai, coverage = materialize_judge(definition, ids, pairs, units, calls)
    ai = [r for r in ai if r["value"] is not None]
    humans = _human_rows(dataset, set(manifest["splits"]["test"]), pairs)
    old = human_analysis["reviewer_split_half"]
    models = analysis["models"]
    records = []
    for repeat in range(old["repeats"]):
        left, right, original_counts = reviewer_halves(humans, np.random.default_rng(old["seed"] + repeat))
        previous = old["runs"][repeat]
        assert original_counts["left"]["pairs"] == previous["matched_pairs"]
        assert original_counts["left"]["subjects"] == previous["matched_subjects"]
        record = {**compare_partition(left, right, ai, models), "repeat": repeat, "seed": old["seed"] + repeat,
                  "original_matched_pairs": previous["matched_pairs"],
                  "original_matched_subjects": previous["matched_subjects"],
                  "n_original_matched_criterion_units": len({unit(r) for r in left}),
                  "reviewer_partition_sha256": digest({key: original_counts[key] for key in ("left_reviewer_ids", "right_reviewer_ids")})}
        records.append(record)
    summaries = summarize_partitions(records)
    valid = [r for r in records if r["connected"] and r["converged"]]
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(protocol),
              "script_sha256": sha(__file__), "script_file": "experiments/report-revision/human_comparator.py",
              "source_code_sha256": {name: sha(Path("src/studentbench_leaderboard") / name)
                                     for name in ("ranking.py", "human_analysis.py", "study_analysis.py")},
              "variant": variant, "split": "test", "n_models": len(models),
              "selection_sha256": manifest["selection_sha256"], "dataset_sha256": dataset.fingerprint,
              "n_partitions": len(records), "n_valid_partitions": len(valid),
              "n_disconnected_partitions": sum(not r["connected"] for r in records),
              "n_nonconverged_partitions": sum(not r["converged"] for r in records),
              "full_selected_judge_coverage": coverage, "summary": summaries,
              "valid_partition_support": {side: {key: summarize_numbers([r["support"][side][key] for r in valid])
                                                        for key in scope([])} for side in ("left", "right", "ai")},
              "valid_partition_original_units": summarize_numbers([r["n_original_matched_criterion_units"] for r in valid]),
              "source_artifact_sha256": before, "runs": records,
              "interpretation": "Each of200 existing reviewer partitions provides a common held-out human-group reference for AI and the other human group on identical scored pair/exact-criterion support. Both reference orientations are averaged within each partition. Percentiles describe dependent repeated-partition sensitivity, not formal confidence intervals, p-values or a claim of human/AI superiority. Original full-test point metrics use more data and a different reference. Outputs, reviewed student cohort and the selected judge are fixed."}
    assert all(sha(Path(path)) == value for path, value in before.items())
    result["protected_artifacts_byte_unchanged"] = True
    path = OUTPUT / "human_comparator.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"file": str(path), "n_valid_partitions": len(valid), "summary": summaries}), flush=True)


if __name__ == "__main__":
    main()
