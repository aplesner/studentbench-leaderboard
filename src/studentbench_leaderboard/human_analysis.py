"""Human reference reliability and sampling-density analysis of model rankings.

This module makes no model calls. A subject set must be supplied explicitly so
validation labels can be analysed without opening a held-out test reference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np

from .data import Dataset
from .ranking import aggregate_pairs, bootstrap_alignment, compare_rankings, fit_bradley_terry


def human_criterion_rows(dataset: Dataset, subjects: set[str] | list[str]) -> list[dict]:
    """Return individual reviewer votes, with fixed canonical plan orientation.

    Strength is deliberately discarded: preference direction maps to an A-win
    value of 0, .5, or 1. The 1--5 per-plan ratings are never used here.
    """
    subjects = set(subjects)
    rows = []
    for grading_id, session in sorted(dataset.sessions.items()):
        subject = session["review_subject_id"]
        if subject not in subjects:
            continue
        original = [session["lesson_plan_a_id"], session["lesson_plan_b_id"]]
        plans = sorted(original)
        orientation = 1 if original == plans else -1
        for grade in dataset.grades_by_session[grading_id]:
            if grade["preference_score"] == "":
                continue
            preference = orientation * int(grade["preference_score"])
            rows.append({
                "subject": subject,
                "pair_id": "|".join([subject, *plans]),
                "model_a": dataset.plans[plans[0]]["ai_model_preset_id"],
                "model_b": dataset.plans[plans[1]]["ai_model_preset_id"],
                "plan_a": plans[0], "plan_b": plans[1],
                "criterion": grade["question_variant_id"] + "|" + dataset.rating_scale(grade),
                "criterion_family": grade["rubric_question_id"],
                "reviewer": session["human_reviewer_anon_id"],
                "grading_id": grading_id,
                "preference": preference,
                "value": (np.sign(preference).item() + 1) / 2,
            })
    return rows


def _unit(row: dict) -> tuple[str, str]:
    return row["pair_id"], row["criterion"]


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return dict(result)


def coverage(rows: list[dict]) -> dict:
    pair_reviewers = defaultdict(set)
    for row in rows:
        pair_reviewers[row["pair_id"]].add(row["reviewer"])
    return {
        "subjects": len({r["subject"] for r in rows}),
        "pairs": len(pair_reviewers),
        "review_sessions": len({r["grading_id"] for r in rows}),
        "reviewers": len({r["reviewer"] for r in rows}),
        "criterion_votes": len(rows),
        "matched_criterion_units": len({_unit(r) for r in rows}),
        "models": sorted({r[k] for r in rows for k in ("model_a", "model_b")}),
        "reviews_per_pair": dict(sorted(Counter(map(len, pair_reviewers.values())).items())),
        "mean_reviews_per_pair": mean(map(len, pair_reviewers.values())) if pair_reviewers else None,
    }


def direct_human_agreement(rows: list[dict]) -> dict:
    """Average cross-reviewer agreement per repeated exact criterion unit.

    Units receive equal weight regardless of whether they have two, three, or
    four reviewers. The reviewer pairs are not independent observations.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[_unit(row)].append(row)
    directions, exact, n_pairs = [], [], 0
    confusion = np.zeros((3, 3), dtype=float)
    for votes in groups.values():
        pairs = [(a, b) for a, b in combinations(votes, 2) if a["reviewer"] != b["reviewer"]]
        if not pairs:
            continue
        directions.append(mean(a["value"] == b["value"] for a, b in pairs))
        exact.append(mean(a["preference"] == b["preference"] for a, b in pairs))
        n_pairs += len(pairs)
        # Symmetrization avoids inventing a privileged annotator/reference side.
        for a, b in pairs:
            i, j = int(2 * a["value"]), int(2 * b["value"])
            confusion[i, j] += 0.5 / len(pairs)
            confusion[j, i] += 0.5 / len(pairs)
    return {
        "repeated_criterion_units": len(directions), "reviewer_pair_comparisons": n_pairs,
        "direction_agreement": mean(directions) if directions else None,
        "five_label_exact_agreement": mean(exact) if exact else None,
        "three_way_confusion_unit_weighted": confusion.tolist(),
        "confusion_order": ["B", "tie", "A"],
        "interpretation": "Equal weight per exact pair/criterion unit; symmetric cross-reviewer "
                          "comparisons. Not an independent-observation confidence interval.",
    }


def reviewer_halves(rows: list[dict], rng: np.random.Generator) -> tuple[list[dict], list[dict], dict]:
    """Globally disjoint reviewers, then restrict both sides to identical units."""
    reviewers = sorted({r["reviewer"] for r in rows})
    rng.shuffle(reviewers)
    left_reviewers = set(reviewers[:len(reviewers) // 2])
    left = [r for r in rows if r["reviewer"] in left_reviewers]
    right = [r for r in rows if r["reviewer"] not in left_reviewers]
    units = {_unit(r) for r in left} & {_unit(r) for r in right}
    left = [r for r in left if _unit(r) in units]
    right = [r for r in right if _unit(r) in units]
    assert not ({r["reviewer"] for r in left} & {r["reviewer"] for r in right})
    assert {_unit(r) for r in left} == {_unit(r) for r in right}
    return left, right, {
        "left_reviewer_ids": sorted(left_reviewers),
        "right_reviewer_ids": sorted(set(reviewers) - left_reviewers),
        "left": coverage(left), "right": coverage(right),
    }


def consensus_agreement(left: list[dict], right: list[dict]) -> dict:
    grouped = []
    for rows in (left, right):
        values = defaultdict(list)
        for row in rows:
            values[_unit(row)].append(row["value"])
        grouped.append({k: mean(v) for k, v in values.items()})
    common = sorted(set(grouped[0]) & set(grouped[1]))
    return {
        "n_units": len(common),
        "direction_agreement": mean(bool(np.sign(grouped[0][k] - .5) == np.sign(grouped[1][k] - .5))
                                    for k in common) if common else None,
        "win_share_mae": mean(abs(grouped[0][k] - grouped[1][k]) for k in common) if common else None,
    }


def summarize_numbers(values: list[float | int | None]) -> dict:
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {"n": len(valid), "mean": mean(valid) if valid else None,
            "p025": float(np.quantile(valid, .025)) if valid else None,
            "median": float(np.median(valid)) if valid else None,
            "p975": float(np.quantile(valid, .975)) if valid else None}


def _metric_summary(runs: list[dict]) -> dict:
    keys = ("spearman", "kendall_tau_b", "pair_order_agreement", "top1_overlap",
            "rank_mae", "normalized_rank_mae", "probability_mae", "probability_rmse")
    return {key: summarize_numbers([run["metrics"].get(key) for run in runs]) for key in keys}


def split_half_stability(rows: list[dict], models: list[str], repeats: int, seed: int) -> dict:
    runs = []
    for repeat in range(repeats):
        left, right, counts = reviewer_halves(rows, np.random.default_rng(seed + repeat))
        left_fit = fit_bradley_terry(aggregate_pairs(left), models=models)
        right_fit = fit_bradley_terry(aggregate_pairs(right), models=models)
        runs.append({"repeat": repeat, "matched_pairs": counts["left"]["pairs"],
                     "matched_subjects": counts["left"]["subjects"],
                     "left_connected": left_fit["connected"], "right_connected": right_fit["connected"],
                     "metrics": compare_rankings(left_fit, right_fit),
                     "criterion_agreement": consensus_agreement(left, right)})
    return {
        "repeats": repeats, "seed": seed,
        "interpretation": "Global random reviewer-disjoint halves; both sides restricted to identical "
                          "plan-pair/exact-criterion units. Percentile range describes split sensitivity, "
                          "not a confidence interval or an attainable accuracy ceiling.",
        "summary": _metric_summary(runs),
        "direction_agreement": summarize_numbers([r["criterion_agreement"]["direction_agreement"]
                                                    for r in runs]),
        "matched_pairs": summarize_numbers([r["matched_pairs"] for r in runs]),
        "runs": runs,
    }


def _nested_samples(rows: list[dict], rng: np.random.Generator) -> list[tuple[str, float | int, list[dict]]]:
    result = []
    fractions = (.1, .25, .5, .75, 1.0)
    for axis, key in (("subjects", "subject"), ("pairs", "pair_id")):
        groups = _group(rows, key)
        ids = sorted(groups)
        rng.shuffle(ids)
        for fraction in fractions:
            n = max(1, int(np.ceil(fraction * len(ids)))) if ids else 0
            result.append((axis, fraction, [r for item in ids[:n] for r in groups[item]]))
    pairs = _group(rows, "pair_id")
    review_order = {}
    for pair_id, votes in pairs.items():
        reviewers = sorted({r["reviewer"] for r in votes})
        rng.shuffle(reviewers)
        review_order[pair_id] = reviewers
    for cap in (1, 2, 3, 4):
        selected = [r for pair_id, votes in pairs.items() for r in votes
                    if r["reviewer"] in review_order[pair_id][:cap]]
        result.append(("reviews_per_pair_cap", cap, selected))
    return result


def density_stability(pool: list[dict], reference: list[dict], models: list[str], *,
                      repeats: int, seed: int, independent: bool) -> dict:
    reference_fit = fit_bradley_terry(aggregate_pairs(reference), models=models)
    groups = defaultdict(list)
    for repeat in range(repeats):
        for axis, density, sampled in _nested_samples(pool, np.random.default_rng(seed + repeat)):
            fit = fit_bradley_terry(aggregate_pairs(sampled), models=models)
            counts = coverage(sampled)
            groups[(axis, density)].append({
                "repeat": repeat, "subjects": counts["subjects"], "pairs": counts["pairs"],
                "review_sessions": counts["review_sessions"],
                "mean_reviews_per_pair": counts["mean_reviews_per_pair"],
                "connected": fit["connected"], "metrics": compare_rankings(reference_fit, fit),
            })
    experiments = []
    for (axis, density), runs in groups.items():
        experiments.append({
            "axis": axis, "density": density, "summary": _metric_summary(runs),
            "coverage": {k: summarize_numbers([r[k] for r in runs])
                         for k in ("subjects", "pairs", "review_sessions", "mean_reviews_per_pair")},
            "connected_fraction": mean(r["connected"] for r in runs), "runs": runs,
        })
    return {
        "reference_type": "independent_reviewer_reference" if independent else "self_overlap_full_reference",
        "interpretation": (
            "All sampled reviewers are disjoint from the fixed reference reviewers. Plan pairs and exact "
            "criteria were matched before sampling; fewer pairs/subjects then produce a schedule shift. "
            "Intervals describe repeated nested subsampling conditional on this one reviewer partition."
            if independent else
            "The full-data reference contains the sampled votes. This measures internal stability and is "
            "optimistic for independent human alignment; at full density it agrees with itself by construction."
        ),
        "pool": coverage(pool), "reference": coverage(reference), "reference_fit": reference_fit,
        "repeats": repeats, "seed": seed, "experiments": experiments,
    }


def human_rank_bootstrap(rows: list[dict], models: list[str], n_bootstrap: int, seed: int) -> dict:
    """Subject-clustered rank uncertainty for the entire human reference."""
    pairs = aggregate_pairs(rows)
    subjects = sorted({r["subject"] for r in pairs})
    point = fit_bradley_terry(pairs, models=models)
    rng = np.random.default_rng(seed)
    rank_values, score_values = defaultdict(list), defaultdict(list)
    top_counts, valid = Counter(), 0
    for _ in range(n_bootstrap):
        multiplicity = Counter(rng.choice(subjects, size=len(subjects), replace=True).tolist())
        sampled = [{**r, "weight": r["weight"] * multiplicity[r["subject"]]}
                   for r in pairs if multiplicity[r["subject"]]]
        fit = fit_bradley_terry(sampled, models=models)
        if not fit["connected"] or not fit["converged"]:
            continue
        valid += 1
        best = min(r["rank"] for r in fit["leaderboard"])
        for row in fit["leaderboard"]:
            rank_values[row["model"]].append(row["rank"])
            score_values[row["model"]].append(row["score"])
            top_counts[row["model"]] += row["rank"] == best
    return {
        "method": "subject_cluster_percentile_bootstrap", "seed": seed,
        "n_bootstrap": n_bootstrap, "valid_bootstrap_fits": valid,
        "leaderboard": [{**row, "rank_interval": summarize_numbers(rank_values[row["model"]]),
                         "score_interval": summarize_numbers(score_values[row["model"]]),
                         "top1_frequency": top_counts[row["model"]] / valid if valid else None}
                        for row in point["leaderboard"]],
        "interpretation": "95% percentile intervals conditional on connected, converged resamples. "
                          "Resample student contexts with replacement; keep all reviews and plan pairs "
                          "within a context. Does not model remaining repeated-tutor dependence.",
    }


def run_analysis(dataset: Dataset, subjects: list[str], output_dir: Path, *,
                 seed: int = 20260910, repeats: int = 200, bootstrap: int = 500,
                 split_name: str = "validation") -> dict:
    if repeats < 1 or bootstrap < 0:
        raise ValueError("repeats must be positive and bootstrap nonnegative")
    if len(subjects) != len(set(subjects)) or set(subjects) - set(dataset.pretests):
        raise ValueError("Requested subjects must be unique and present in this dataset")
    rows = human_criterion_rows(dataset, subjects)
    if not rows:
        raise ValueError("No human pairwise votes in the requested subject split")
    models = coverage(rows)["models"]
    pairs = aggregate_pairs(rows)
    left, right, fixed_halves = reviewer_halves(rows, np.random.default_rng(seed))
    result = {
        "schema_version": 1, "instrument": dataset.instrument, "split": split_name,
        "dataset_sha256": dataset.fingerprint, "subject_ids": sorted(subjects),
        "seed": seed, "aggregation": "Direction wins: A=1, tie=.5, B=0. Average reviewers within exact "
                   "criterion, variants within criterion family, then the eight criterion families equally "
                   "within each unique subject/plan pair. Rank authoring "
                   "model presets using regularized Bradley-Terry; criterion votes are not independent matches.",
        "coverage": coverage(rows), "leaderboard": fit_bradley_terry(pairs, models=models),
        "human_subject_bootstrap": human_rank_bootstrap(rows, models, bootstrap, seed),
        "criterion_leaderboards": {
            criterion: fit_bradley_terry(aggregate_pairs(subset), models=models)
            for criterion, subset in sorted(_group(rows, "criterion").items())},
        "direct_human_agreement": direct_human_agreement(rows),
        "reviewer_split_half": split_half_stability(rows, models, repeats, seed),
        "fixed_reviewer_halves": fixed_halves,
        "fixed_half_subject_bootstrap": bootstrap_alignment(
            aggregate_pairs(left), aggregate_pairs(right), models=models, n_bootstrap=bootstrap, seed=seed),
        "density_self_overlap": density_stability(rows, rows, models, repeats=repeats, seed=seed,
                                                  independent=False),
        "density_independent_reviewers": density_stability(left, right, models, repeats=repeats,
                                                           seed=seed, independent=True),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split_name}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssot", type=Path, default=Path("studentbench_data_ssot"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("development", "validation", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.split == "test" and not manifest.get("selection_locked", False):
        raise ValueError("Test reference stays sealed until manifest.selection_locked is true")
    dataset = Dataset(args.ssot, manifest.get("instrument", "gre_quant"))
    if manifest.get("dataset_sha256") != dataset.fingerprint:
        raise ValueError("Frozen manifest and current SSOT fingerprints differ")
    result = run_analysis(dataset, manifest["splits"][args.split], args.output,
                          seed=manifest.get("seed", 20260910), repeats=args.repeats,
                          bootstrap=args.bootstrap, split_name=args.split)
    print(json.dumps({"output": str(args.output / (args.split + ".json")),
                      "coverage": result["coverage"],
                      "reviewer_split_half": result["reviewer_split_half"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
