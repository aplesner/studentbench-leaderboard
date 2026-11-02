"""Offline paired uncertainty for differences in model-ranking alignment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .data import Dataset, digest
from .ranking import aggregate_pairs, compare_rankings, fit_bradley_terry
from .study import call_path, catalog, write_json
from .study_analysis import (
    _call_votes, _human_rows, _key, criterion_family, load_derived_judges,
    materialize_judge, scheduled_judges,
)


METRICS = ("kendall_tau_b", "spearman", "pair_order_agreement", "rank_mae")
BASELINE = "enhanced_terra_high"
EXAMPLE = "terra_high_one_example"
INTERPRETATION = (
    "These paired percentile intervals resample the same complete human-reference subject cohort for every "
    "judge. All comparisons within a sampled subject retain its common multiplicity; human and both AI "
    "leaderboards are refitted on each draw. Intervals condition on connected, converged fits and defined "
    "metrics, and on the fixed observed judgments and one selected development example. They do not capture "
    "new judge generations, alternative examples, backend changes, or all repeated-reviewer dependence. "
    "Delta is candidate minus cached Terra. Positive is better except rank MAE, where negative is better. "
    "Resampling sign proportions are descriptive, not p-values or probabilities of future superiority. "
    "Provider/panel comparisons are multiple exploratory comparisons without multiplicity adjustment. "
    "The existing validation cohort was reused after the original test was inspected."
)


def _pair_rows(rows: list[dict], *, label: str) -> list[dict]:
    """Accept already aggregated pairs, so bootstrap copies cannot be averaged away."""
    result, seen = [], set()
    for original in rows:
        row = dict(original)
        for key in ("subject", "pair_id", "model_a", "model_b"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{label} needs a nonempty {key}")
        key = (row["subject"], row["pair_id"])
        if key in seen:
            raise ValueError(f"{label} has duplicate pairs; aggregate criteria before bootstrapping")
        seen.add(key)
        if row.get("value") is None:
            continue
        if (type(row["value"]) not in (int, float) or not isfinite(row["value"])
                or not 0 <= row["value"] <= 1):
            raise ValueError(f"{label} pair value must be finite in [0, 1]")
        weight = row.get("weight", 1.0)
        if type(weight) not in (int, float) or not isfinite(weight) or weight <= 0:
            raise ValueError(f"{label} pair weights must be positive finite numbers")
        row["weight"] = float(weight)
        if row["model_b"] < row["model_a"]:
            row["model_a"], row["model_b"] = row["model_b"], row["model_a"]
            row["value"] = 1 - row["value"]
        result.append(row)
    return sorted(result, key=lambda row: (row["subject"], row["pair_id"]))


def _sample(rows: list[dict], counts: dict[str, int]) -> list[dict]:
    # Multiplication, not post-sampling aggregation, preserves repeated clusters.
    return [{**row, "weight": row["weight"] * counts[row["subject"]]}
            for row in rows if counts[row["subject"]]]


def _delta(reference: dict, candidate: dict) -> dict[str, float | None]:
    return {metric: candidate[metric] - reference[metric]
            if candidate[metric] is not None and reference[metric] is not None else None
            for metric in METRICS}


def paired_bootstrap_deltas(human_pairs: list[dict], baseline_pairs: list[dict],
                            candidates: dict[str, list[dict]], *, models: list[str],
                            n_bootstrap: int = 500, seed: int = 0,
                            confidence: float = .95, ridge: float = 1.0) -> dict:
    """Compare candidates on identical full-human-cohort subject resamples.

    Rows must already have one observation per subject/plan-pair. Candidate
    omissions stay omitted on each draw; they never shrink the human cohort or
    become ties. New unreviewed pairs are outside this matched-cohort analysis.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 0 or not 0 < confidence < 1:
        raise ValueError("Use nonnegative integer bootstrap count and confidence in (0, 1)")
    if type(seed) is not int or seed < 0:
        raise ValueError("Use a nonnegative integer seed")
    if not candidates or any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("Named candidates are required")
    human = _pair_rows(human_pairs, label="Human")
    baseline = _pair_rows(baseline_pairs, label="Baseline")
    candidate_rows = {name: _pair_rows(rows, label=name) for name, rows in sorted(candidates.items())}
    if not human:
        raise ValueError("Human pairs are required")
    identity = {(row["subject"], row["pair_id"]): (row["model_a"], row["model_b"]) for row in human}
    for name, rows in [("baseline", baseline), *candidate_rows.items()]:
        for row in rows:
            if identity.get((row["subject"], row["pair_id"])) != (row["model_a"], row["model_b"]):
                raise ValueError(f"{name} has a pair or model identity outside the human reference")
    subjects = sorted({row["subject"] for row in human})
    universe = sorted(set(models))
    point_human = fit_bradley_terry(human, models=universe, ridge=ridge)
    point_baseline = fit_bradley_terry(baseline, models=universe, ridge=ridge)
    baseline_metrics = compare_rankings(point_human, point_baseline)
    comparisons, samples = {}, {}
    for name, rows in candidate_rows.items():
        fit = fit_bradley_terry(rows, models=universe, ridge=ridge)
        metrics = compare_rankings(point_human, fit)
        comparisons[name] = {
            "baseline_variant": BASELINE, "candidate_variant": name,
            "coverage": {"human_pairs": len(human), "baseline_pairs": len(baseline),
                         "candidate_pairs": len(rows), "n_subjects": len(subjects),
                         "baseline_scored_subjects": len({r["subject"] for r in baseline}),
                         "candidate_scored_subjects": len({r["subject"] for r in rows})},
            "point_fit_status": {"human": {k: point_human[k] for k in ("connected", "converged")},
                                 "baseline": {k: point_baseline[k] for k in ("connected", "converged")},
                                 "candidate": {k: fit[k] for k in ("connected", "converged")}},
            "metrics": {metric: {"baseline": baseline_metrics[metric], "candidate": metrics[metric],
                                  "delta": _delta(baseline_metrics, metrics)[metric],
                                  "higher_is_better": metric != "rank_mae"} for metric in METRICS},
            "n_valid_fits": 0, "n_disconnected_draws": 0, "n_nonconverged_draws": 0,
            "invalid_draw_counts": {source: {"disconnected": 0, "nonconverged": 0}
                                    for source in ("human", "baseline", "candidate")},
            "input_pairs_sha256": digest(rows),
        }
        samples[name] = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_bootstrap):
        counts = Counter(rng.choice(subjects, size=len(subjects), replace=True).tolist())
        resamples.append([counts[subject] for subject in subjects])
        human_fit = fit_bradley_terry(_sample(human, counts), models=universe, ridge=ridge)
        baseline_fit = fit_bradley_terry(_sample(baseline, counts), models=universe, ridge=ridge)
        baseline_scores = compare_rankings(human_fit, baseline_fit)
        for name, rows in candidate_rows.items():
            # Empty candidates remain unavailable; no arbitrary global ranks.
            candidate_fit = fit_bradley_terry(_sample(rows, counts), models=universe, ridge=ridge)
            fits = {"human": human_fit, "baseline": baseline_fit, "candidate": candidate_fit}
            summary = comparisons[name]
            disconnected = any(not fit["connected"] for fit in fits.values())
            nonconverged = any(not fit["converged"] for fit in fits.values())
            summary["n_disconnected_draws"] += disconnected
            summary["n_nonconverged_draws"] += nonconverged
            for source, fit in fits.items():
                summary["invalid_draw_counts"][source]["disconnected"] += not fit["connected"]
                summary["invalid_draw_counts"][source]["nonconverged"] += not fit["converged"]
            if disconnected or nonconverged:
                continue
            summary["n_valid_fits"] += 1
            differences = _delta(baseline_scores, compare_rankings(human_fit, candidate_fit))
            for metric, value in differences.items():
                if value is not None:
                    samples[name][metric].append(value)
    alpha = (1 - confidence) / 2
    for name, summary in comparisons.items():
        for metric, entry in summary["metrics"].items():
            values = np.asarray(samples[name][metric], dtype=float)
            low, high = np.quantile(values, [alpha, 1 - alpha]).tolist() if len(values) else (None, None)
            entry.update(low=low, high=high, n_valid=len(values),
                         proportion_delta_gt_zero=float(np.mean(values > 0)) if len(values) else None,
                         proportion_delta_lt_zero=float(np.mean(values < 0)) if len(values) else None,
                         proportion_delta_equal_zero=float(np.mean(values == 0)) if len(values) else None,
                         proportion_improved=float(np.mean(values > 0 if entry["higher_is_better"] else values < 0))
                         if len(values) else None)
    return {"schema_version": 1, "method": "paired_subject_percentile_bootstrap_of_alignment_difference",
            "baseline_variant": BASELINE, "n_bootstrap": n_bootstrap, "confidence": confidence,
            "seed": seed, "ridge": ridge, "n_subjects": len(subjects), "models": universe,
            "resample_counts_sha256": digest(resamples), "subject_order_sha256": digest(subjects),
            "human_pairs_sha256": digest(human), "baseline_pairs_sha256": digest(baseline),
            "comparisons": comparisons, "interpretation": INTERPRETATION}


def _load(study: Path, filename: str):
    return json.loads((study / filename).read_text())


def _round_rows(study: Path, name: str, jobs_file: str, dataset: Dataset,
                manifest: dict, variants: dict, exclusions: set[str]) -> dict:
    """Reconstruct the completed main analysis without making or rerunning calls."""
    jobs = _load(study, jobs_file)
    analysis = _load(study, name + "_analysis.json")
    scheduled = {(j["variant"], j["pair_id"], j.get("draw", 0)) for j in jobs}
    if len(scheduled) != len(jobs) or not jobs:
        raise ValueError("Paired comparison needs a unique nonempty job schedule")
    from .extension_terminal import load_terminal_failures
    terminal = load_terminal_failures(study, jobs, ssot=dataset.root, scope_only=True)
    if (analysis["jobs_sha256"] != digest(jobs) or analysis["dataset_sha256"] != dataset.fingerprint
            or analysis["split"] != "validation"):
        raise ValueError("Main analysis provenance differs from the paired-comparison schedule")
    pairs = catalog(manifest)
    ids = sorted({j["pair_id"] for j in jobs})
    if any(pairs[p]["split"] != "validation" or not pairs[p]["human_reviewed"] for p in ids):
        raise ValueError("Extension paired deltas require human-reviewed validation pairs only")
    units = {p: [{"criterion": c["id"] + "|" + c["scale_id"],
                  "criterion_family": criterion_family(c["id"], dataset)}
                 for c in dataset.rubric(pairs[p]["grading_id"])] for p in ids}
    expected = {(p, c["criterion"]) for p in ids for c in units[p]}
    human = aggregate_pairs([row for row in _human_rows(dataset, set(manifest["splits"]["validation"]), pairs)
                             if _key(row) in expected])
    calls = {}
    for job in jobs:
        path = call_path(study, job)
        if not path.exists():
            if job["variant"] not in exclusions and (job["variant"], job["pair_id"], job.get("draw", 0)) not in terminal:
                raise ValueError("Executable schedule is incomplete; wait for calls or verify terminal failures before paired deltas")
            continue
        record = json.loads(path.read_text())
        key = (job["variant"], job["pair_id"], job.get("draw", 0))
        if key != (record.get("variant"), record.get("pair_id"), record.get("draw", 0)):
            raise ValueError("Cached paired-comparison call identity changed")
        entry = pairs[job["pair_id"]]
        if (record.get("dataset_sha256") != dataset.fingerprint
                or any(record.get(k) != entry[k] for k in ("subject", "plan_ids", "model_a", "model_b"))
                or set(_call_votes(record)) != {c["criterion"] for c in units[job["pair_id"]]}):
            raise ValueError("Cached paired-comparison call differs from frozen data/criteria")
        calls[key] = record
    definitions = scheduled_judges(variants, scheduled, load_derived_judges(study, variants))
    actual = {entry["variant"]: entry for entry in analysis["judges"]}
    if set(actual) != {entry["variant"] for entry in definitions}:
        raise ValueError("Main analysis judge definitions differ from paired-comparison definitions")
    rows, coverage = {}, {}
    for definition in definitions:
        variant = definition["variant"]
        materialized, counts = materialize_judge(definition, ids, pairs, units, calls)
        rows[variant], coverage[variant] = aggregate_pairs(materialized), counts
        if (actual[variant]["members"] != [list(m) for m in definition["members"]]
                or actual[variant].get("weights") != definition.get("weights")
                or counts["scored_criteria"] != actual[variant]["coverage"]["scored_criteria"]):
            raise ValueError("Main analysis panel members, weights, or scored coverage changed")
    return {"human": human, "judges": rows, "coverage": coverage, "analysis": analysis,
            "pair_ids": ids, "jobs_sha256": digest(jobs), "terminal_failures": terminal}


def run_paired_deltas(study: Path, *, ssot: Path = Path("studentbench_data_ssot"),
                      n_bootstrap: int = 500, seed: int | None = None,
                      output: Path | None = None, include_provider: bool = True) -> dict:
    """Write completed example contrasts, optionally including provider contrasts.

    The example-only phase never reads incomplete provider outcomes. The later
    default run writes all phases with exactly the same example resamples.
    """
    if type(include_provider) is not bool:
        raise ValueError("include_provider must be a boolean")
    study = Path(study)
    manifest = _load(study, "manifest.json")
    if manifest.get("evaluation_role") != "validation_extension_after_original_test_exposure":
        raise ValueError("This wrapper is for the frozen validation extension only")
    dataset = Dataset(ssot, manifest["instrument"])
    if dataset.fingerprint != manifest["dataset_sha256"]:
        raise ValueError("Paired comparison SSOT differs from the frozen manifest")
    variants = {v["variant"]: v for v in _load(study, "variants.json")}
    availability = _load(study, "availability.json") if (study / "availability.json").exists() else {}
    exclusions = set(availability.get("exclusions", {}))
    example = _round_rows(study, "example_validation", "example_jobs.json", dataset, manifest, variants, exclusions)
    phases = {"example_validation": example}
    candidates = {EXAMPLE: example["judges"][EXAMPLE]}
    if include_provider:
        provider = _round_rows(study, "provider_validation", "provider_jobs.json", dataset, manifest, variants, exclusions)
        if (provider["pair_ids"] != example["pair_ids"] or provider["human"] != example["human"]
                or provider["judges"][BASELINE] != example["judges"][BASELINE]):
            raise ValueError("Provider and example comparisons need identical pairs, human reference, and cached Terra")
        phases["provider_validation"] = provider
        candidates.update({name: rows for name, rows in provider["judges"].items() if name != BASELINE})
    models = sorted({plan["ai_model_preset_id"] for plan in dataset.plans.values()})
    result = paired_bootstrap_deltas(example["human"], example["judges"][BASELINE], candidates,
                                     models=models, n_bootstrap=n_bootstrap,
                                     seed=manifest["seed"] if seed is None else seed)
    from .extension_terminal import terminal_failure_summary
    terminal = {key: value for phase in phases.values() for key, value in phase["terminal_failures"].items()}
    result.update(evaluation_role=manifest["evaluation_role"],
                  dataset_sha256=dataset.fingerprint,
                  complete_phases=list(phases),
                  pending_phases=[] if include_provider else ["provider_validation"],
                  status=("complete_with_terminal_missing_outcomes" if terminal else "complete") if include_provider
                         else "example_complete_provider_pending",
                  terminal_failures=terminal_failure_summary(study, terminal),
                  main_analysis_sha256={name: hashlib.sha256((study / (name + "_analysis.json")).read_bytes()).hexdigest()
                                        for name in phases},
                  jobs_sha256={name: phase["jobs_sha256"] for name, phase in phases.items()})
    for name, comparison in result["comparisons"].items():
        source = phases["example_validation" if name == EXAMPLE else "provider_validation"]
        comparison.update(role="primary_example" if name == EXAMPLE else "exploratory_provider",
                          source_analysis=source["analysis"]["name"],
                          criterion_coverage=source["coverage"][name])
        original = {j["variant"]: j for j in source["analysis"]["judges"]}
        for metric, values in comparison["metrics"].items():
            for field, variant in (("baseline", BASELINE), ("candidate", name)):
                expected = original[variant]["metrics"][metric]
                actual = values[field]
                if (expected is None) != (actual is None) or (expected is not None and abs(expected - actual) > 1e-10):
                    raise ValueError("Paired point metric differs from the completed main analysis")
    write_json(output or study / "paired_deltas.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/ranking-extension-20260911"))
    parser.add_argument("--ssot", type=Path, default=Path("studentbench_data_ssot"))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    run_paired_deltas(args.study, ssot=args.ssot, n_bootstrap=args.bootstrap)
