"""Offline, model-level analysis of explicitly scheduled ranking-study calls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import tomllib

import numpy as np

from .data import Dataset, digest
from .human_analysis import human_criterion_rows, summarize_numbers
from .ranking import aggregate_pairs, bootstrap_alignment, compare_rankings, fit_bradley_terry
from .study import DEFAULT_STUDY, call_path, catalog, write_json


METRICS = ("spearman", "kendall_tau_b", "pair_order_agreement", "top1_overlap",
           "rank_mae", "normalized_rank_mae", "probability_mae", "probability_rmse")

# The original study used this exact TOML document to describe the built-in
# compositions below. It predates executable extension recipes; accepting only
# the complete known document avoids silently skipping malformed new recipes.
_HISTORICAL_PANEL_DESCRIPTION = {
    "judge_7": {"strategy": "balanced_order_soft", "member_variant": "enhanced_luna_high",
                "draws": [0, 1], "aggregation": "mean_win_share", "missing": "abstain_if_any_member_missing"},
    "judge_8": {"strategy": "panel_soft", "member_models": ["luna", "terra", "sol", "astra"],
                "efforts": ["low", "high"], "aggregation": "mean_win_share",
                "missing": "abstain_if_any_member_missing"},
    "judge_9": {"strategy": "panel_strict", "member_models": ["luna", "terra", "sol", "astra"],
                "efforts": ["low", "high"], "aggregation": "strict_direction_majority",
                "missing": "included_in_majority_denominator"},
}


def criterion_family(identifier: str, dataset: Dataset) -> str:
    match = re.search(r"_q(\d{2})(?:_|$)", identifier)
    return "q" + match.group(1) if match else dataset.questions[identifier]["rubric_question_id"]


def _key(row: dict) -> tuple[str, str]:
    return row["pair_id"], row["criterion"]


def _human_rows(dataset: Dataset, subjects: set[str], pairs: dict) -> list[dict]:
    rows = human_criterion_rows(dataset, subjects)
    for row in rows:
        plans = [row["plan_a"], row["plan_b"]]
        row["pair_id"] = digest([row["subject"], *plans])[:20]
        entry = pairs.get(row["pair_id"])
        if entry is None or entry["plan_ids"] != plans or entry["subject"] != row["subject"]:
            raise ValueError("Human pair does not match the frozen study catalog")
        row["criterion_family"] = criterion_family(row["criterion"].split("|", 1)[0], dataset)
    return rows


def combine_votes(values: list[float | None], method: str,
                  weights: list[float] | None = None) -> float | None:
    """Averaged win share or strict majority; missing votes never become ties."""
    if weights is not None:
        if (len(weights) != len(values) or any(type(w) not in (int, float) or not isfinite(w) or w <= 0
                                              for w in weights)):
            raise ValueError("Vote weights must be positive finite numbers matching all members")
        if method != "soft":
            raise ValueError("Explicit member weights are supported only for soft aggregation")
    if method == "soft":
        if not values or any(v is None for v in values):
            return None
        return float(np.average(values, weights=weights)) if weights is not None else float(np.mean(values))
    if method == "strict":
        counts = Counter(value for value in values if value is not None)
        winners = [value for value, count in counts.items() if count > len(values) / 2]
        return winners[0] if winners else None
    raise ValueError(f"Unknown vote combination: {method}")


def load_derived_judges(study: Path, variants: dict) -> list[dict]:
    """Read optional prespecified offline recipes, never model outputs.

    The availability amendment is a separate file and freeze: it does not
    replace or silently reduce the membership of an original recipe. Read only
    these explicitly supported filenames, not arbitrary experimental TOMLs.
    """
    output, names, judge_ids = [], set(), set()
    for filename in ("derived_judges.toml", "derived_judges_available.toml"):
        for definition in _load_derived_judge_file(Path(study) / filename, variants):
            if definition["variant"] in names or definition["judge_id"] in judge_ids:
                raise ValueError("Derived-judge names and IDs must be distinct across recipe files")
            output.append(definition)
            names.add(definition["variant"])
            judge_ids.add(definition["judge_id"])
    return output


def _load_derived_judge_file(path: Path, variants: dict) -> list[dict]:
    """Every member references one declared variant and integer draw."""
    if not path.exists():
        return []
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = path.with_name(path.stem + "_manifest.json")
    if frozen.exists() and json.loads(frozen.read_text()).get("file_sha256") != file_hash:
        raise ValueError("Frozen derived-judge recipe fingerprint changed")
    data = tomllib.loads(path.read_text())
    if (path.name == "derived_judges.toml" and data == _HISTORICAL_PANEL_DESCRIPTION
            and json.dumps(data, sort_keys=True) == json.dumps(_HISTORICAL_PANEL_DESCRIPTION, sort_keys=True)):
        return []  # scheduled_judges reconstructs these historical compositions.
    if set(data) != {"schema_version", "judges"} or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Invalid derived-judge file schema")
    if not isinstance(data["judges"], list) or not data["judges"]:
        raise ValueError("Derived-judge file needs at least one recipe")
    output, names, judge_ids = [], set(), set()
    for recipe in data["judges"]:
        required = {"judge_id", "variant", "label", "aggregation", "members"}
        if (not isinstance(recipe, dict) or not required <= set(recipe)
                or set(recipe) - required - {"description"}):
            raise ValueError("Invalid derived-judge recipe fields")
        for key in ("judge_id", "variant"):
            if not isinstance(recipe[key], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", recipe[key]):
                raise ValueError(f"Invalid derived-judge {key}")
        if (recipe["variant"] in names or recipe["judge_id"] in judge_ids or recipe["variant"] in variants
                or recipe["judge_id"] in {v.get("judge_id") for v in variants.values()}):
            raise ValueError("Derived-judge names and IDs must be distinct from existing recipes/variants")
        if not isinstance(recipe["label"], str) or not recipe["label"].strip():
            raise ValueError("Derived judges need a nonempty label")
        if recipe["aggregation"] != "soft":
            raise ValueError("TOML-derived panels currently require soft fractional-win aggregation")
        if not isinstance(recipe["members"], list) or len(recipe["members"]) < 2:
            raise ValueError("Derived panels need at least two members")
        members, weights, weighted = [], [], []
        for member in recipe["members"]:
            if (not isinstance(member, dict) or not {"variant", "draw"} <= set(member)
                    or set(member) - {"variant", "draw", "weight"}):
                raise ValueError("Invalid derived member fields")
            variant, draw, weight = member["variant"], member["draw"], member.get("weight", 1.0)
            if not isinstance(variant, str) or variant not in variants:
                raise ValueError("Derived member refers to an undeclared variant")
            if type(draw) is not int or draw < 0:
                raise ValueError("Derived member draw must be a nonnegative integer")
            if (variant, draw) in members:
                raise ValueError("Duplicate derived member variant/draw")
            if type(weight) not in (int, float) or not isfinite(weight) or weight <= 0:
                raise ValueError("Derived member weights must be positive finite numbers")
            members.append((variant, draw)); weights.append(float(weight))
            weighted.append({"variant": variant, "draw": draw, "weight": float(weight),
                             "family": variants[variant].get("family")})
        total = sum(weights)
        if not isfinite(total):
            raise ValueError("Total derived member weight must be finite")
        for member in weighted:
            member["normalized_weight"] = member["weight"] / total
        recipe_hash = digest(recipe)
        member_hash = digest([variants[variant] for variant, _draw in members])
        output.append({"variant": recipe["variant"], "judge_id": recipe["judge_id"],
                       "judge_version": f"{recipe['judge_id']}_{recipe_hash[:12]}_{member_hash[:12]}",
                       "label": recipe["label"], "members": members, "aggregation": "soft",
                       "weights": weights, "normalized_weights": [w / total for w in weights],
                       "weighted_members": weighted, "derived_definition": recipe,
                       "definition_sha256": recipe_hash, "definition_file": path.name,
                       "definition_file_sha256": file_hash,
                       "member_configuration_sha256": member_hash,
                       "missing_member_policy": "Any missing call, criterion, or member abstention makes that criterion missing."})
        names.add(recipe["variant"]); judge_ids.add(recipe["judge_id"])
    return output


def _call_votes(record: dict) -> dict[str, float | None]:
    values = {}
    for criterion in record["criteria"]:
        score = record["normalized_ratings"][criterion["id"]]["score"]
        if score is not None and (isinstance(score, bool) or score not in (-2, -1, 0, 1, 2)):
            raise ValueError("Saved normalized score is outside the signed five-label scale")
        values[criterion["id"] + "|" + criterion["scale_id"]] = (
            None if score is None else .5 if score == 0 else 1.0 if score > 0 else 0.0)
    return values


def scheduled_judges(variants: dict, scheduled: set[tuple[str, str, int]],
                     derived: list[dict] | None = None) -> list[dict]:
    """Define strategies from declared jobs only, without reading labels."""
    names = sorted({variant for variant, _pair, _draw in scheduled})
    definitions = []
    for name in names:
        info = variants[name]
        label = info.get("label", f"{name}: {info.get('model', '')}, {info.get('reasoning_effort', '')}")
        definitions.append({"variant": name, "label": label, "members": [(name, 0)], "aggregation": "soft"})
        if any(variant == name and draw == 1 for variant, _pair, draw in scheduled):
            for method in ("soft", "strict"):
                definitions.append({"variant": name + "_balanced" + ("_strict" if method == "strict" else ""),
                                    "label": f"{name}, both presentation orders ({method})",
                                    "members": [(name, 0), (name, 1)], "aggregation": method})
    for effort in ("low", "high"):
        members = [(f"{model}_{effort}", 0) for model in ("luna", "terra", "sol", "astra")]
        if all(name in names for name, _draw in members):
            for method in ("soft", "strict"):
                definitions.append({"variant": f"panel_{effort}" + ("_strict" if method == "strict" else ""),
                                    "label": f"Four-model panel, {effort} effort ({method})",
                                    "members": members, "aggregation": method})
    pair_ids = {pair_id for _variant, pair_id, _draw in scheduled}
    existing = {definition["variant"] for definition in definitions}
    for definition in derived or []:
        if definition["variant"] in existing:
            raise ValueError("Derived panel name conflicts with a built-in scheduled strategy")
        # A recipe is eligible only when every member draw was explicitly
        # scheduled on every round pair. Missing completed calls are handled
        # separately as missing outcomes, never filled by fewer members.
        if pair_ids and all((variant, pair_id, draw) in scheduled for variant, draw in definition["members"] for pair_id in pair_ids):
            definitions.append(definition)
            existing.add(definition["variant"])
    return definitions


def materialize_judge(definition: dict, pair_ids: list[str], pairs: dict,
                      units: dict[str, list[dict]], calls: dict) -> tuple[list[dict], dict]:
    """Combine existing normalized calls with strict completeness bookkeeping."""
    rows = []
    used = {}
    completed_pairs = 0
    expected_criteria = sum(len(units[pair_id]) for pair_id in pair_ids)
    direction_changed = 0
    compared_criteria = 0
    for pair_id in pair_ids:
        keys = [(variant, pair_id, draw) for variant, draw in definition["members"]]
        records = [calls.get(key) for key in keys]
        for key, record in zip(keys, records):
            if record is not None:
                used[key] = record
        complete = all(record is not None for record in records)
        completed_pairs += complete
        votes = [_call_votes(record) for record in records] if complete else []
        entry = pairs[pair_id]
        for criterion in units[pair_id]:
            values = [vote.get(criterion["criterion"]) for vote in votes]
            value = combine_votes(values, definition["aggregation"], definition.get("weights")) if complete else None
            if complete and all(v is not None for v in values) and len(values) > 1:
                compared_criteria += 1
                direction_changed += len(set(values)) > 1
            rows.append({"subject": entry["subject"], "pair_id": pair_id,
                         "model_a": entry["model_a"], "model_b": entry["model_b"],
                         **criterion, "value": value})
    scored = [row for row in rows if row["value"] is not None]
    expected_calls = len(pair_ids) * len(definition["members"])
    stats = {"scheduled_pairs": len(pair_ids), "completed_pairs": completed_pairs,
             "scored_pairs": len({r["pair_id"] for r in scored}),
             "requested_criteria": expected_criteria, "scored_criteria": len(scored),
             "criterion_coverage": len(scored) / expected_criteria if expected_criteria else 0,
             "pair_coverage": len({r["pair_id"] for r in scored}) / len(pair_ids) if pair_ids else 0,
             "expected_calls": expected_calls, "n_calls": len(used),
             "missing_calls": expected_calls - len(used),
             "disagreeing_member_criteria": direction_changed, "compared_member_criteria": compared_criteria,
             "estimated_usd": sum(r.get("estimated_usd", 0) for r in used.values()),
             "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) or 0 for r in used.values()),
             "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) or 0 for r in used.values())}
    return rows, stats


def _subject_balanced(rows: list[dict]) -> list[dict]:
    counts = Counter(row["subject"] for row in rows)
    scale = len(rows) / len(counts) if counts else 1
    return [{**row, "weight": scale / counts[row["subject"]]} for row in rows]


def density_analysis(judge_pairs: list[dict], human_fit: dict, models: list[str], *,
                     repeats: int = 100, seed: int = 0) -> dict:
    """Nested scheduled-pair subsets against one fixed human reference."""
    fractions = (.25, .5, .75, 1.0)
    grouped = defaultdict(list)
    subjects_total = len({row["subject"] for row in judge_pairs})
    for repeat in range(repeats):
        ordered = np.random.default_rng(seed + repeat).permutation(len(judge_pairs)).tolist()
        for fraction in fractions:
            count = max(1, int(np.ceil(fraction * len(ordered)))) if ordered else 0
            sample = [judge_pairs[index] for index in ordered[:count]]
            fit = fit_bradley_terry(sample, models=models)
            grouped[fraction].append({"repeat": repeat, "n_pairs": len(sample),
                                      "n_subjects": len({row["subject"] for row in sample}),
                                      "model_coverage": fit["model_coverage"],
                                      "connected": fit["connected"], "converged": fit["converged"],
                                      "metrics": compare_rankings(human_fit, fit)})
    return {"method": "nested_seeded_pair_subsampling", "repeats": repeats, "seed": seed,
            "n_available_pairs": len(judge_pairs), "n_available_subjects": subjects_total,
            "reference": "fixed scheduled human leaderboard", "experiments": [
                {"fraction": fraction, "summary": {
                    metric: summarize_numbers([run["metrics"][metric] for run in runs]) for metric in METRICS},
                 "coverage": {key: summarize_numbers([run[key] for run in runs])
                              for key in ("n_pairs", "n_subjects", "model_coverage")},
                 "connected_fraction": sum(run["connected"] for run in runs) / len(runs),
                 "valid_fit_fraction": sum(run["connected"] and run["converged"] for run in runs) / len(runs),
                 "runs": runs} for fraction, runs in grouped.items()],
            "interpretation": "Ranges describe sampling sensitivity, not confidence intervals. Fewer pairs can also reduce subject coverage and change the opponent schedule; the human reference remains fixed."}


def augmentation_analysis(reviewed_pairs: list[dict], unreviewed_pairs: list[dict],
                          human_fit: dict, models: list[str], *, repeats: int = 100,
                          seed: int = 0) -> dict:
    """Hold reviewed AI comparisons fixed while adding nested unreviewed pairs."""
    grouped = defaultdict(list)
    for repeat in range(repeats):
        ordered = np.random.default_rng(seed + repeat).permutation(len(unreviewed_pairs)).tolist()
        for fraction in (0.0, .25, .5, .75, 1.0):
            count = int(np.ceil(fraction * len(ordered)))
            sample = reviewed_pairs + [unreviewed_pairs[index] for index in ordered[:count]]
            fit = fit_bradley_terry(sample, models=models)
            grouped[fraction].append({"repeat": repeat, "n_pairs": len(sample),
                                      "n_added_unreviewed_pairs": count,
                                      "n_subjects": len({row["subject"] for row in sample}),
                                      "connected": fit["connected"], "converged": fit["converged"],
                                      "metrics": compare_rankings(human_fit, fit)})
    return {"method": "fixed_reviewed_pairs_plus_nested_unreviewed_augmentation",
            "n_reviewed_pairs": len(reviewed_pairs), "n_available_unreviewed_pairs": len(unreviewed_pairs),
            "repeats": repeats, "seed": seed, "experiments": [
                {"fraction": fraction, "summary": {
                    metric: summarize_numbers([run["metrics"][metric] for run in runs]) for metric in METRICS},
                 "coverage": {key: summarize_numbers([run[key] for run in runs])
                              for key in ("n_pairs", "n_subjects", "n_added_unreviewed_pairs")},
                 "valid_fit_fraction": sum(run["connected"] and run["converged"] for run in runs) / len(runs),
                 "runs": runs} for fraction, runs in grouped.items()],
            "interpretation": "Every draw retains all scored AI comparisons for human-reviewed pairs, then adds distinct pairs without human reviews. Alignment uses the same fixed human reference. Additional comparisons change the opponent schedule and weighting; they do not supply independent human verification of those new pairs."}


def analyze_round(study: Path, name: str, jobs_path: Path, bootstrap: int = 200,
                  density: bool = False, density_repeats: int = 100,
                  ssot: Path = Path("studentbench_data_ssot")) -> dict:
    """Analyze cached calls; accesses human labels only for the jobs' split.

    The primary reference covers the scheduled human-reviewed exact criteria.
    Whole-split and scored-support references are reported as sensitivities.
    This function makes no model calls and never selects a winning judge.
    """
    study, jobs_path = Path(study), Path(jobs_path)
    manifest = json.loads((study / "manifest.json").read_text())
    pairs = catalog(manifest)
    jobs = json.loads(jobs_path.read_text())
    scheduled = {(job["variant"], job["pair_id"], job.get("draw", 0)) for job in jobs}
    if len(scheduled) != len(jobs) or not jobs:
        raise ValueError("Analysis needs a nonempty unique job schedule")
    pair_ids = sorted({job["pair_id"] for job in jobs})
    splits = {pairs[pair_id]["split"] for pair_id in pair_ids}
    if len(splits) != 1:
        raise ValueError("One round cannot combine development, validation, and test subjects")
    split = splits.pop()
    if split == "test":
        selection = manifest.get("selection")
        if (manifest.get("selection_locked") is not True or not isinstance(selection, dict)
                or manifest.get("selection_sha256") != digest(selection)):
            raise ValueError("Test analysis requires a frozen selection with a matching selection lock hash")
    dataset = Dataset(ssot, manifest["instrument"])
    if dataset.fingerprint != manifest["dataset_sha256"]:
        raise ValueError("Analysis SSOT fingerprint differs from the frozen study")
    variants = {row["variant"]: row for row in json.loads((study / "variants.json").read_text())}
    derived = load_derived_judges(study, variants)
    definitions = scheduled_judges(variants, scheduled, derived)
    models = sorted({plan["ai_model_preset_id"] for plan in dataset.plans.values()})
    units = {pair_id: [{"criterion": criterion["id"] + "|" + criterion["scale_id"],
                        "criterion_family": criterion_family(criterion["id"], dataset)}
                       for criterion in dataset.rubric(pairs[pair_id]["grading_id"])] for pair_id in pair_ids}
    expected_units = {(pair_id, criterion["criterion"]) for pair_id in pair_ids for criterion in units[pair_id]}
    human_full_rows = _human_rows(dataset, set(manifest["splits"][split]), pairs)
    human_rows = [row for row in human_full_rows if _key(row) in expected_units]
    human_pairs, human_full_pairs = aggregate_pairs(human_rows), aggregate_pairs(human_full_rows)
    human_fit = fit_bradley_terry(human_pairs, models=models)
    human_full_fit = fit_bradley_terry(human_full_pairs, models=models)
    calls = {}
    for job in jobs:
        path = call_path(study, job)
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        entry = pairs[job["pair_id"]]
        if any(record.get(key, 0 if key == "draw" else None) != job.get(key, 0 if key == "draw" else None)
               for key in ("variant", "pair_id", "draw")):
            raise ValueError(f"Cached call identity differs from the scheduled job: {path}")
        if record["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError(f"Cached call uses different SSOT: {path}")
        for key in ("subject", "plan_ids", "model_a", "model_b"):
            if record[key] != entry[key]:
                raise ValueError(f"Cached call disagrees with catalog field {key}: {path}")
        if set(_call_votes(record)) != {criterion["criterion"] for criterion in units[job["pair_id"]]}:
            raise ValueError(f"Cached call criteria differ from the scheduled task: {path}")
        calls[(job["variant"], job["pair_id"], job.get("draw", 0))] = record
    output = {"name": name, "split": split, "dataset_sha256": dataset.fingerprint,
              "jobs_sha256": digest(jobs), "models": models, "n_pairs": len(pair_ids),
              "n_subjects": len({pairs[pair_id]["subject"] for pair_id in pair_ids}),
              "n_human_reviewed_pairs": len(human_pairs), "n_expected_calls": len(jobs),
              "n_completed_calls": len(calls), "complete": len(calls) == len(jobs),
              "human": human_fit, "human_full": human_full_fit,
              "human_coverage": {"criterion_votes": len(human_rows),
                                 "exact_criteria": len({_key(row) for row in human_rows}),
                                 "scheduled_criteria": len(expected_units)},
              "judges": [], "density": {},
              "interpretation": "Primary metrics compare the fixed scheduled human-reviewed exact-criterion reference with each judge's scheduled-pair model leaderboard. Whole-split human and scored-support sensitivities are separate. Missing calls/abstentions never become ties. Human preferences are reference measurements, not assumed ground truth."}
    if derived:
        included = {definition["variant"] for definition in definitions}
        output["derived_judge_definitions"] = [{**definition, "all_members_scheduled": definition["variant"] in included}
                                                for definition in derived]
    human_units = {_key(row) for row in human_rows}
    human_balanced_fit = fit_bradley_terry(_subject_balanced(human_pairs), models=models)
    for definition in definitions:
        rows, coverage = materialize_judge(definition, pair_ids, pairs, units, calls)
        pair_rows = aggregate_pairs(rows)
        reviewed_pairs = [row for row in pair_rows if pairs[row["pair_id"]]["human_reviewed"]]
        unreviewed_pairs = [row for row in pair_rows if not pairs[row["pair_id"]]["human_reviewed"]]
        coverage.update(human_reviewed_scheduled_pairs=sum(pairs[pair_id]["human_reviewed"] for pair_id in pair_ids),
                        unreviewed_scheduled_pairs=sum(not pairs[pair_id]["human_reviewed"] for pair_id in pair_ids),
                        human_reviewed_scored_pairs=len(reviewed_pairs), unreviewed_scored_pairs=len(unreviewed_pairs))
        fit = fit_bradley_terry(pair_rows, models=models)
        scored_units = {_key(row) for row in rows if row["value"] is not None}
        matched_human = [row for row in human_rows if _key(row) in scored_units]
        matched_judge = [row for row in rows if row["value"] is not None and _key(row) in human_units]
        matched_human_fit = fit_bradley_terry(aggregate_pairs(matched_human), models=models)
        matched_judge_fit = fit_bradley_terry(aggregate_pairs(matched_judge), models=models)
        intervals = None
        if bootstrap and human_pairs and pair_rows:
            shared_subjects = {r["subject"] for r in human_pairs} & {r["subject"] for r in pair_rows}
            if shared_subjects:
                intervals = bootstrap_alignment(human_pairs, pair_rows, models=models,
                                                n_bootstrap=bootstrap, seed=manifest["seed"])
        criteria = {}
        for family in sorted({row["criterion_family"] for row in human_rows + rows}):
            reference = fit_bradley_terry(aggregate_pairs([row for row in human_rows if row["criterion_family"] == family]), models=models)
            predicted = fit_bradley_terry(aggregate_pairs([row for row in rows if row["criterion_family"] == family]), models=models)
            criteria[family] = {"human": reference, "judge": predicted, "metrics": compare_rankings(reference, predicted)}
        balanced_fit = fit_bradley_terry(_subject_balanced(pair_rows), models=models)
        reviewed_fit = fit_bradley_terry(reviewed_pairs, models=models)
        rubric_ablations = {}
        if split == "test":
            for label, excluded in (("exclude_q07", {"q07"}), ("exclude_q08", {"q08"}),
                                    ("exclude_q07_q08", {"q07", "q08"})):
                reference = fit_bradley_terry(aggregate_pairs([row for row in human_rows if row["criterion_family"] not in excluded]), models=models)
                predicted = fit_bradley_terry(aggregate_pairs([row for row in rows if row["criterion_family"] not in excluded]), models=models)
                rubric_ablations[label] = {"excluded_families": sorted(excluded), "human": reference,
                                           "judge": predicted, "metrics": compare_rankings(reference, predicted),
                                           "interpretation": "Predeclared diagnostic sensitivity only; not used to change the selected judge after seeing test results."}
        output["judges"].append({**definition,
                                 "metrics": compare_rankings(human_fit, fit), "fit": fit,
                                 "metrics_full_human": compare_rankings(human_full_fit, fit),
                                 "matched": {"human": matched_human_fit, "judge": matched_judge_fit,
                                             "metrics": compare_rankings(matched_human_fit, matched_judge_fit)},
                                 "bootstrap": intervals, "coverage": coverage,
                                 "bootstrap_cohort_warning": (
                                     "Bootstrap uses common scored subjects, so its point estimate differs from the primary full scheduled cohort."
                                     if intervals and (intervals["excluded_human_subjects"] or intervals["excluded_judge_subjects"]) else None),
                                 "estimated_usd": coverage["estimated_usd"], "n_calls": coverage["n_calls"],
                                 "criteria": criteria,
                                 "rubric_ablations": rubric_ablations,
                                 "reviewed_only": {"fit": reviewed_fit, "metrics": compare_rankings(human_fit, reviewed_fit)},
                                 "subject_balanced": {"human": human_balanced_fit, "judge": balanced_fit,
                                                      "metrics": compare_rankings(human_balanced_fit, balanced_fit)}})
        if density:
            output["density"][definition["variant"]] = density_analysis(
                pair_rows, human_fit, models, repeats=density_repeats, seed=manifest["seed"])
            output["density"][definition["variant"]]["cohort"] = "reviewed_plus_unreviewed" if unreviewed_pairs else "reviewed_only"
            if unreviewed_pairs:
                output["density"][definition["variant"]]["reviewed_only"] = density_analysis(
                    reviewed_pairs, human_fit, models, repeats=density_repeats, seed=manifest["seed"])
                output["density"][definition["variant"]]["augmentation"] = augmentation_analysis(
                    reviewed_pairs, unreviewed_pairs, human_fit, models,
                    repeats=density_repeats, seed=manifest["seed"])
    write_json(study / f"{name}_analysis.json", output)
    return output


def analyze_legacy(study: Path, old_judgments_path: Path, bootstrap: int = 200,
                   ssot: Path = Path("studentbench_data_ssot")) -> dict:
    """Re-evaluate exposed legacy results as model rankings, without new calls."""
    study = Path(study)
    manifest = json.loads((study / "manifest.json").read_text())
    pairs = catalog(manifest)
    dataset = Dataset(ssot, manifest["instrument"])
    if dataset.fingerprint != manifest["dataset_sha256"]:
        raise ValueError("Analysis SSOT fingerprint differs from the frozen study")
    models = sorted({plan["ai_model_preset_id"] for plan in dataset.plans.values()})
    records = [json.loads(line) for line in Path(old_judgments_path).read_text().splitlines() if line.strip()]
    grouped, versions, criteria_by_pair = defaultdict(list), {}, {}
    seen = set()
    for record in records:
        if record["instrument"] != dataset.instrument or record["dataset_sha256"] != dataset.fingerprint:
            raise ValueError("Legacy record instrument or SSOT fingerprint differs")
        plans = sorted(record["plan_ids"])
        pair_id = digest([record["subject"], *plans])[:20]
        if pair_id not in pairs or pairs[pair_id]["split"] != "development":
            raise ValueError("Legacy analysis may only use frozen development subjects")
        name = record["judge_id"]
        if (name, pair_id) in seen:
            raise ValueError("Legacy analysis found duplicate judge/pair comparisons")
        seen.add((name, pair_id))
        version = record.get("judge_version", name)
        if name in versions and versions[name] != version:
            raise ValueError("Legacy analysis cannot pool multiple judge versions")
        versions[name] = version
        expected = {(criterion["id"], criterion["scale_id"]) for criterion in record["criteria"]}
        if pair_id in criteria_by_pair and criteria_by_pair[pair_id] != expected:
            raise ValueError("Legacy judges used different exact criteria for the same pair")
        criteria_by_pair[pair_id] = expected
        grouped[name].append((pair_id, record))
    expected_units = {(pair_id, identifier + "|" + scale)
                      for pair_id, criteria in criteria_by_pair.items() for identifier, scale in criteria}
    human_full_rows = _human_rows(dataset, set(manifest["splits"]["development"]), pairs)
    human_rows = [row for row in human_full_rows if _key(row) in expected_units]
    human_pairs = aggregate_pairs(human_rows)
    human_fit = fit_bradley_terry(human_pairs, models=models)
    human_full_fit = fit_bradley_terry(aggregate_pairs(human_full_rows), models=models)
    output = {"name": "legacy_development", "split": "development", "dataset_sha256": dataset.fingerprint,
              "models": models, "n_pairs": len(criteria_by_pair),
              "n_subjects": len({pairs[pair_id]["subject"] for pair_id in criteria_by_pair}),
              "n_human_reviewed_pairs": len(human_pairs), "complete": True,
              "human": human_fit, "human_full": human_full_fit, "judges": [], "density": {},
              "interpretation": "Exposed development-only results from the previous 100-pair experiment. Reference averages all matching human reviewers within the exact task criteria, then gives rubric families and plan pairs equal weight. These data were already used to diagnose and redesign judges; they are not held-out evidence."}
    for name, comparisons in sorted(grouped.items()):
        rows = []
        for pair_id, record in comparisons:
            entry = pairs[pair_id]
            orientation = 1 if record["plan_ids"] == entry["plan_ids"] else -1
            for criterion in record["criteria"]:
                raw = record["ratings"][criterion["id"]]["score"]
                score = None if raw is None else orientation * raw
                rows.append({"subject": entry["subject"], "pair_id": pair_id,
                             "model_a": entry["model_a"], "model_b": entry["model_b"],
                             "criterion": criterion["id"] + "|" + criterion["scale_id"],
                             "criterion_family": criterion_family(criterion["id"], dataset),
                             "value": None if score is None else .5 if score == 0 else 1.0 if score > 0 else 0.0})
        pair_rows = aggregate_pairs(rows)
        fit = fit_bradley_terry(pair_rows, models=models)
        scored_units = {_key(row) for row in rows if row["value"] is not None}
        matched_human = fit_bradley_terry(aggregate_pairs([row for row in human_rows if _key(row) in scored_units]), models=models)
        intervals = bootstrap_alignment(human_pairs, pair_rows, models=models, n_bootstrap=bootstrap,
                                        seed=manifest["seed"]) if bootstrap and pair_rows else None
        criteria = {}
        for family in sorted({row["criterion_family"] for row in rows}):
            reference = fit_bradley_terry(aggregate_pairs([row for row in human_rows if row["criterion_family"] == family]), models=models)
            predicted = fit_bradley_terry(aggregate_pairs([row for row in rows if row["criterion_family"] == family]), models=models)
            criteria[family] = {"human": reference, "judge": predicted, "metrics": compare_rankings(reference, predicted)}
        n_calls = sum(len(record["calls"]) for _pair, record in comparisons)
        coverage = {"scheduled_pairs": len(criteria_by_pair), "completed_pairs": len(comparisons),
                    "scored_pairs": len(pair_rows), "requested_criteria": len(expected_units),
                    "scored_criteria": len(scored_units), "criterion_coverage": len(scored_units) / len(expected_units),
                    "pair_coverage": len(pair_rows) / len(criteria_by_pair), "n_calls": n_calls,
                    "missing_calls": 0, "estimated_usd": None}
        output["judges"].append({"variant": name, "label": f"Legacy {name} ({versions[name]})",
                                 "fit": fit, "metrics": compare_rankings(human_fit, fit),
                                 "metrics_full_human": compare_rankings(human_full_fit, fit),
                                 "matched": {"human": matched_human, "judge": fit,
                                             "metrics": compare_rankings(matched_human, fit)},
                                 "bootstrap": intervals, "coverage": coverage, "criteria": criteria,
                                 "estimated_usd": None, "n_calls": n_calls,
                                 "bootstrap_cohort_warning": (
                                     "Bootstrap uses common scored subjects, so its point estimate differs from the primary full scheduled cohort."
                                     if intervals and (intervals["excluded_human_subjects"] or intervals["excluded_judge_subjects"]) else None)})
    write_json(study / "legacy_development_analysis.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--ssot", type=Path, default=Path("studentbench_data_ssot"))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--density-repeats", type=int, default=100)
    args = parser.parse_args()
    result = analyze_round(args.study, args.name, args.jobs, args.bootstrap,
                           args.density, args.density_repeats, args.ssot)
    print(json.dumps({"name": result["name"], "complete": result["complete"],
                      "judges": [{"variant": row["variant"], "metrics": row["metrics"],
                                  "coverage": row["coverage"]} for row in result["judges"]]}, indent=2))


if __name__ == "__main__":
    main()
