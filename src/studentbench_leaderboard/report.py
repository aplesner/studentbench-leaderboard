"""Descriptive rankings and agreement on exactly matched human-review units."""

from __future__ import annotations

from collections import defaultdict
from statistics import correlation, mean

from .data import Dataset, score
from .judge_config import check_strategy_history


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def unit(mode: str, subject: str, plans: list[str], question: str, scale: str) -> tuple:
    return mode, subject, tuple(sorted(plans)), question, scale


def canonical_score(plans: list[str], value: int | None) -> int | None:
    if value is None:
        return None
    return -value if len(plans) == 2 and plans != sorted(plans) else value


def human_scores(dataset: Dataset) -> dict[tuple, list[int]]:
    values = defaultdict(list)
    for grade in dataset.grades:
        session = dataset.sessions[grade["grading_id"]]
        subject = session["review_subject_id"]
        plans = [session["lesson_plan_a_id"], session["lesson_plan_b_id"]]
        question, scale = grade["question_variant_id"], dataset.rating_scale(grade)
        for slot, plan in zip(("a", "b"), plans):
            value = score(grade[f"lesson_plan_{slot}_rating"], 1, 5)
            if value is not None:
                values[unit("single", subject, [plan], question, scale)].append(value)
        value = canonical_score(plans, score(grade["preference_score"], -2, 2))
        if value is not None:
            values[unit("pairwise", subject, plans, question, scale)].append(value)
    return dict(values)


def ranks(values: list[float]) -> list[float]:
    positions = defaultdict(list)
    for position, value in enumerate(sorted(values), 1):
        positions[value].append(position)
    return [mean(positions[value]) for value in values]


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return correlation(ranks(left), ranks(right))


def leaderboard(dataset: Dataset, values: dict[tuple, list[int]], level: str = "model") -> list[dict]:
    groups = defaultdict(list)
    subjects = defaultdict(set)
    for key, scores in values.items():
        if not scores:
            continue
        mode, subject, plans, question, scale = key
        value = mean(scores)
        for index, plan_id in enumerate(plans):
            entity = plan_id if level == "plan" else dataset.plans[plan_id]["ai_model_preset_id"]
            group = (mode, question, scale, entity)
            # One vote per plan/pretest/criterion, or per unordered pair/pretest/criterion.
            contribution = value if mode == "single" else (1 + sign(value) * (1 if index == 0 else -1)) / 2
            groups[group].append(contribution)
            subjects[group].add(subject)
    rows = []
    for (mode, question, scale, entity), scores in groups.items():
        rows.append({"mode": mode, "question_variant_id": question, "rating_scale_id": scale,
                     level: entity, "metric": "mean_rating" if mode == "single" else "win_share",
                     "score": mean(scores), "n_units": len(scores),
                     "n_subjects": len(subjects[(mode, question, scale, entity)])})
    return sorted(rows, key=lambda r: (r["mode"], r["question_variant_id"], r["rating_scale_id"],
                                       -r["score"], r[level]))


def judge_report(dataset: Dataset, records: list[dict], level: str = "model") -> dict:
    reference = human_scores(dataset)
    judges = {}
    values = defaultdict(lambda: defaultdict(list))
    abstentions = defaultdict(lambda: defaultdict(int))
    seen = set()
    strategy_history = {}
    for record in records:
        if record["instrument"] != dataset.instrument:
            continue
        if record.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported judgment record schema")
        if record["dataset_sha256"] != dataset.fingerprint:
            raise ValueError("Results were generated from different SSOT inputs")
        request_id = record["request_id"]
        if request_id in seen:
            raise ValueError(f"Duplicate request in results: {request_id}")
        seen.add(request_id)
        if record["schema_version"] == 2:
            previous = strategy_history.get(record["judge_id"])
            check_strategy_history(record["judge"], [previous] if previous else [])
            strategy_history[record["judge_id"]] = record
        # Named IDs resolve to multiple versions; never pool those versions in a report.
        judge_id = record.get("judge_version", record["judge_id"])
        if judge_id in judges and judges[judge_id] != record["judge"]:
            raise ValueError(f"Conflicting configuration for judge version {judge_id}")
        judges[judge_id] = record["judge"]
        for criterion in record["criteria"]:
            key = unit(record["mode"], record["subject"], record["plan_ids"],
                       criterion["id"], criterion["scale_id"])
            value = canonical_score(record["plan_ids"], record["ratings"][criterion["id"]]["score"])
            if value is None:
                abstentions[judge_id][key] += 1
            else:
                values[judge_id][key].append(value)
    reports = []
    for judge_id, config in judges.items():
        groups = defaultdict(list)
        all_units = set(values[judge_id]) | set(abstentions[judge_id])
        for key in all_units:
            groups[(key[0], key[3], key[4])].append(key)
        agreement = []
        for (mode, question, scale), keys in sorted(groups.items()):
            matched = [k for k in keys if reference.get(k) and values[judge_id].get(k)]
            predicted = [mean(values[judge_id][k]) for k in matched]
            human = [mean(reference[k]) for k in matched]
            agreement.append({
                "mode": mode, "question_variant_id": question, "rating_scale_id": scale,
                "n_requested_units": len(keys), "n_matched_units": len(matched),
                "n_scored_units": sum(bool(values[judge_id].get(k)) for k in keys),
                "n_human_ratings": sum(len(reference[k]) for k in matched),
                "n_abstentions": sum(abstentions[judge_id].get(k, 0) for k in keys),
                "mae_against_human_mean": mean(abs(a-b) for a, b in zip(predicted, human)) if matched else None,
                "spearman_against_human_mean": spearman(predicted, human),
                "direction_agreement_against_human_mean": (
                    mean(sign(a) == sign(b) for a, b in zip(predicted, human))
                    if matched and mode == "pairwise" else None),
            })
        reports.append({"judge_id": config.get("judge_id", judge_id), "judge_version": judge_id,
                        "judge": config, "agreement": agreement,
                        "leaderboard": leaderboard(dataset, values[judge_id], level)})
    return {"instrument": dataset.instrument, "dataset_sha256": dataset.fingerprint,
            "interpretation": "Descriptive, per-criterion estimates; no claim of rank significance. "
                              "Repeat judgments are averaged within matched units. Human ratings "
                              "are averaged within those same units. Single-plan references were "
                              "elicited in paired reviews. Pairwise win share is schedule-dependent.",
            "judges": reports}
