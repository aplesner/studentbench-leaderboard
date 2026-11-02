"""Prepare one fixed, anonymized human-reviewed DEVELOPMENT illustration.

No APIs, validation outcomes, test outcomes, or synthetic assistant answers are
used. Source identifiers belong only in the separate local provenance file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

from studentbench_leaderboard.comparison import ConfiguredJudge
from studentbench_leaderboard.data import Dataset, digest, score
from studentbench_leaderboard.judge import make_task
from studentbench_leaderboard.judge_config import load_definition
from studentbench_leaderboard.pretest_context import PretestContextBank


POLICY = {
    "id": "one-development-human-example.v1",
    "eligibility": [
        "Subject belongs to the original frozen development split.",
        "At least two distinct reviewers; one review per reviewer.",
        "All reviews use the same A/B presentation, so human prose requires no A/B rewriting.",
        "All reviews cover the same eight exact rubric-variant/scale units with nonempty explanations.",
        "Across criteria, pooled human directions favor both A and B at least once.",
    ],
    "selection_order": [
        "Most distinct reviewers.",
        "Highest mean per-criterion majority-direction proportion among nonmissing votes.",
        "Most criteria with unanimous nonmissing direction.",
        "Lowest SHA256 of [policy_id, subject, sorted_plan_ids] as deterministic tie-break.",
    ],
    "interpretation": "Chosen illustration of criterion tradeoffs and reviewer disagreement; "
                      "not a representative sample, error-free gold answer, or evidence that this prompt improves ranking.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_example(dataset: Dataset, manifest: dict) -> tuple[dict, dict]:
    """Select using development annotations only; never inspect other labels."""
    development = set(manifest["splits"]["development"])
    groups = defaultdict(list)
    for grading_id, session in sorted(dataset.sessions.items()):
        if session["review_subject_id"] in development:
            key = (session["review_subject_id"], *sorted([session["lesson_plan_a_id"], session["lesson_plan_b_id"]]))
            groups[key].append(grading_id)
    candidates, exclusions = [], Counter()
    for key, grading_ids in sorted(groups.items()):
        sessions = [dataset.sessions[g] for g in grading_ids]
        reviewers = {s["human_reviewer_anon_id"] for s in sessions}
        if len(reviewers) < 2:
            exclusions["single_reviewer"] += 1
            continue
        if len(reviewers) != len(sessions):
            exclusions["repeated_reviewer"] += 1
            continue
        orders = {(s["lesson_plan_a_id"], s["lesson_plan_b_id"]) for s in sessions}
        if len(orders) != 1:
            exclusions["mixed_presentation_order"] += 1
            continue
        grades = [dataset.grades_by_session[g] for g in grading_ids]
        units = [{(r["question_variant_id"], dataset.rating_scale(r)) for r in review} for review in grades]
        if not all(len(u) == 8 and u == units[0] for u in units):
            exclusions["different_exact_rubrics"] += 1
            continue
        if any(not r["reviewer_reasoning"].strip() for review in grades for r in review):
            exclusions["missing_explanation"] += 1
            continue
        directions = defaultdict(list)
        for review in grades:
            for grade in review:
                value = score(grade["preference_score"], -2, 2)
                if value is not None:
                    directions[grade["question_variant_id"]].append((value > 0) - (value < 0))
        if len(directions) != 8:
            exclusions["criterion_without_preference"] += 1
            continue
        pooled = {(sum(v) > 0) - (sum(v) < 0) for v in directions.values()}
        if not {-1, 1} <= pooled:
            exclusions["no_bidirectional_tradeoff"] += 1
            continue
        proportions = [max(Counter(v).values()) / len(v) for v in directions.values()]
        candidates.append({
            "subject": key[0], "unordered_plan_ids": list(key[1:]), "plan_ids": list(next(iter(orders))),
            "grading_ids": grading_ids, "n_reviewers": len(reviewers),
            "mean_majority_direction_share": mean(proportions),
            "unanimous_direction_criteria": sum(p == 1 for p in proportions),
            "tie_break_sha256": digest([POLICY["id"], key[0], list(key[1:])]),
        })
    if not candidates:
        raise ValueError("No development example satisfies the fixed eligibility policy")
    candidates.sort(key=lambda x: (-x["n_reviewers"], -x["mean_majority_direction_share"],
                                    -x["unanimous_direction_criteria"], x["tie_break_sha256"]))
    return candidates[0], {"development_pairs_considered": len(groups), "eligible_pairs": len(candidates),
                           "exclusions": dict(exclusions), "selection_policy": POLICY}


def _redact_identifiers(value, replacements: dict[str, str], counts: Counter):
    if isinstance(value, dict):
        return {k: _redact_identifiers(v, replacements, counts) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_identifiers(v, replacements, counts) for v in value]
    if isinstance(value, str):
        for source, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
            value, n = re.subn(re.escape(source), lambda _: replacement, value, flags=re.IGNORECASE)
            counts[source] += n
    return value


def prepare(dataset: Dataset, study: Path) -> tuple[dict, dict]:
    manifest_path = study / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_sha256"] != dataset.fingerprint:
        raise ValueError("Frozen study and current SSOT fingerprints differ")
    split_sets = {name: set(values) for name, values in manifest["splits"].items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("development", "validation"),
                                                    ("development", "test"), ("validation", "test"))):
        raise ValueError("Frozen subject splits overlap")
    chosen, selection = select_example(dataset, manifest)
    variants = json.loads((study / "variants.json").read_text())
    variant = next(v for v in variants if v["variant"] == "enhanced_terra_high")
    config_dir = study / "configs" / variant["variant"]
    definition = load_definition(variant["judge_id"], config_dir)
    if definition.plan_projection != "human_display":
        raise ValueError("Example must use the same human-display projection as the enhanced judge")
    context_spec = variant["pretest_context"]
    source_bank = (config_dir / context_spec["file"]).resolve()
    if not source_bank.is_relative_to(config_dir.resolve()):
        raise ValueError("Context bank must be inside its frozen judge configuration")
    context = PretestContextBank.load(dataset.root, source_bank=source_bank,
                                      expected_source_sha256=context_spec["sha256"])
    task = make_task(dataset, chosen["grading_ids"][0], chosen["plan_ids"])
    task = context.enrich_task(task)
    presented = ConfiguredJudge(definition).call_task(task, False)
    # The task's JSON payload is the same input projection used by the zero-shot
    # enhanced judge. Only its surrounding experimental template is excluded.
    input_payload = {
        "instrument": presented.instrument, "pretest": presented.pretest,
        "rubric": [{k: v for k, v in q.items() if k != "rating_labels"} for q in presented.rubric],
        "plans": dict(zip(("A", "B"), presented.plans)),
    }
    human_reviews, reviewer_mapping = [], {}
    replacements = {chosen["subject"]: "example student"}
    for slot, plan_id in zip(("A", "B"), chosen["plan_ids"]):
        replacements[plan_id] = f"Plan {slot}"
        plan = dataset.plans[plan_id]
        for field in ("ai_model_preset_id", "ai_model_endpoint_id"):
            if plan.get(field):
                replacements[plan[field]] = "[author identity omitted]"
    for index, grading_id in enumerate(chosen["grading_ids"], 1):
        session = dataset.sessions[grading_id]
        reviewer = f"Reviewer {index}"
        reviewer_mapping[reviewer] = session["human_reviewer_anon_id"]
        replacements[session["human_reviewer_anon_id"]] = reviewer
        replacements[grading_id] = reviewer + " review"
        replacements[session["comparison_id"]] = "example comparison"
        if chosen["plan_ids"] != [session["lesson_plan_a_id"], session["lesson_plan_b_id"]]:
            raise ValueError("Human review presentation differs from the example")
        criteria = []
        for grade in sorted(dataset.grades_by_session[grading_id], key=lambda r: int(r["question_position"])):
            criteria.append({
                "criterion_id": grade["question_variant_id"],
                "quality_scale_id": dataset.rating_scale(grade),
                "plan_a_rating": {"score": score(grade["lesson_plan_a_rating"], 1, 5),
                                  "label": grade["lesson_plan_a_rating_label"] or None},
                "plan_b_rating": {"score": score(grade["lesson_plan_b_rating"], 1, 5),
                                  "label": grade["lesson_plan_b_rating_label"] or None},
                "preference": {"score": score(grade["preference_score"], -2, 2),
                               "label": grade["preference_label"] or None},
                "reason": grade["reviewer_reasoning"],
            })
        human_reviews.append({"reviewer": reviewer, "presentation": "A and B match example_input.plans.A/B",
                              "criteria": criteria})
    scales = {q["scale_id"] for q in task.rubric}
    example = {
        "schema_version": 1, "type": "one_human_reviewed_development_example",
        "instructions": "This is ONE completed comparison illustrating human rubric use. "
                        "It is not the current pair to judge. The full source human reviews are shown separately "
                        "and may disagree or contain mistakes. Learn the meaning of the rubric and discuss "
                        "evidence independently; do not assume every human reason is correct or copy these "
                        "preferences onto another pair. No ideal assistant response is supplied.",
        "preference_convention": "+2/+1 favors example Plan A; -2/-1 favors example Plan B; "
                                 "0 is the recorded equal-preference label. Null means a missing human judgment, "
                                 "not a tie. Per-plan 1–5 quality ratings are separate observations and are "
                                 "not converted into the direct preference.",
        "human_quality_scales": {scale_id: dataset.scales[scale_id] for scale_id in sorted(scales)},
        "human_preference_scale": dataset.scales["lesson_plan_a_vs_b_preference"],
        "example_input": input_payload, "human_reviews": human_reviews,
        "human_reason_text_policy": "Original wording is retained, except any source identifiers are anonymized. "
                                    "A/B presentation is unchanged. Human disagreements are not collapsed.",
    }
    redactions = Counter()
    example = _redact_identifiers(copy.deepcopy(example), replacements, redactions)
    text = json.dumps(example, ensure_ascii=False)
    forbidden = set(dataset.sessions[g]["human_reviewer_anon_id"] for g in chosen["grading_ids"])
    forbidden.update(chosen["grading_ids"] + chosen["plan_ids"] + [chosen["subject"]])
    forbidden.update(p["ai_model_preset_id"] for p in dataset.plans.values())
    for split in ("validation", "test"):
        forbidden.update(manifest["splits"][split])
        for pair in manifest["pair_catalog"][split]:
            forbidden.update(pair["plan_ids"])
    if any(identifier.lower() in text.lower() for identifier in forbidden):
        raise ValueError("Source or held-out identity leaked into the example payload")
    provenance = {
        "schema_version": 1, "source_split": "development", "dataset_sha256": dataset.fingerprint,
        "original_study_manifest_sha256": sha256(manifest_path),
        "source_subject": chosen["subject"], "source_plan_ids_in_example_order": chosen["plan_ids"],
        "source_grading_ids": chosen["grading_ids"], "anonymous_reviewer_mapping": reviewer_mapping,
        "selection": {**selection, "chosen": chosen}, "context": context.provenance,
        "source_projection": definition.plan_projection, "source_judge_configuration": definition.configuration(),
        "identifier_redactions": {k: v for k, v in redactions.items() if v},
        "no_ideal_assistant_response": True, "n_full_comparisons": 1,
        "human_annotations_preserved_individually": True,
        "validation_or_test_annotations_used_for_example_selection": False,
        "new_validation_model_outcomes_used_for_example_selection": False,
        "historical_note": "The original ranking-study rounds used zero human-reviewed prompt examples. "
                           "This artifact is a new one-example calibration ablation only.",
    }
    return example, provenance


def _write_once(path: Path, value: dict) -> str:
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != rendered:
        raise ValueError(f"Refusing to replace the fixed example artifact: {path}")
    path.write_text(rendered)
    return sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/ranking-study-20260910"))
    parser.add_argument("--ssot", type=Path, default=Path("studentbench_data_ssot"))
    parser.add_argument("--output", type=Path, default=Path("results/ranking-extension-20260911"))
    args = parser.parse_args()
    example, provenance = prepare(Dataset(args.ssot, "gre_quant"), args.study)
    args.output.mkdir(parents=True, exist_ok=True)
    example_path = args.output / "example.json"
    example_hash = _write_once(example_path, example)
    provenance.update(example_file=example_path.name, example_sha256=example_hash,
                      example_bytes=example_path.stat().st_size, preparation_script_sha256=sha256(Path(__file__)))
    provenance_hash = _write_once(args.output / "example_provenance.json", provenance)
    print(json.dumps({"example": str(example_path), "example_sha256": example_hash,
                      "provenance_sha256": provenance_hash, "bytes": provenance["example_bytes"],
                      "reviewers": len(example["human_reviews"]), "selection": provenance["selection"]["chosen"]}, indent=2))


if __name__ == "__main__":
    main()
