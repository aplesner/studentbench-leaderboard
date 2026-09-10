"""Read the frozen SSOT human AI-vs-AI study without modifying it."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

STUDY = "2_human_ai_vs_ai_lesson_plan_rubric_grading_data"
INSTRUMENTS = ("gre_quant", "gre_verbal")
DEFAULT_SCALE = "lesson_plan_quality_default"
ACCURACY_SCALE = "lesson_plan_answer_accuracy_specific"
INPUTS = (
    "human_grading_sessions.csv", "rubric_grades.csv", "rubric_questions.csv",
    "rating_scales.csv", "review_subject_pretest_responses.csv", "reviewed_lesson_plans.json",
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def index_unique(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if not row[key] or row[key] in result:
            raise ValueError(f"Missing or duplicate {key}: {row[key]!r}")
        result[row[key]] = row
    return result


def score(value: str, low: int, high: int) -> int | None:
    if value == "":
        return None
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"Score outside {low}..{high}: {value}")
    return number


class Dataset:
    def __init__(self, root: Path, instrument: str):
        if instrument not in INSTRUMENTS:
            raise ValueError(f"Unknown instrument: {instrument}")
        self.root = root.resolve()
        self.instrument = instrument
        self.directory = self.root / STUDY / instrument
        if not self.directory.is_dir():
            raise ValueError(f"SSOT study not found: {self.directory}. Set --ssot or STUDENTBENCH_SSOT.")
        self.hashes = {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                       for name in INPUTS}
        self.fingerprint = digest(self.hashes)
        self.sessions = index_unique(read_csv(self.directory / INPUTS[0]), "grading_id")
        self.grades = read_csv(self.directory / INPUTS[1])
        self.questions = index_unique(read_csv(self.directory / INPUTS[2]), "question_variant_id")
        self.scales: dict[str, dict[int, str]] = defaultdict(dict)
        for row in read_csv(self.directory / INPUTS[3]):
            self.scales[row["scale_id"]][int(row["score"])] = row["label"]
        self.pretests: dict[str, list[dict]] = defaultdict(list)
        for row in read_csv(self.directory / INPUTS[4]):
            self.pretests[row["review_subject_id"]].append(row)
        self.plans = index_unique(json.loads((self.directory / INPUTS[5]).read_text())['plans'],
                                  "lesson_plan_id")
        self.grades_by_session: dict[str, list[dict]] = defaultdict(list)
        seen = set()
        for grade in self.grades:
            session = self.sessions[grade["grading_id"]]
            question = self.questions[grade["question_variant_id"]]
            key = (grade["grading_id"], grade["question_variant_id"])
            if key in seen:
                raise ValueError(f"Duplicate rubric grade: {key}")
            seen.add(key)
            for field in ("instrument_id", "rubric_version", "comparison_type"):
                if grade[field] != session[field] or grade[field] != question[field]:
                    raise ValueError(f"Mismatched {field}: {key}")
            if grade["comparison_id"] != session["comparison_id"]:
                raise ValueError(f"Mismatched comparison: {key}")
            for slot in ("a", "b"):
                rating = score(grade[f"lesson_plan_{slot}_rating"], 1, 5)
                label = grade[f"lesson_plan_{slot}_rating_label"]
                if rating is not None and label != self.scales[self.rating_scale(grade)][rating]:
                    raise ValueError(f"Unknown rating label: {key}: {label!r}")
            score(grade["preference_score"], -2, 2)
            self.grades_by_session[grade["grading_id"]].append(grade)
        for session in self.sessions.values():
            subject = session["review_subject_id"]
            if session["instrument_id"] != instrument or subject not in self.pretests:
                raise ValueError(f"Missing pretest or wrong instrument: {session['grading_id']}")
            if session["review_display_variant"] != "complete_plan":
                raise ValueError("Only complete-plan AI-vs-AI reviews are supported")
            for slot in ("a", "b"):
                plan = self.plans[session[f"lesson_plan_{slot}_id"]]
                if subject not in plan["review_subject_ids"]:
                    raise ValueError("Plan and review refer to different pretests")
            if session["lesson_plan_a_id"] == session["lesson_plan_b_id"]:
                raise ValueError("A review must contain two distinct plans")
            if not self.grades_by_session[session["grading_id"]]:
                raise ValueError("Review session has no grades")

    @staticmethod
    def rating_scale(grade: dict) -> str:
        labels = grade["lesson_plan_a_rating_label"] + grade["lesson_plan_b_rating_label"]
        return ACCURACY_SCALE if "solution" in labels.lower() else DEFAULT_SCALE

    def rubric(self, grading_id: str) -> list[dict]:
        """Expose wording and scale definitions, never a reviewer's selected labels."""
        result = []
        for grade in sorted(self.grades_by_session[grading_id], key=lambda r: int(r["question_position"])):
            question = self.questions[grade["question_variant_id"]]
            scale_id = self.rating_scale(grade)
            result.append({"id": question["question_variant_id"], "question": question["question_text"],
                           "scale_id": scale_id, "rating_labels": self.scales[scale_id]})
        return result

    def summary(self) -> dict:
        return {"instrument": self.instrument, "sessions": len(self.sessions),
                "rubric_grades": len(self.grades), "plans": len(self.plans),
                "review_subjects": len(self.pretests), "question_variants": len(self.questions),
                "models": sorted({p["ai_model_preset_id"] for p in self.plans.values()}),
                "input_sha256": self.fingerprint}

    def verify_manifest(self) -> int:
        manifest = json.loads((self.root / "release_manifest.json").read_text())
        files = {row["path"]: row["sha256"] for row in manifest["files"]}
        for name, actual in self.hashes.items():
            path = f"{STUDY}/{self.instrument}/{name}"
            if files.get(path) != actual:
                raise ValueError(f"SSOT manifest hash mismatch: {path}")
        return len(self.hashes)
