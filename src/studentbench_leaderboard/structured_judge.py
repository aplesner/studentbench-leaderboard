"""Evidence-first responses with winner-to-score conversion performed in code."""
from __future__ import annotations

from dataclasses import replace
import json

from .judge import Task, parse_result


def project_plan(plan, projection: str):
    """Remove archival answer keys the original human review UI did not display."""
    if projection == "as_stored":
        return plan
    if projection != "human_display":
        raise ValueError(f"Unknown plan projection: {projection}")
    if isinstance(plan, dict):
        return {k: project_plan(v, projection) for k, v in plan.items() if k != "answer_original"}
    if isinstance(plan, list):
        return [project_plan(v, projection) for v in plan]
    return plan


class EvidenceTask(Task):
    def messages(self):
        # Pairwise preference does not use the unrelated absolute 1–5 labels.
        task = replace(self, rubric=[{k: v for k, v in q.items() if k != "rating_labels"}
                                    for q in self.rubric])
        return Task.messages(task)

    def schema(self):
        fields = {"evidence_a": {"type": "string"}, "evidence_b": {"type": "string"},
                  "reason": {"type": "string"},
                  "winner": {"type": "string", "enum": ["A", "B", "tie", "insufficient"]},
                  "strength": {"type": "string", "enum": ["none", "slight", "clear"]}}
        entry = {"type": "object", "properties": fields, "required": list(fields),
                 "additionalProperties": False}
        return {"type": "object", "properties": {q["id"]: entry for q in self.rubric},
                "required": [q["id"] for q in self.rubric], "additionalProperties": False}


class AbsoluteTask(Task):
    def schema(self):
        fields = {"evidence_a": {"type": "string"}, "evidence_b": {"type": "string"},
                  "reason": {"type": "string"},
                  "rating_a": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, None]},
                  "rating_b": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, None]}}
        entry = {"type": "object", "properties": fields, "required": list(fields),
                 "additionalProperties": False}
        return {"type": "object", "properties": {q["id"]: entry for q in self.rubric},
                "required": [q["id"] for q in self.rubric], "additionalProperties": False}


def response_task(task: Task, style: str) -> Task:
    cls = {"score_reason": Task, "evidence_winner": EvidenceTask,
           "absolute_ratings": AbsoluteTask}[style]
    return cls(**task.__dict__)


def parse_judge_response(raw: str, task: Task) -> dict:
    if type(task) is Task:
        return parse_result(raw, task)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(result, dict) or set(result) != {q["id"] for q in task.rubric}:
        raise ValueError("Response must contain exactly the requested criteria")
    expected = set(next(iter(task.schema()["properties"].values()))["properties"])
    ratings = {}
    for key, item in result.items():
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(f"Invalid response fields for {key}")
        for field in ("evidence_a", "evidence_b", "reason"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Missing {field} for {key}")
        if isinstance(task, EvidenceTask):
            winner, strength = item["winner"], item["strength"]
            if winner not in {"A", "B", "tie", "insufficient"}:
                raise ValueError("Invalid winner")
            if ((winner in {"tie", "insufficient"} and strength != "none") or
                    (winner in {"A", "B"} and strength not in {"slight", "clear"})):
                raise ValueError("Winner and strength are inconsistent")
            value = None if winner == "insufficient" else 0 if winner == "tie" else (
                (1 if winner == "A" else -1) * (1 if strength == "slight" else 2))
        else:
            a, b = item["rating_a"], item["rating_b"]
            for value in (a, b):
                if value is not None and (type(value) is not int or not 1 <= value <= 5):
                    raise ValueError("Invalid absolute rating")
            value = None if a is None or b is None else max(-2, min(2, a - b))
        ratings[key] = {"score": value, "reason": item["reason"]}
    return ratings
