"""Public two-plan comparison API and configurable judging strategies."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median_low
from time import monotonic

from .data import digest
from .judge import LiteLLMJudge, Task, evaluate, parse_result
from .judge_config import ENGINE_VERSION, SOFT_STRATEGIES, JudgeDefinition, load_definition
from .structured_judge import response_task, parse_judge_response, project_plan


@dataclass(frozen=True)
class ComparisonContext:
    """Shared evidence and IDs; human answers and author metadata do not belong here."""

    instrument: str
    subject: str
    pretest: list[dict]
    rubric: list[dict]
    plan_ids: tuple[str, str] = ("plan_a", "plan_b")
    dataset_sha256: str = ""


def request_id(task: Task, judge: ConfiguredJudge) -> str:
    return digest({"task": judge.task_metadata(task), "judge": judge.configuration()})


def compare_plans(plan_a: dict, plan_b: dict, judge_id: str, *, context: ComparisonContext,
                  judges_dir: Path | None = None, client: object | None = None) -> dict:
    """Compare two plans using a stable judge ID; resolve subversions internally.

    Returns a report-compatible result including every call and normalized vote.
    Does not write files. Both plans must be for the pretest supplied in context.
    """
    if not all(isinstance(plan, dict) and plan for plan in (plan_a, plan_b)):
        raise ValueError("Both lesson plans must be nonempty JSON objects")
    if (len(context.plan_ids) != 2 or len(set(context.plan_ids)) != 2
            or not all(isinstance(key, str) and key for key in context.plan_ids)):
        raise ValueError("Two distinct nonempty plan IDs are required")
    if not context.instrument or not context.subject or not context.pretest or not context.rubric:
        raise ValueError("Comparison context requires instrument, subject, pretest, and rubric")
    criteria = [q["id"] for q in context.rubric]
    if len(set(criteria)) != len(criteria):
        raise ValueError("Rubric IDs must be unique")
    fingerprint = context.dataset_sha256 or digest({"plans": [plan_a, plan_b],
                                                     "context": context.__dict__})
    task = Task(context.instrument, context.subject, list(context.plan_ids), [plan_a, plan_b],
                context.pretest, context.rubric, fingerprint)
    judge = ConfiguredJudge(load_definition(judge_id, judges_dir), client=client)
    return {"request_id": request_id(task, judge), **evaluate(task, judge)}


def aggregate(task: Task, calls: list[dict], strategy: str) -> tuple[dict, dict]:
    ratings, diagnostics = {}, {}
    for criterion in task.rubric:
        key = criterion["id"]
        scores = [call["normalized_ratings"][key]["score"] for call in calls]
        scored = [score for score in scores if score is not None]
        votes = Counter("abstain" if score is None else "A" if score > 0 else "B" if score < 0 else "tie"
                        for score in scores)
        winner = next((label for label in ("A", "tie", "B") if votes[label] > len(calls) / 2), None)
        value = None
        if winner == "tie":
            value = 0
        elif winner is not None:
            direction = 1 if winner == "A" else -1
            # Retain preference intensity conservatively among the winning-direction voters.
            value = direction * median_low([abs(score) for score in scored if score * direction > 0])
        ratings[key] = {"score": value, "reason": (
            calls[0]["normalized_ratings"][key]["reason"] if strategy == "single" else
            f"Votes relative to input plans: A={votes['A']}, tie={votes['tie']}, B={votes['B']}, "
            f"abstain={votes['abstain']}. " +
            (f"Strict majority: {winner}." if winner is not None else "No strict majority; inconclusive.")
        )}
        diagnostics[key] = {"votes": {label: votes[label] for label in ("A", "tie", "B", "abstain")},
                            "normalized_scores": scores, "majority": winner,
                            "score_changed": len(set(scored)) > 1 if len(scored) >= 2 else None,
                            "direction_changed": len({(s > 0) - (s < 0) for s in scored}) > 1
                            if len(scored) >= 2 else None}
        if strategy in SOFT_STRATEGIES:
            shares = [None if score is None else 1.0 if score > 0 else 0.0 if score < 0 else .5
                      for score in scores]
            probability = sum(shares) / len(shares) if shares and all(p is not None for p in shares) else None
            diagnostics[key].update(
                preference_probability=probability, normalized_win_shares=shares,
                preference_status="complete" if probability is not None else "member_abstention",
                preference_interpretation="Mean normalized win share; 0.5 can reflect opposing votes and does not assert a categorical tie.")
    return ratings, diagnostics


@dataclass
class ConfiguredJudge:
    definition: JudgeDefinition
    client: object | None = None

    def configuration(self) -> dict:
        config = self.definition.configuration()
        if self.client is not None:
            routing = [self.client.configuration(m.model, reasoning_effort=m.reasoning_effort)
                       for m in self.definition.members]
            config = {**config, "version": f"{config['version']}-route-{digest(routing)}",
                      "routing": routing}
        return config

    @property
    def call_count(self) -> int:
        return self.definition.call_count

    def schedule(self, task: Task) -> list[tuple[int, bool]]:
        if task.mode != "pairwise" or len(task.plans) != 2:
            raise ValueError("Configured judges compare exactly two plans; use --model for single-plan ratings")
        strategy = self.definition.strategy
        if strategy == "single":
            return [(0, False)]
        if strategy in {"order_check", "balanced_soft"}:
            return [(0, False), (0, True)]
        if strategy in {"panel", "panel_soft"}:
            return [(index, False) for index in range(len(self.definition.members))]
        rng = random.Random(int(digest({"seed": self.definition.seed, "instrument": task.instrument,
                                        "subject": task.subject, "plans": task.plan_ids}), 16))
        return [(0, bool(rng.getrandbits(1))) for _ in range(self.definition.repetitions)]

    def call_task(self, task: Task, swapped: bool) -> Task:
        plans = [project_plan(p, self.definition.plan_projection) for p in task.plans]
        task = replace(task, plan_ids=list(reversed(task.plan_ids)) if swapped else task.plan_ids,
                       plans=list(reversed(plans)) if swapped else plans,
                       system_prompt=self.definition.system_prompt,
                       user_template=self.definition.user_template)
        return response_task(task, self.definition.response_style)

    def preview(self, task: Task) -> dict:
        calls = []
        for index, (member_index, swapped) in enumerate(self.schedule(task), 1):
            call_task = self.call_task(task, swapped)
            member = self.definition.members[member_index]
            calls.append({"call_index": index, "member": member.configuration(),
                          "order": "BA" if swapped else "AB", "plan_ids": call_task.plan_ids,
                          "messages": call_task.messages(), "response_schema": call_task.schema()})
        return {"judge": self.configuration(), "call_count": len(calls), "calls": calls,
                "prompt_version": ENGINE_VERSION,
                "prompt_sha256": digest([call["messages"] for call in calls])}

    def task_metadata(self, task: Task) -> dict:
        preview = self.preview(task)
        return {**task.metadata(), "prompt_version": preview["prompt_version"],
                "prompt_sha256": preview["prompt_sha256"]}

    def complete(self, task: Task) -> tuple[str, dict]:
        calls = []
        for index, (member_index, swapped) in enumerate(self.schedule(task), 1):
            member = self.definition.members[member_index]
            call_task = self.call_task(task, swapped)
            started = monotonic()
            raw, metadata = LiteLLMJudge(member.model, member.reasoning_effort, client=self.client).complete(call_task)
            ratings = parse_judge_response(raw, call_task)
            normalized = {key: {**rating, "score": -rating["score"]
                                if swapped and rating["score"] is not None else rating["score"]}
                          for key, rating in ratings.items()}
            calls.append({"call_index": index, "member": member.configuration(),
                          "order": "BA" if swapped else "AB", "plan_ids": call_task.plan_ids,
                          "messages": call_task.messages(), "response_schema": call_task.schema(),
                          "prompt_sha256": digest(call_task.messages()),
                          "ratings": ratings, "normalized_ratings": normalized,
                          "raw_response": raw, "duration_seconds": monotonic() - started,
                          "created_at": datetime.now(timezone.utc).isoformat(), **metadata})
        ratings, diagnostics = aggregate(task, calls, self.definition.strategy)
        metadata = {
            "calls": calls, "diagnostics": diagnostics, "prompt_version": ENGINE_VERSION,
            "prompt_sha256": digest([call["messages"] for call in calls]),
        }
        if self.definition.strategy in SOFT_STRATEGIES:
            metadata["preference_probabilities"] = {
                key: diagnostic["preference_probability"] for key, diagnostic in diagnostics.items()}
        return json.dumps(ratings, allow_nan=False), metadata
