"""Small CLI; all data reads are local, model calls require an explicit judge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import Dataset, INSTRUMENTS, digest
from .judge import LiteLLMJudge, evaluate, make_task
from .report import human_scores, judge_report, leaderboard


def records_from(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def output_path(path: Path, dataset: Dataset) -> Path:
    path = path.resolve()
    if path.is_relative_to(dataset.root):
        raise ValueError("Output must be outside the read-only SSOT directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def emit(value: object, output: Path | None, dataset: Dataset) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output is None:
        print(content, end="")
    else:
        output_path(output, dataset).write_text(content, encoding="utf-8")
        print(f"Wrote {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Judge StudentBench lesson plans and validate against human reviews.")
    root.add_argument("--ssot", type=Path, default=Path(os.environ.get("STUDENTBENCH_SSOT", "studentbench_data_ssot")))
    root.add_argument("--instrument", choices=INSTRUMENTS, default="gre_quant")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate joins and summarize available data")
    inspect.add_argument("--verify", action="store_true", help="Verify six input files against the SSOT manifest")
    listing = commands.add_parser("reviews", help="List review IDs and their lesson-plan IDs")
    listing.add_argument("--limit", type=int, default=10)
    judge = commands.add_parser("judge", help="Judge one review or a bounded batch of human-reviewed tasks")
    selection = judge.add_mutually_exclusive_group(required=True)
    selection.add_argument("--grading-id", help="Review ID supplies pretest and exact rubric/scale versions")
    selection.add_argument("--limit", type=int, help="First N distinct tasks in stable review-ID order")
    judge.add_argument("--plans", nargs="+", help="One or two SSOT plan IDs; requires --grading-id")
    judge.add_argument("--mode", choices=("single", "pairwise"), default="pairwise")
    judge.add_argument("--slot", choices=("a", "b"), default="a", help="Plan to rate in single mode")
    judge.add_argument("--swap", action="store_true", help="Reverse A/B to audit position bias")
    judge.add_argument("--model", help="Explicit LiteLLM provider/model identifier")
    judge.add_argument("--reasoning-effort")
    judge.add_argument("--dry-run", action="store_true", help="Write exact prompts; no model import or API call")
    judge.add_argument("--output", type=Path, help="JSONL results, or JSON prompt previews with --dry-run")
    report = commands.add_parser("report", help="Compare saved judge ratings with humans and rank plans/models")
    report.add_argument("results", type=Path)
    report.add_argument("--level", choices=("model", "plan"), default="model")
    report.add_argument("--output", type=Path)
    human = commands.add_parser("human-leaderboard", help="Descriptive reference leaderboard from human grades")
    human.add_argument("--mode", choices=("single", "pairwise"), default="single")
    human.add_argument("--level", choices=("model", "plan"), default="model")
    human.add_argument("--output", type=Path)
    return root


def run(args: argparse.Namespace) -> None:
    dataset = Dataset(args.ssot, args.instrument)
    if args.command == "inspect":
        summary = dataset.summary()
        if args.verify:
            summary["verified_input_files"] = dataset.verify_manifest()
        emit(summary, None, dataset)
    elif args.command == "reviews":
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        fields = ("grading_id", "review_subject_id", "lesson_plan_a_id", "lesson_plan_b_id")
        emit([{key: row[key] for key in fields} for row in
              sorted(dataset.sessions.values(), key=lambda r: r['grading_id'])[:args.limit]], None, dataset)
    elif args.command == "human-leaderboard":
        rows = leaderboard(dataset, {k: v for k, v in human_scores(dataset).items() if k[0] == args.mode}, args.level)
        emit({"instrument": dataset.instrument, "dataset_sha256": dataset.fingerprint,
              "interpretation": "Descriptive per-criterion rankings; no significance estimates. "
                                "Repeated human ratings are averaged per unit; pairwise win shares "
                                "depend on the observed comparison schedule.", "leaderboard": rows}, args.output, dataset)
    elif args.command == "report":
        emit(judge_report(dataset, records_from(args.results), args.level), args.output, dataset)
    elif args.command == "judge":
        run_judge(args, dataset)


def run_judge(args: argparse.Namespace, dataset: Dataset) -> None:
    if args.plans and not args.grading_id:
        raise ValueError("--plans requires --grading-id for the rubric and pretest context")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.dry_run and (not args.model or not args.output):
        raise ValueError("Model calls require --model and --output (use results/*.jsonl)")
    if args.output:
        path = output_path(args.output, dataset)
    else:
        path = None
    judge = LiteLLMJudge(args.model or "dry-run", args.reasoning_effort)
    ids = [args.grading_id] if args.grading_id else sorted(dataset.sessions)
    tasks, seen = [], set()
    for grading_id in ids:
        task = make_task(dataset, grading_id, args.plans, args.mode, args.slot, args.swap)
        request_id = digest({"task": task.metadata(), "judge": judge.configuration()})
        if request_id not in seen:
            seen.add(request_id)
            tasks.append((request_id, task))
        if args.limit and len(tasks) >= args.limit:
            break
    if args.dry_run:
        emit({"calls": len(tasks), "tasks": [{"request_id": request_id, **task.metadata(),
              "messages": task.messages(), "response_schema": task.schema()} for request_id, task in tasks]},
             path, dataset)
        return
    existing = {r["request_id"] for r in records_from(path)} if path.exists() else set()
    completed = 0
    for request_id, task in tasks:
        if request_id in existing:
            continue
        record = {"request_id": request_id, **evaluate(task, judge)}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        completed += 1
        print(f"Saved {completed}: {task.mode} {', '.join(task.plan_ids)}", flush=True)
    print(f"Completed {completed}; reused {len(tasks) - completed}; results: {path}")


def main() -> None:
    arguments = parser()
    args = arguments.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError) as error:
        arguments.exit(1, f"Error: {error}\n")
