"""Small CLI; all data reads are local, model calls require an explicit judge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import Dataset, INSTRUMENTS, digest
from .comparison import ConfiguredJudge, request_id as configured_request_id
from .judge import LiteLLMJudge, evaluate, make_task
from .judge_config import DEFAULT_JUDGES_DIR, check_strategy_history, load_definition
from .report import human_scores, judge_report, leaderboard

DEFAULT_RESULTS = Path("results/judgments.jsonl")
PROFILES = ("auto", "openrouter", "openai", "anthropic", "direct", "tailscale")


def connection_arguments(command: argparse.ArgumentParser, *, global_options: bool = False) -> None:
    default = None if global_options else argparse.SUPPRESS
    command.add_argument("--profile", choices=PROFILES, default=default,
                         help="Connection profile: auto prefers GPT/Claude native keys, otherwise OpenRouter")
    command.add_argument("--config", type=Path, default=default, help="Connection profile TOML file")


def execution_arguments(command: argparse.ArgumentParser) -> None:
    connection_arguments(command)
    command.add_argument("--benchmark", type=Path,
                         default=Path(os.environ.get("STUDENTBENCH_BENCHMARK", "data/gre-quant-v1")),
                         help="Prepared portable benchmark directory")
    command.add_argument("--dry-run", action="store_true", help="Preview exact requests without calling models")
    command.add_argument("--max-new-calls", type=int, default=1000,
                         help="Run at most N previously uncached jobs in this invocation")
    command.add_argument("--workers", type=int, default=1, help="Maximum simultaneous model requests")
    command.add_argument("--timeout", type=int, default=180, help="Timeout per model request in seconds")


def llm_client(args: argparse.Namespace):
    from .llm_client import LLMClient

    return LLMClient(profile=getattr(args, "profile", None), config_path=getattr(args, "config", None))


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


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
    root = argparse.ArgumentParser(description="Generate and evaluate StudentBench lesson plans.")
    root.add_argument("--ssot", type=Path, default=Path(os.environ.get("STUDENTBENCH_SSOT", "studentbench_data_ssot")))
    root.add_argument("--instrument", choices=INSTRUMENTS, default="gre_quant")
    connection_arguments(root, global_options=True)
    commands = root.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="Prepare and inspect portable benchmark inputs")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    prepare = benchmark_commands.add_parser("prepare", help="Freeze local SSOT and selected-study assets")
    prepare.add_argument("--output", type=Path, default=Path("data/gre-quant-v1"))
    prepare.add_argument("--study", type=Path, default=Path("results/ranking-study-20260910"))
    prepare.add_argument("--source-bank", type=Path, help="Explicit selected-judge question-bank supplement")
    check = benchmark_commands.add_parser("inspect", help="Verify a prepared benchmark's checksums and joins")
    check.add_argument("path", type=Path, nargs="?", default=Path("data/gre-quant-v1"))
    generate = commands.add_parser("generate", help="Generate a new model's plans for a frozen benchmark")
    execution_arguments(generate)
    generate.add_argument("--model", required=True, help="Model ID; OpenRouter is the default provider")
    generate.add_argument("--output", type=Path, required=True, help="Resumable generation run directory")
    generate.add_argument("--reasoning-effort")
    generate.add_argument("--limit", type=int, help="Generate a bounded deterministic subset of students")
    evaluate_command = commands.add_parser("evaluate", help="Judge new plans and produce a model leaderboard")
    execution_arguments(evaluate_command)
    evaluate_command.add_argument("generation_dir", type=Path)
    evaluate_command.add_argument("--output", type=Path, help="Defaults to GENERATION_DIR/evaluation")
    evaluate_command.add_argument("--cache-dir", type=Path,
                                  default=Path(os.environ.get("STUDENTBENCH_JUDGE_CACHE", "results/judge-cache")),
                                  help="Shared validated judge cache for subsequent candidate models")
    evaluate_command.add_argument("--comparisons-per-student", type=int,
                                  help="Compare each new plan with at most N baseline plans")
    evaluate_command.add_argument("--seed", type=int, default=20261102)
    evaluate_command.add_argument("--bootstrap", type=int, default=500,
                                  help="Number of student-cluster bootstrap resamples")
    multi = commands.add_parser("evaluate-many", help="Fit one joint leaderboard for several generator runs")
    execution_arguments(multi)
    multi.add_argument("generation_dirs", type=Path, nargs="+")
    multi.add_argument("--output", type=Path, required=True)
    multi.add_argument("--cache-dir", type=Path,
                       default=Path(os.environ.get("STUDENTBENCH_JUDGE_CACHE", "results/judge-cache")))
    multi.add_argument("--include-new-pairs", action=argparse.BooleanOptionalAction, default=True,
                       help="Include comparisons between the new generator systems")
    multi.add_argument("--comparisons-per-student", type=int)
    multi.add_argument("--seed", type=int, default=20261102)
    multi.add_argument("--bootstrap", type=int, default=500)
    doctor = commands.add_parser("doctor", help="Check connection configuration without inference")
    connection_arguments(doctor)
    doctor.add_argument("--model", help="Check the resolved route for this model")
    registry = commands.add_parser("judges", help="List configured judge IDs, versions, and call counts")
    registry.add_argument("--judges-dir", type=Path, default=DEFAULT_JUDGES_DIR)
    inspect = commands.add_parser("inspect", help="Validate joins and summarize available data")
    inspect.add_argument("--verify", action="store_true", help="Verify six input files against the SSOT manifest")
    listing = commands.add_parser("reviews", help="List review IDs and their lesson-plan IDs")
    listing.add_argument("--limit", type=int, default=10)
    judge = commands.add_parser("judge", help="Judge one review or a bounded batch of human-reviewed tasks")
    connection_arguments(judge)
    selection = judge.add_mutually_exclusive_group(required=True)
    selection.add_argument("--grading-id", help="Review ID supplies pretest and exact rubric/scale versions")
    selection.add_argument("--limit", type=int, help="First N distinct tasks in stable review-ID order")
    judge.add_argument("--plans", nargs="+", help="One or two SSOT plan IDs; requires --grading-id")
    judge.add_argument("--mode", choices=("single", "pairwise"), default="pairwise")
    judge.add_argument("--slot", choices=("a", "b"), default="a", help="Plan to rate in single mode")
    judge.add_argument("--swap", action="store_true", help="Reverse A/B to audit position bias")
    identity = judge.add_mutually_exclusive_group()
    identity.add_argument("--judge", help="Configured judge ID, e.g. judge_1")
    identity.add_argument("--model", help="Ad hoc LiteLLM provider/model identifier")
    judge.add_argument("--judges-dir", type=Path, default=DEFAULT_JUDGES_DIR)
    judge.add_argument("--reasoning-effort")
    judge.add_argument("--dry-run", action="store_true", help="Write exact prompts; no model import or API call")
    judge.add_argument("--output", type=Path,
                       help=f"JSONL results (default: {DEFAULT_RESULTS}); with --dry-run, "
                            "write JSON prompt previews here instead of stdout")
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
    if args.command in {"benchmark", "generate", "evaluate", "evaluate-many", "doctor"}:
        run_workflow(args)
        return
    if args.command == "judges":
        if not args.judges_dir.is_dir():
            raise ValueError(f"Judge configuration directory not found: {args.judges_dir}")
        definitions = [load_definition(path.stem, args.judges_dir)
                       for path in sorted(args.judges_dir.glob("*.toml"))]
        print(json.dumps([{"judge_id": d.judge_id, "version": d.version, "strategy": d.strategy,
                           "calls_per_comparison": d.call_count, "description": d.description,
                           "members": [m.configuration() for m in d.members]}
                          for d in definitions], indent=2))
        return
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


def run_workflow(args: argparse.Namespace) -> None:
    if args.command == "doctor":
        result = llm_client(args).doctor(args.model)
    elif args.command == "benchmark":
        from .benchmark import Benchmark, prepare_benchmark

        if args.benchmark_command == "prepare":
            result = prepare_benchmark(args.ssot, args.output, study=args.study, source_bank=args.source_bank)
        else:
            benchmark = Benchmark.load(args.path)
            result = {"benchmark": str(args.path), "instrument": benchmark.instrument,
                      "fingerprint": benchmark.fingerprint, "subjects": len(benchmark.contexts),
                      "baseline_plans": len(benchmark.plans), "models": benchmark.models,
                      "judge_id": benchmark.judge_id, "baseline_records": len(benchmark.baseline_records)}
    else:
        from .benchmark import Benchmark

        benchmark = Benchmark.load(args.benchmark)
        destination = (args.output if args.command in {"generate", "evaluate-many"} else
                       args.output or args.generation_dir / "evaluation")
        if destination.resolve().is_relative_to(args.ssot.resolve()):
            raise ValueError("Output must be outside the read-only SSOT directory")
        if args.command in {"evaluate", "evaluate-many"} and args.cache_dir.resolve().is_relative_to(args.ssot.resolve()):
            raise ValueError("Judge cache must be outside the read-only SSOT directory")
        client = llm_client(args)
        settings = {"dry_run": args.dry_run, "max_new_calls": args.max_new_calls,
                    "workers": args.workers, "timeout": args.timeout}
        if args.command == "generate":
            from .generation import run_generation

            result = run_generation(benchmark, args.output, args.model, client,
                                    reasoning_effort=args.reasoning_effort, limit=args.limit, **settings)
        elif args.command == "evaluate-many":
            from .evaluation import run_multi_evaluation

            result = run_multi_evaluation(benchmark, args.generation_dirs, client, output=args.output,
                                          cache_dir=args.cache_dir, include_new_pairs=args.include_new_pairs,
                                          comparisons_per_student=args.comparisons_per_student,
                                          seed=args.seed, bootstrap=args.bootstrap, **settings)
        else:
            from .evaluation import run_evaluation

            result = run_evaluation(benchmark, args.generation_dir, client, output=args.output,
                                    cache_dir=args.cache_dir,
                                    comparisons_per_student=args.comparisons_per_student,
                                    seed=args.seed, bootstrap=args.bootstrap, **settings)
    print_json(result)
    if args.command == "doctor" and not result.get("ok", False):
        raise ValueError("Connection check failed; resolve the reported configuration or connectivity problem")
    if not getattr(args, "dry_run", False) and result.get("complete") is False:
        raise ValueError("Run is incomplete; inspect the saved failures and rerun to resume")


def run_judge(args: argparse.Namespace, dataset: Dataset) -> None:
    if args.plans and not args.grading_id:
        raise ValueError("--plans requires --grading-id for the rubric and pretest context")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.dry_run and not (args.judge or args.model):
        raise ValueError("Model calls require --judge or --model")
    if args.judge and args.reasoning_effort:
        raise ValueError("Set reasoning_effort in the judge TOML; its subversion updates automatically")
    destination = args.output
    if destination is None and not args.dry_run:
        destination = DEFAULT_RESULTS
    path = output_path(destination, dataset) if destination is not None else None
    client = llm_client(args) if args.judge or args.model else None
    judge = (ConfiguredJudge(load_definition(args.judge, args.judges_dir), client=client) if args.judge else
             LiteLLMJudge(args.model or "dry-run", args.reasoning_effort, client=client))
    ids = [args.grading_id] if args.grading_id else sorted(dataset.sessions)
    tasks, seen = [], set()
    for grading_id in ids:
        task = make_task(dataset, grading_id, args.plans, args.mode, args.slot, args.swap)
        if isinstance(judge, ConfiguredJudge):
            judge.schedule(task)  # Validate the two-plan strategy before any provider call.
        request_id = (configured_request_id(task, judge) if isinstance(judge, ConfiguredJudge) else
                      digest({"task": task.metadata(), "judge": judge.configuration()}))
        if request_id not in seen:
            seen.add(request_id)
            tasks.append((request_id, task))
        if args.limit and len(tasks) >= args.limit:
            break
    if args.dry_run:
        previews = []
        for request_id, task in tasks:
            detail = (judge.preview(task) if isinstance(judge, ConfiguredJudge) else
                      {"messages": task.messages(), "response_schema": task.schema(), "call_count": 1})
            previews.append({"request_id": request_id, **task.metadata(), **detail})
        emit({"calls": sum(item["call_count"] for item in previews), "tasks": previews}, path, dataset)
        return
    previous = records_from(path) if path.exists() else []
    if isinstance(judge, ConfiguredJudge):
        check_strategy_history(judge.configuration(), previous)
    existing = {r["request_id"] for r in previous}
    completed = 0
    pending = sum(request_id not in existing for request_id, _ in tasks)
    if isinstance(judge, ConfiguredJudge):
        print(f"Using {judge.definition.version}: {pending} comparisons, "
              f"{pending * judge.call_count} model calls; results: {path}", flush=True)
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
