"""Explicit offline recovery of losslessly compatible provider serialization errors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .data import digest
from .extension import (
    DEFAULT_STUDY, _normalise, _prepare, _response_metadata, _validate_cache,
    openrouter_cost, sha256,
)
from .extension_parser import PARSER_V2, parse_provider_response
from .study import call_path, write_json

RECOVERY_CONTRACT = {
    "version": "extension-saved-format-recovery.v1",
    "eligibility": "missing-openrouter-job-with-exact-original-strict-schema-request-and-complete-saved-response",
    "selection": "earliest-compatible-error-created_at-then-relative-filename",
    "mutations": "new-success-record-only-source-errors-and-existing-successes-unchanged",
    "billing": "recovered-source-error-is-the-same-paid-attempt-as-the-success-record-count-it-once",
}


def recovery_candidate(study: Path, path: Path, prepared, provider: dict, catalog_sha: str,
                       *, parser_version: str = PARSER_V2) -> dict:
    """Reconstruct one saved attempt. No API imports or adapter invocations."""
    study, path = study.resolve(), path.resolve()
    if not path.is_relative_to(study):
        raise ValueError("Recovery source must be inside the extension directory")
    error = json.loads(path.read_text())
    if any(error.get(key, 0 if key == "draw" else None) != value for key, value in prepared.job.items()):
        raise ValueError("Saved error belongs to a different job")
    if (error.get("transport") != prepared.contract or error.get("request_id") != prepared.requested_id or
            prepared.contract["response_format"] != "json_schema"):
        raise ValueError("Recovery requires the exact original strict-schema request")
    if not isinstance(error.get("created_at"), str):
        raise ValueError("Saved error lacks a response observation timestamp")
    observed = datetime.fromisoformat(error["created_at"])
    if observed.tzinfo is None:
        raise ValueError("Saved error timestamp must include a timezone")
    metadata = error.get("response_metadata") or {}
    payload = metadata.get("raw_response_payload")
    if not isinstance(payload, dict):
        raise ValueError("Recovery requires the complete original provider payload")
    raw, returned_metadata = _response_metadata(payload, provider)
    if raw != error.get("raw_response") or any(metadata.get(key) != value for key, value in returned_metadata.items()):
        raise ValueError("Saved response text/metadata disagrees with the raw provider payload")
    ratings, parser = parse_provider_response(raw, prepared.call_task, version=parser_version)
    cost, provenance = openrouter_cost(returned_metadata["usage"], provider, catalog_sha)
    record = {**prepared.job, "subject": prepared.task.subject, "plan_ids": prepared.task.plan_ids,
              "presented_plan_ids": prepared.call_task.plan_ids, "order": "BA" if prepared.swapped else "AB",
              "split": "validation", "human_reviewed": prepared.entry["human_reviewed"],
              "model_a": prepared.entry["model_a"], "model_b": prepared.entry["model_b"],
              "grading_id": prepared.entry["grading_id"], "criteria": prepared.task.metadata()["criteria"],
              "judge": prepared.judge, "request_id": prepared.requested_id, "requested_request_id": prepared.requested_id,
              "messages": prepared.contract["body"]["messages"], "requested_messages": prepared.call_task.messages(),
              "response_schema": prepared.call_task.schema(), "raw_response": raw, "ratings": ratings,
              "normalized_ratings": _normalise(ratings, prepared.swapped), "dataset_sha256": prepared.task.dataset_sha256,
              "runner_sha256": error.get("runner_sha256"), "created_at": error["created_at"],
              "duration_seconds": error.get("duration_seconds"), "attempts": error.get("attempt"),
              "estimated_usd": cost, "cost_provenance": provenance, "run_id": error.get("run_id"),
              "requested_response_format": "json_schema", "effective_response_format": "json_schema",
              "transport": {"requested": prepared.contract, "effective": prepared.contract,
                            "compatibility_fallbacks": []},
              "recovered_from_error": {"file": str(path.relative_to(study)), "sha256": sha256(path)},
              "recovery_contract": RECOVERY_CONTRACT,
              "recovery_contract_sha256": digest(RECOVERY_CONTRACT),
              "recovery_source_sha256": sha256(Path(__file__)),
              "recovered_at": datetime.now(timezone.utc).isoformat(),
              "recovery_observation_note": "created_at is the source error observation time. Original runner hash and duration "
                                           "remain null when not recorded in that saved attempt.",
              **returned_metadata, **parser}
    record["evaluation_id"] = digest({"request_id": prepared.requested_id,
                                      "parser_contract_sha256": parser["parser_contract_sha256"]})
    if prepared.context_provenance is not None:
        record["pretest_context_provenance"] = prepared.context_provenance
    return record


def validate_recovered_record(study: Path, record: dict, prepared, provider: dict, catalog_sha: str) -> None:
    """Revalidate the immutable source attempt and deterministic recovery choice."""
    source = record["recovered_from_error"]
    path = (study / source["file"]).resolve()
    if not path.is_relative_to(study.resolve()) or not path.is_file() or sha256(path) != source["sha256"]:
        raise ValueError("Recovered source error path or byte hash changed")
    expected = recovery_candidate(study, path, prepared, provider, catalog_sha, parser_version=record["parser_version"])
    # These observation/provenance values are copied, not regenerated timings.
    for key in ("created_at", "run_id", "attempts", "duration_seconds", "runner_sha256", "raw_response",
                "recovery_contract", "recovery_contract_sha256", "recovered_from_error"):
        if record.get(key) != expected[key]:
            raise ValueError(f"Recovered source metadata changed: {key}")
    candidates = []
    for other in (study / "errors").glob(f"{prepared.job['variant']}--{prepared.job['pair_id']}--{prepared.job['draw']}--*.json"):
        try:
            candidate = recovery_candidate(study, other, prepared, provider, catalog_sha, parser_version=record["parser_version"])
            candidates.append((datetime.fromisoformat(candidate["created_at"]).timestamp(),
                               str(other.relative_to(study)), candidate["recovered_from_error"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates or min(candidates, key=lambda row: row[:2])[2] != source:
        raise ValueError("Recovered attempt is not the earliest compatible saved response")


def recover_jobs(study: Path, jobs: list[dict], *, write: bool = False, timeout: int = 420) -> dict:
    """Preview or explicitly write recoverable missing provider jobs, with no API."""
    study = Path(study)
    pending, reused, providers, catalog_sha = _prepare(study, jobs, timeout)
    availability = study / "availability.json"
    excluded = set(json.loads(availability.read_text()).get("exclusions", {})) if availability.exists() else set()
    error_index = {}
    for path in sorted((study / "errors").glob("*.json")):
        error = json.loads(path.read_text())
        key = (error.get("variant"), error.get("pair_id"), error.get("draw", 0))
        error_index.setdefault(key, []).append(path)
    output = []
    for prepared in pending:
        job = prepared.job
        model = prepared.variant["model"]
        if job["variant"] in excluded or not model.startswith("openrouter/"):
            output.append({**job, "status": "ineligible", "reason": "Excluded availability or native route"})
            continue
        provider = providers[model.removeprefix("openrouter/")]
        candidates, rejected = [], []
        for path in error_index.get((job["variant"], job["pair_id"], job["draw"]), []):
            try:
                record = recovery_candidate(study, path, prepared, provider, catalog_sha)
                timestamp = datetime.fromisoformat(record["created_at"]).timestamp()
                candidates.append((timestamp, str(path.relative_to(study)), record))
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append({"file": str(path.relative_to(study)), "reason": str(exc)})
        if not candidates:
            output.append({**job, "status": "unrecoverable", "rejected_attempts": rejected})
            continue
        candidates.sort(key=lambda row: row[:2])
        record = candidates[0][2]
        if write:
            path = call_path(study, job)
            if path.exists():
                raise ValueError("A success appeared during recovery; refusing to overwrite it")
            # Validate in a distinct temporary file before publishing the cache.
            temporary = path.with_name(path.name + ".recovery-validation")
            try:
                write_json(temporary, record)
                _validate_cache(temporary, prepared, {}, catalog_sha, provider)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        output.append({**job, "status": "recovered" if write else "recoverable",
                       "recovered_from_error": record["recovered_from_error"],
                       "transformations": record["transformations"],
                       "compatible_attempts": len(candidates),
                       "other_compatible_attempt_files": [row[1] for row in candidates[1:]],
                       "rejected_attempts": rejected})
    result = {"schema_version": 1, "recovery_contract": RECOVERY_CONTRACT, "network_calls": 0,
              "write": write, "existing_successes_reused": reused, "missing_jobs_examined": len(pending),
              "recovered": sum(row["status"] == "recovered" for row in output),
              "recoverable": sum(row["status"] == "recoverable" for row in output), "jobs": output}
    if write:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        write_json(study / "recoveries" / f"{stamp}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--write", action="store_true", help="Explicitly publish new recovered caches; default is preview only")
    args = parser.parse_args()
    print(json.dumps(recover_jobs(args.study, json.loads(args.jobs.read_text()), write=args.write), indent=2))


if __name__ == "__main__":
    main()
