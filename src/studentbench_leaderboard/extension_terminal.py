"""Validate explicit terminal missing outcomes after bounded failed API attempts.

This module never writes declarations, invents ratings, calls a provider, or
changes a schedule. A terminal declaration is an execution status, not a tie.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from .extension import _prepare, _request_id, _response_metadata, request_contract, sha256
from .extension_parser import parse_provider_response
from .structured_judge import parse_judge_response
from .study import call_path

MINIMUM_FAILED_ATTEMPTS = 3


def job_key(job: dict) -> tuple[str, str, int]:
    return job["variant"], job["pair_id"], job.get("draw", 0)


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an explicit timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _validate_attempt(study: Path, evidence: dict, prepared, provider: dict | None,
                      declared_at: datetime) -> dict:
    if not isinstance(evidence, dict) or set(evidence) != {"file", "sha256"}:
        raise ValueError("Terminal evidence contains exactly file and sha256")
    if not isinstance(evidence["file"], str):
        raise ValueError("Terminal evidence file must be a path")
    path = (study / evidence["file"]).resolve()
    error_root = (study / "errors").resolve()
    if not path.is_relative_to(error_root) or not path.is_file() or path.suffix != ".json":
        raise ValueError("Terminal evidence must be an existing saved error under this study/errors")
    if sha256(path) != evidence["sha256"]:
        raise ValueError("Terminal evidence byte hash changed")
    error = json.loads(path.read_text())
    if job_key(error) != job_key(prepared.job):
        raise ValueError("Terminal evidence belongs to another scheduled job")
    run_id, attempt = error.get("run_id"), error.get("attempt")
    if not isinstance(run_id, str) or not run_id or type(attempt) is not int or attempt < 1:
        raise ValueError("Terminal evidence requires a recorded run_id and positive attempt number")
    observed_at = _timestamp(error.get("created_at"), "Failed attempt created_at")
    if observed_at > declared_at:
        raise ValueError("Terminal declaration predates its failed-attempt evidence")
    transport = error.get("transport")
    if not isinstance(transport, dict):
        raise ValueError("Terminal evidence lacks its exact request contract")
    reconstructed = request_contract(prepared.call_task, prepared.variant,
                                     response_format=transport.get("response_format"),
                                     timeout=prepared.contract["timeout_seconds"])
    if (transport != reconstructed or error.get("request_id") !=
            _request_id(reconstructed, prepared.judge, prepared.job["draw"])):
        raise ValueError("Terminal attempt differs from the reconstructed frozen request")
    metadata = error.get("response_metadata") or {}
    payload = metadata.get("raw_response_payload") or error.get("raw_response_payload") or {}
    response_id = payload.get("id") or metadata.get("response_id")
    response_id = response_id if isinstance(response_id, str) and response_id else None
    remote_status = type(error.get("status_code")) is int and 400 <= error["status_code"] <= 599
    transport_failure = error.get("type") in {
        "APIConnectionError", "APITimeoutError", "Timeout", "TimeoutError", "RateLimitError"}
    if not (response_id or remote_status or transport_failure):
        raise ValueError("Terminal evidence does not establish a failed API attempt")
    # An old parser error is not terminal if the complete saved response is now
    # accepted losslessly. Recover it instead of discarding a valid judgment.
    recoverable = False
    try:
        if provider and payload:
            raw, _ = _response_metadata(payload, provider)
            parse_provider_response(raw, prepared.call_task)
            recoverable = True
        elif not provider and isinstance(error.get("raw_response"), str):
            parse_judge_response(error["raw_response"], prepared.call_task)
            recoverable = True
    except (KeyError, TypeError, ValueError):
        pass
    if recoverable:
        raise ValueError("A saved attempt is recoverable and cannot support a terminal missing outcome")
    return {"file": str(path.relative_to(study.resolve())), "sha256": evidence["sha256"],
            "run_id": run_id, "attempt": attempt, "response_id": response_id,
            "created_at": error["created_at"], "type": error.get("type"),
            "status_code": error.get("status_code"), "request_id": error["request_id"]}


def load_terminal_failures(study: Path, scheduled_jobs: list[dict], *,
                           ssot: Path = Path("studentbench_data_ssot"),
                           scope_only: bool = False) -> dict[tuple[str, str, int], dict]:
    """Return validated declarations; optional phase scoping never reads other outcomes.

    The final verifier uses the complete original schedule with scope_only=False.
    A phase-only analysis can skip declarations outside its own job list; this
    preserves the example-only path's separation from provider outcomes.
    """
    study = Path(study)
    path = study / "terminal_failures.json"
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    if (not isinstance(document, dict) or document.get("schema_version") != 1 or
            not isinstance(document.get("jobs"), list)):
        raise ValueError("Terminal failures require schema_version=1 and a jobs list")
    scheduled = {job_key(job): {**job, "draw": job.get("draw", 0)} for job in scheduled_jobs}
    declarations = {}
    all_keys = set()
    for row in document["jobs"]:
        if not isinstance(row, dict) or not {"variant", "pair_id", "status", "reason", "declared_at", "evidence"} <= set(row):
            raise ValueError("Terminal job declaration is missing required fields")
        if set(row) - {"variant", "pair_id", "draw", "status", "reason", "declared_at", "evidence"}:
            raise ValueError("Terminal job declaration has unknown fields")
        key = job_key(row)
        if key in all_keys:
            raise ValueError("Terminal job declarations must be unique")
        all_keys.add(key)
        if key not in scheduled:
            if scope_only:
                continue
            raise ValueError("Terminal declaration is not in the original scheduled job universe")
        if row["status"] != "terminal_failure" or not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("Terminal job requires explicit terminal_failure status and reason")
        if type(row.get("draw", 0)) is not int or row.get("draw", 0) < 0:
            raise ValueError("Terminal draw must be a nonnegative integer")
        _timestamp(row["declared_at"], "Terminal declared_at")
        if not isinstance(row["evidence"], list) or len(row["evidence"]) < MINIMUM_FAILED_ATTEMPTS:
            raise ValueError("Terminal job requires at least three distinct failed API attempts")
        if call_path(study, scheduled[key]).exists():
            raise ValueError("A completed call cannot be declared a terminal missing outcome")
        declarations[key] = {**row, "draw": row.get("draw", 0)}
    if not declarations:
        return {}
    availability = study / "availability.json"
    unavailable = set(json.loads(availability.read_text()).get("exclusions", {})) if availability.exists() else set()
    if any(key[0] in unavailable for key in declarations):
        raise ValueError("Provider availability exclusions must remain separate from terminal individual jobs")
    pending, _, providers, _ = _prepare(study, [scheduled[key] for key in declarations], 420, ssot=ssot)
    prepared = {job_key(job.job): job for job in pending}
    result = {}
    for key, row in declarations.items():
        current = prepared[key]
        model = current.variant["model"]
        provider = providers.get(model.removeprefix("openrouter/")) if model.startswith("openrouter/") else None
        attempts = [_validate_attempt(study, evidence, current, provider, _timestamp(row["declared_at"], "Terminal declared_at"))
                    for evidence in row["evidence"]]
        if (len({attempt["file"] for attempt in attempts}) != len(attempts) or
                len({(attempt["run_id"], attempt["attempt"]) for attempt in attempts}) != len(attempts)):
            raise ValueError("Terminal evidence must identify distinct failed API attempts")
        response_ids = [attempt["response_id"] for attempt in attempts if attempt["response_id"] is not None]
        if len(set(response_ids)) != len(response_ids):
            raise ValueError("Repeated copies of one provider response are not distinct failed attempts")
        result[key] = {**row, "n_failed_api_attempts": len(attempts), "verified_attempts": attempts,
                       "outcome": "missing", "rating_imputed": False}
    return result


def terminal_failure_summary(study: Path, failures: dict) -> dict:
    path = Path(study) / "terminal_failures.json"
    return {"file": path.name if path.exists() else None, "sha256": sha256(path) if path.exists() else None,
            "n_jobs": len(failures), "by_variant": dict(Counter(key[0] for key in failures)),
            "details": [failures[key] for key in sorted(failures)],
            "interpretation": "Terminal jobs retain missing outcomes after at least three verified failed API attempts. "
                              "They are not ties, abstentions or fabricated ratings, and the original schedule stays intact."}
