"""Explicit, resumable validation-extension calls through OpenRouter or native OpenAI.

This runner never changes the original study. Cached controls require a trusted
import ledger; every other cache is reconstructed and strictly reparsed before
any new API request is dispatched.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import tomllib
import uuid

from .comparison import ConfiguredJudge
from .data import Dataset, digest
from .extension_parser import parse_provider_response
from .judge import LiteLLMJudge, Task, make_task
from .judge_config import load_definition
from .structured_judge import parse_judge_response
from .study import call_path, catalog, estimated_cost, write_json

DEFAULT_STUDY = Path("results/ranking-extension-20260911")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CONTRACT_VERSION = "validation-extension.v1"
FORMATS = ("json_schema", "json_object", "prompt_json")
PROVIDER_POLICY = {"allow_fallbacks": True, "require_parameters": True}
LEGACY_MAX_OUTPUT_TOKENS = 32768
DEFAULT_WORKERS = 128
DEFAULT_PER_MODEL_LIMIT = 128
MAX_CONCURRENCY = 512


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_:.+-]+", value) is not None


def _redact(message: str) -> str:
    for key, value in os.environ.items():
        if any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) > 8:
            message = message.replace(value, "[REDACTED]")
    return message[:4000]


def _normalise(ratings: dict, swapped: bool) -> dict:
    return {key: {**value, "score": -value["score"] if swapped and value["score"] is not None
                  else value["score"]} for key, value in ratings.items()}


def _schema_format(schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "lesson_plan_judgment", "strict": True, "schema": schema}}


def request_contract(task: Task, variant: dict, *, response_format: str = "json_schema",
                     timeout: int = 420) -> dict:
    """The exact semantic request and transport knobs, without credentials."""
    if response_format not in FORMATS:
        raise ValueError("Unknown response format")
    model = variant["model"]
    messages = task.messages()
    schema = task.schema()
    if response_format != "json_schema":
        # JSON mode does not transmit a schema. Preserve its meaning explicitly
        # in the actual prompt, and retain this change in the request identity.
        messages = [dict(message) for message in messages]
        messages[0]["content"] += ("\nReturn one JSON object matching this exact JSON Schema. "
                                    "Do not include Markdown fences or other text.\n" +
                                    json.dumps(schema, ensure_ascii=False, allow_nan=False, sort_keys=True))
    if model.startswith("openrouter/"):
        exact_model = model.removeprefix("openrouter/")
        if "/" not in exact_model or exact_model.startswith(("openrouter/", "~")):
            raise ValueError("OpenRouter requires one explicit provider/model ID")
        # Missing metadata retains the original contract for cache reconstruction.
        # New calls must choose an explicit cap or null before dispatch below.
        max_tokens = variant.get("max_tokens", LEGACY_MAX_OUTPUT_TOKENS)
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("OpenRouter max_tokens must be a positive integer or null")
        body = {"model": exact_model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body["provider"] = dict(PROVIDER_POLICY)
        if variant.get("reasoning_effort"):
            body["reasoning"] = {"effort": variant["reasoning_effort"]}
        adapter = "openrouter-openai-sdk"
        endpoint = OPENROUTER_BASE_URL + "/chat/completions"
    elif model.startswith("openai/"):
        if response_format != "json_schema":
            raise ValueError("Native control uses its original strict schema adapter")
        if variant.get("max_tokens") is not None:
            raise ValueError("Native LiteLLM adapter does not send max_tokens")
        body = {"model": model, "messages": messages}
        if variant.get("reasoning_effort"):
            body["reasoning_effort"] = variant["reasoning_effort"]
        adapter = "litellm-native"
        endpoint = "native-openai/chat/completions"
    else:
        raise ValueError("Extension models require an explicit openrouter/ or openai/ route")
    if response_format == "json_schema":
        body["response_format"] = _schema_format(schema)
    elif response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    return {"version": CONTRACT_VERSION, "adapter": adapter, "endpoint": endpoint,
            "body": body, "timeout_seconds": timeout, "sdk_retries": 0,
            "response_format": response_format, "semantic_schema_sha256": digest(schema)}


def _request_id(contract: dict, judge: dict, draw: int) -> str:
    return digest({"transport": contract, "judge": judge, "draw": draw})


def _format_incompatibility(exc: Exception) -> bool:
    """Only a clear capability rejection permits a format change, never bad JSON."""
    if getattr(exc, "status_code", None) not in (400, 404, 422):
        return False
    message = str(exc).lower()
    format_named = any(word in message for word in
                       ("response_format", "response format", "json_schema", "json schema", "structured output"))
    unsupported = any(word in message for word in
                      ("not supported", "unsupported", "does not support", "no endpoints", "unavailable"))
    return format_named and unsupported


def _transient(exc: Exception) -> bool:
    return (getattr(exc, "status_code", None) in (408, 429, 500, 502, 503, 504) or
            type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "Timeout", "TimeoutError", "RateLimitError"))


class ResponseError(ValueError):
    """A returned response failed validation; retain its original payload."""

    def __init__(self, message: str, payload: dict, status_code: int | None = None):
        super().__init__(message)
        self.payload = payload
        self.status_code = status_code


def _response_metadata(payload: dict, model_entry: dict) -> tuple[str, dict]:
    # A gateway can send its error envelope after committing HTTP 200. Preserve
    # the envelope and classify its explicit status just like an HTTP error.
    if error := payload.get("error"):
        status = error.get("code") if isinstance(error, dict) else None
        if type(status) is not int or not 400 <= status <= 599:
            status = None
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise ResponseError(f"OpenRouter error response: {message}", payload, status)
    choices = payload.get("choices") or []
    if len(choices) == 1 and (error := choices[0].get("error")):
        status = error.get("code") if isinstance(error, dict) else None
        if type(status) is not int or not 400 <= status <= 599:
            status = None
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise ResponseError(f"OpenRouter choice error response: {message}", payload, status)
    expected = {model_entry["id"], model_entry.get("canonical_slug")}
    if payload.get("model") not in expected - {None}:
        raise ResponseError("Returned model differs from the explicitly requested catalog model", payload)
    if len(choices) != 1:
        raise ResponseError("Expected exactly one completion choice", payload)
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ResponseError(f"Incomplete completion: {choice.get('finish_reason')!r}", payload)
    message = choice.get("message") or {}
    if not isinstance(message.get("content"), str) or message.get("refusal"):
        raise ResponseError("Judge returned no text or refused", payload)
    return message["content"], {"usage": payload.get("usage") or {}, "response_model": payload["model"],
                                "response_id": payload.get("id"), "response_provider": payload.get("provider"),
                                "raw_response_payload": payload}


def openrouter_complete(contract: dict, model_entry: dict) -> tuple[str, dict]:
    """One HTTP attempt. All retries and compatibility decisions live above it."""
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required for new OpenRouter calls")
    body = contract["body"]
    kwargs = {key: body[key] for key in ("model", "messages", "max_tokens", "response_format") if key in body}
    kwargs["extra_body"] = {key: body[key] for key in ("provider", "reasoning") if key in body}
    # Explicit base URL prevents an environment override from changing routing.
    with OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, max_retries=0,
                timeout=contract["timeout_seconds"]) as client:
        response = client.chat.completions.create(**kwargs)
    payload = response.model_dump(mode="json")
    return _response_metadata(payload, model_entry)


def openrouter_cost(usage: dict, model_entry: dict, catalog_sha256: str) -> tuple[float, dict]:
    """Prefer returned billing; catalog fallback is explicitly only nominal."""
    returned = usage.get("cost")
    if isinstance(returned, (int, float)) and not isinstance(returned, bool) and math.isfinite(returned) and returned >= 0:
        return float(returned), {"source": "openrouter.usage.cost", "is_estimate": False,
                                 "catalog_sha256": catalog_sha256, "model_id": model_entry["id"]}
    pricing = model_entry.get("pricing") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (prompt, completion)):
        raise ValueError("No returned cost or usable token counts for cost estimation")
    cache = usage.get("prompt_tokens_details") or {}
    cached = min(prompt, max(0, cache.get("cached_tokens", 0) or 0))
    written = min(prompt - cached, max(0, cache.get("cache_write_tokens", 0) or 0))
    input_rate = float(pricing["prompt"])
    output_rate = float(pricing["completion"])
    cached_rate = float(pricing.get("input_cache_read", input_rate))
    written_rate = float(pricing.get("input_cache_write", input_rate))
    # completion_tokens already includes reasoning; do not add reasoning again.
    value = ((prompt - cached - written) * input_rate + cached * cached_rate + written * written_rate +
             completion * output_rate + float(pricing.get("request", "0")))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid catalog pricing")
    return value, {"source": "openrouter.catalog.nominal", "is_estimate": True,
                   "catalog_sha256": catalog_sha256, "model_id": model_entry["id"],
                   "pricing": pricing, "note": "Nominal catalog rates; endpoint, time and context overrides may differ."}


def _failed_attempt_billing(error: dict, model: str, provider: dict | None, catalog_sha: str) -> dict:
    """Keep any returned usage even when a paid response failed strict parsing."""
    usage = ((error.get("response_metadata") or {}).get("usage") or
             (error.get("raw_response_payload") or {}).get("usage") or {})
    if usage:
        try:
            if provider:
                cost, provenance = openrouter_cost(usage, provider, catalog_sha)
            elif all(type(usage.get(key)) is int and usage[key] >= 0
                     for key in ("prompt_tokens", "completion_tokens")):
                cost = estimated_cost(model, usage)
                provenance = {"source": "native.fixed-token-prices", "is_estimate": True}
            else:
                raise ValueError("Missing native token counts")
            return {"usage": usage, "estimated_usd": cost, "cost_provenance": provenance}
        except (KeyError, TypeError, ValueError):
            pass
    return {"usage": usage, "estimated_usd": None,
            "cost_provenance": {"source": "unavailable", "is_estimate": None,
                                "note": "Provider did not return usable cost/token usage for this failed attempt."}}


def economic_cost(record: dict) -> dict:
    """Report BYOK upstream inference separately without changing cached charges.

    OpenRouter's usage.cost can be only the gateway charge when a customer uses
    their own upstream key. In that case add reported upstream inference cost
    once; for ordinary OpenRouter billing it is already included in the charge.
    Missing BYOK upstream cost remains unknown, never a zero-cost inference.
    """
    def amount(value):
        return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 else None)

    usage = (record.get("usage") or (record.get("response_metadata") or {}).get("usage") or
             (record.get("raw_response_payload") or {}).get("usage") or {})
    provenance = record.get("cost_provenance") or {}
    transport = record.get("transport") or {}
    effective = transport.get("effective", transport)
    is_router = (effective.get("adapter") == "openrouter-openai-sdk" or
                 provenance.get("source", "").startswith("openrouter.") or
                 "is_byok" in usage or "cost_details" in usage)
    router = amount(usage.get("cost")) if is_router else None
    details = usage.get("cost_details") or {}
    upstream = amount(details.get("upstream_inference_cost")) if is_router else None
    byok = usage.get("is_byok") if type(usage.get("is_byok")) is bool else None
    legacy = amount(record.get("estimated_usd"))
    if is_router and byok:
        value = router + upstream if router is not None and upstream is not None else None
        basis = "router-charge-plus-reported-byok-upstream" if value is not None else "byok-upstream-or-router-charge-unavailable"
        estimate = False if value is not None else None
    elif is_router and router is not None:
        value, basis, estimate = router, "returned-openrouter-charge", False
    elif legacy is not None:
        value, basis, estimate = legacy, provenance.get("source", "legacy-native-token-estimate"), True
    else:
        value, basis, estimate = None, "unavailable", None
    return {"economic_usd": value, "router_charge_usd": router, "upstream_inference_usd": upstream,
            "is_byok": byok, "basis": basis, "is_estimate": estimate, "complete": value is not None}


def _load_frozen(study: Path) -> tuple[dict, dict, dict, str]:
    variants_list = json.loads((study / "variants.json").read_text())
    variants = {variant["variant"]: variant for variant in variants_list}
    if len(variants) != len(variants_list) or not all(_safe_name(name) for name in variants):
        raise ValueError("Variant names must be unique safe path components")
    catalog_path = study / "provider_catalog.json"
    provider_catalog = json.loads(catalog_path.read_text())
    providers = {entry["id"]: entry for entry in provider_catalog["data"]}
    if len(providers) != len(provider_catalog["data"]):
        raise ValueError("Provider catalog contains duplicate IDs")
    judges = {}
    for name, variant in variants.items():
        directory = study / "configs" / name
        definition = load_definition(variant["judge_id"], directory)
        if (definition.strategy != "single" or len(definition.members) != 1 or
                definition.members[0].model != variant["model"] or
                definition.members[0].reasoning_effort != variant.get("reasoning_effort") or
                definition.response_style != variant["response_style"]):
            raise ValueError(f"Variant metadata differs from its frozen configuration: {name}")
        if context := variant.get("pretest_context"):
            declared = tomllib.loads((directory / "context_input.toml").read_text())
            if (declared["source_bank"] != context["file"] or declared["source_sha256"] != context["sha256"] or
                    declared["projection"] != definition.plan_projection):
                raise ValueError(f"Pretest context differs from frozen input configuration: {name}")
        if variant["model"].startswith("openrouter/"):
            exact_id = variant["model"].removeprefix("openrouter/")
            if exact_id not in providers:
                raise ValueError(f"Explicit model absent from provider catalog: {exact_id}")
            provider = providers[exact_id]
            if not {"structured_outputs", "response_format"} <= set(provider.get("supported_parameters", [])):
                raise ValueError(f"Catalog does not advertise schema support: {exact_id}")
            efforts = (provider.get("reasoning") or {}).get("supported_efforts", [])
            if variant.get("reasoning_effort") and efforts is not None and variant["reasoning_effort"] not in efforts:
                raise ValueError(f"Catalog does not advertise requested effort: {exact_id}")
        judges[name] = ConfiguredJudge(definition)
    inventory = {"manifest_sha256": sha256(study / "manifest.json"),
                 "provider_catalog_sha256": sha256(catalog_path),
                 "variants": {name: digest(value) for name, value in variants.items()},
                 "files": {str(path.relative_to(study)): sha256(path)
                           for path in (study / "configs").rglob("*") if path.is_file()}}
    inventory_path = study / "extension_inventory.json"
    if inventory_path.exists():
        previous = json.loads(inventory_path.read_text())
        if any(previous.get(key) != inventory[key] for key in ("manifest_sha256", "provider_catalog_sha256")):
            raise ValueError("Frozen manifest or provider catalog changed")
        if any(inventory[section].get(key) != value for section in ("variants", "files")
               for key, value in previous[section].items()):
            raise ValueError("An existing frozen judge configuration changed")
    if not inventory_path.exists() or json.loads(inventory_path.read_text()) != inventory:
        write_json(inventory_path, inventory)
    return variants, judges, providers, inventory["provider_catalog_sha256"]


@dataclass
class PreparedJob:
    job: dict
    entry: dict
    variant: dict
    task: Task
    call_task: Task
    judge: dict
    swapped: bool
    contract: dict
    requested_id: str
    context_provenance: dict | None


def _validate_cache(path: Path, prepared: PreparedJob, imported: dict, catalog_sha: str,
                    model_entry: dict | None) -> None:
    record = json.loads(path.read_text())
    p = prepared
    identities = {**p.job, "subject": p.task.subject, "plan_ids": p.task.plan_ids,
                  "presented_plan_ids": p.call_task.plan_ids, "split": "validation",
                  "order": "BA" if p.swapped else "AB", "model_a": p.entry["model_a"],
                  "model_b": p.entry["model_b"], "grading_id": p.entry["grading_id"],
                  "human_reviewed": p.entry["human_reviewed"], "dataset_sha256": p.task.dataset_sha256,
                  "criteria": p.task.metadata()["criteria"], "judge": p.judge,
                  "response_schema": p.call_task.schema()}
    for key, value in identities.items():
        if record.get(key) != value:
            raise ValueError(f"Cached {key} mismatch: {path}")
    if p.context_provenance is not None and record.get("pretest_context_provenance") != p.context_provenance:
        raise ValueError(f"Cached pretest provenance mismatch: {path}")
    if "parser_contract" in record:
        if model_entry is None:
            raise ValueError(f"Native/imported responses must retain the strict original parser: {path}")
        ratings, parser_metadata = parse_provider_response(record["raw_response"], p.call_task,
                                                            version=record.get("parser_version"))
        if any(record.get(key) != value for key, value in parser_metadata.items()):
            raise ValueError(f"Cached provider parser provenance mismatch: {path}")
        if record.get("evaluation_id") != digest({"request_id": record["request_id"],
                                                   "parser_contract_sha256": parser_metadata["parser_contract_sha256"]}):
            raise ValueError(f"Cached provider evaluation identity mismatch: {path}")
    else:
        # Pre-adapter successes remain governed by their original strict parser.
        ratings = parse_judge_response(record["raw_response"], p.call_task)
    if record.get("ratings") != ratings or record.get("normalized_ratings") != _normalise(ratings, p.swapped):
        raise ValueError(f"Cached result disagrees with current response parser: {path}")
    if "transport" not in record:
        trusted = imported.get(path.name)
        if not trusted or not p.variant["model"].startswith("openai/"):
            raise ValueError(f"Legacy cache requires an explicit trusted import ledger: {path}")
        legacy_id = digest({"messages": p.call_task.messages(), "schema": p.call_task.schema(),
                            "judge": p.judge, "draw": p.job["draw"]})
        if (trusted.get("sha256") != sha256(path) or trusted.get("request_id") != legacy_id or
                record.get("request_id") != legacy_id or record.get("messages") != p.call_task.messages()):
            raise ValueError(f"Imported cache identity or byte hash mismatch: {path}")
        return
    if path.name in imported:
        raise ValueError(f"Imported control was replaced by an extension-format record: {path}")
    transport = record["transport"]
    effective_format = record.get("effective_response_format")
    effective = request_contract(p.call_task, p.variant, response_format=effective_format,
                                 timeout=p.contract["timeout_seconds"])
    if (record.get("requested_request_id") != p.requested_id or transport.get("requested") != p.contract or
            transport.get("effective") != effective or record.get("requested_messages") != p.call_task.messages() or
            record.get("messages") != effective["body"]["messages"] or
            record.get("requested_response_format") != "json_schema" or
            record.get("request_id") != _request_id(effective, p.judge, p.job["draw"])):
        raise ValueError(f"Cached request changed: {path}")
    transitions = transport.get("compatibility_fallbacks", [])
    formats = ["json_schema"]
    for transition in transitions:
        if transition.get("from") != formats[-1] or transition.get("to") not in FORMATS:
            raise ValueError(f"Invalid cached format transition: {path}")
        if FORMATS.index(transition["to"]) != FORMATS.index(formats[-1]) + 1 or not transition.get("reason"):
            raise ValueError(f"Invalid cached format transition: {path}")
        formats.append(transition["to"])
    if formats[-1] != effective_format:
        raise ValueError(f"Missing cached format transition provenance: {path}")
    if "recovered_from_error" in record:
        if model_entry is None or "parser_contract" not in record:
            raise ValueError("Only explicitly versioned provider responses can be recovered")
        from .extension_recovery import validate_recovered_record
        validate_recovered_record(path.parent.parent, record, p, model_entry, catalog_sha)
    if model_entry:
        raw, metadata = _response_metadata(record["raw_response_payload"], model_entry)
        cost, provenance = openrouter_cost(metadata["usage"], model_entry, catalog_sha)
        if raw != record["raw_response"] or any(record.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Cached raw provider payload disagrees with metadata: {path}")
    else:
        cost = estimated_cost(p.variant["model"], record.get("usage", {}))
        provenance = {"source": "native.fixed-token-prices", "is_estimate": True}
    if record.get("estimated_usd") != cost or record.get("cost_provenance") != provenance:
        raise ValueError(f"Cached cost provenance mismatch: {path}")


def _prepare(study: Path, jobs: list[dict], timeout: int, *,
             ssot: Path = Path("studentbench_data_ssot")) -> tuple[list[PreparedJob], int, dict, str]:
    if not isinstance(jobs, list):
        raise ValueError("Jobs must be a unique list")
    normalised = []
    for job in jobs:
        if not isinstance(job, dict) or set(job) - {"variant", "pair_id", "draw"} or not {"variant", "pair_id"} <= set(job):
            raise ValueError("Each job contains only variant, pair_id and optional draw")
        draw = job.get("draw", 0)
        if not _safe_name(job["variant"]) or not _safe_name(job["pair_id"]) or type(draw) is not int or draw < 0:
            raise ValueError("Invalid job identity")
        normalised.append({**job, "draw": draw})
    if len({(job["variant"], job["pair_id"], job["draw"]) for job in normalised}) != len(normalised):
        raise ValueError("Jobs must be a unique list")
    manifest = json.loads((study / "manifest.json").read_text())
    data = Dataset(ssot, manifest["instrument"])
    if data.fingerprint != manifest["dataset_sha256"]:
        raise ValueError("SSOT fingerprint changed")
    variants, judges, providers, catalog_sha = _load_frozen(study)
    pairs = catalog(manifest)
    total_pairs = sum(len(entries) for entries in manifest["pair_catalog"].values())
    if len(pairs) != total_pairs:
        raise ValueError("Manifest contains duplicate pair IDs")
    imported_path = study / "imported_calls.json"
    imported = json.loads(imported_path.read_text())["calls"] if imported_path.exists() else {}
    banks = {}
    needed = {job["variant"] for job in normalised}
    for name in needed:
        if name not in variants:
            raise ValueError(f"Unknown variant: {name}")
        if context := variants[name].get("pretest_context"):
            from .pretest_context import PretestContextBank
            directory = (study / "configs" / name).resolve()
            source = (directory / context["file"]).resolve()
            if not source.is_relative_to(directory):
                raise ValueError("Pretest source must be inside its frozen configuration")
            banks[name] = PretestContextBank.load(data.root, source_bank=source,
                                                expected_source_sha256=context["sha256"])
    pending, reused = [], 0
    for job in normalised:
        if job["pair_id"] not in pairs or pairs[job["pair_id"]]["split"] != "validation":
            raise ValueError("This exposed-data extension only executes validation pairs")
        entry = pairs[job["pair_id"]]
        if (entry["plan_ids"] != sorted(entry["plan_ids"]) or
                entry["pair_id"] != digest([entry["subject"], *entry["plan_ids"]])[:20] or
                [data.plans[plan_id]["ai_model_preset_id"] for plan_id in entry["plan_ids"]] !=
                [entry["model_a"], entry["model_b"]]):
            raise ValueError("Manifest pair/model identity disagrees with actual lesson plans")
        task = make_task(data, entry["grading_id"], entry["plan_ids"])
        if task.subject != entry["subject"]:
            raise ValueError("Manifest subject disagrees with the grading task")
        name = job["variant"]
        provenance = None
        if name in banks:
            task = banks[name].enrich_task(task)
            provenance = banks[name].provenance
        swapped = bool(int(digest([manifest["seed"], entry["pair_id"]]), 16) % 2) ^ bool(job["draw"] % 2)
        ct = judges[name].call_task(task, swapped)
        judge = judges[name].configuration()
        contract = request_contract(ct, variants[name], timeout=timeout)
        prepared = PreparedJob(job, entry, variants[name], task, ct, judge, swapped, contract,
                               _request_id(contract, judge, job["draw"]), provenance)
        path = call_path(study, job)
        model = variants[name]["model"]
        provider = providers.get(model.removeprefix("openrouter/")) if model.startswith("openrouter/") else None
        if path.exists():
            _validate_cache(path, prepared, imported, catalog_sha, provider)
            reused += 1
        else:
            if path.name in imported:
                raise ValueError(f"Missing imported control must be restored, never rerun: {path}")
            pending.append(prepared)
    return pending, reused, providers, catalog_sha


def _dispatch_jobs(jobs: list[PreparedJob], perform, workers: int,
                   per_model_limit: int = DEFAULT_PER_MODEL_LIMIT) -> None:
    """Reserve model capacity before submission, so waiting jobs hold no worker.

    Model queues rotate after each admission. A slow model can occupy at most
    its own admitted slots; its backlog cannot strand otherwise idle workers.
    """
    queues = {}
    for job in jobs:
        queues.setdefault(job.variant["model"], deque()).append(job)
    rotation = deque(queues)
    active = {model: 0 for model in queues}
    limit = min(workers, per_model_limit)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        while any(queues.values()) or futures:
            while len(futures) < workers:
                admitted = None
                for _ in range(len(rotation)):
                    model = rotation.popleft()
                    rotation.append(model)
                    if queues[model] and active[model] < limit:
                        admitted = model
                        break
                if admitted is None:
                    break
                job = queues[admitted].popleft()
                active[admitted] += 1
                futures[executor.submit(perform, job)] = admitted
            if not futures:
                raise RuntimeError("Pending jobs cannot be admitted by the dispatcher")
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                model = futures.pop(future)
                active[model] -= 1
                future.result()


def run_jobs(study: Path, jobs: list[dict], workers: int = DEFAULT_WORKERS, max_new_calls: int = 1200,
             timeout: int = 420, max_attempts: int = 3, *, check_only: bool = False,
             per_model_limit: int = DEFAULT_PER_MODEL_LIMIT) -> dict:
    """Run explicit validation jobs; max_new_calls counts jobs, not retry attempts."""
    if type(workers) is not int or not 1 <= workers <= MAX_CONCURRENCY:
        raise ValueError(f"workers must be between 1 and {MAX_CONCURRENCY}")
    if type(per_model_limit) is not int or not 1 <= per_model_limit <= MAX_CONCURRENCY:
        raise ValueError(f"per_model_limit must be between 1 and {MAX_CONCURRENCY}")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    if type(timeout) is not int or not 1 <= timeout <= 420:
        raise ValueError("timeout must be between 1 and 420 seconds")
    if type(max_new_calls) is not int or max_new_calls < 0:
        raise ValueError("max_new_calls must be a nonnegative integer")
    study = Path(study)
    pending, reused, providers, catalog_sha = _prepare(study, jobs, timeout)
    if len(pending) > max_new_calls:
        raise ValueError(f"{len(pending)} new calls exceeds configured limit {max_new_calls}")
    implicit_caps = sorted({job.variant["variant"] for job in pending
                            if job.variant["model"].startswith("openrouter/") and
                            "max_tokens" not in job.variant})
    if implicit_caps and not check_only:
        raise ValueError("New OpenRouter calls require explicit max_tokens metadata "
                         "(a positive integer, or null to omit the cap): " + ", ".join(implicit_caps))
    if implicit_caps:
        print(json.dumps({"event": "legacy_cap_reconstruction", "variants": implicit_caps,
                          "max_tokens": LEGACY_MAX_OUTPUT_TOKENS, "execution_allowed": False}), flush=True)
    counts = {"completed": 0, "reused": reused, "failed": 0, "pending": len(pending),
              "estimated_usd": 0.0, "api_attempts": 0, "compatibility_fallbacks": 0}
    if check_only or not pending:
        print(json.dumps({"event": "checked" if check_only else "finished", **counts}), flush=True)
        return counts
    if any(job.variant["model"].startswith("openrouter/") for job in pending) and not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is required for new OpenRouter calls")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    source_hash = digest({path.name: sha256(path) for path in [Path(__file__),
                         *(Path(__file__).with_name(name) for name in
                           ("judge.py", "comparison.py", "judge_config.py", "structured_judge.py", "pretest_context.py", "extension_parser.py"))]})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
    lock = threading.Lock()
    started = time.monotonic()

    def perform(p: PreparedJob) -> None:
        path = call_path(study, p.job)
        model = p.variant["model"]
        provider = providers.get(model.removeprefix("openrouter/")) if model.startswith("openrouter/") else None
        transitions = []
        effective_format = "json_schema"
        total_started = time.monotonic()
        for attempt in range(1, max_attempts + 1):
            contract = request_contract(p.call_task, p.variant, response_format=effective_format, timeout=timeout)
            request_id = _request_id(contract, p.judge, p.job["draw"])
            metadata, raw = {}, None
            with lock:
                counts["api_attempts"] += 1
            try:
                if provider:
                    raw, metadata = openrouter_complete(contract, provider)
                else:
                    raw, metadata = LiteLLMJudge(model, p.variant.get("reasoning_effort"), timeout=timeout).complete(p.call_task)
                if provider:
                    ratings, parser_metadata = parse_provider_response(raw, p.call_task)
                else:
                    ratings, parser_metadata = parse_judge_response(raw, p.call_task), {}
                if provider:
                    cost, cost_provenance = openrouter_cost(metadata.get("usage", {}), provider, catalog_sha)
                else:
                    cost = estimated_cost(model, metadata.get("usage", {}))
                    cost_provenance = {"source": "native.fixed-token-prices", "is_estimate": True}
                record = {**p.job, "subject": p.task.subject, "plan_ids": p.task.plan_ids,
                          "presented_plan_ids": p.call_task.plan_ids, "order": "BA" if p.swapped else "AB",
                          "split": "validation", "human_reviewed": p.entry["human_reviewed"],
                          "model_a": p.entry["model_a"], "model_b": p.entry["model_b"],
                          "grading_id": p.entry["grading_id"], "criteria": p.task.metadata()["criteria"],
                          "judge": p.judge, "request_id": request_id, "requested_request_id": p.requested_id,
                          "messages": contract["body"]["messages"], "requested_messages": p.call_task.messages(),
                          "response_schema": p.call_task.schema(), "raw_response": raw, "ratings": ratings,
                          "normalized_ratings": _normalise(ratings, p.swapped), "dataset_sha256": p.task.dataset_sha256,
                          "runner_sha256": source_hash, "created_at": datetime.now(timezone.utc).isoformat(),
                          "duration_seconds": time.monotonic() - total_started, "attempts": attempt,
                          "estimated_usd": cost, "cost_provenance": cost_provenance, "run_id": run_id,
                          "requested_response_format": "json_schema", "effective_response_format": effective_format,
                          "transport": {"requested": p.contract, "effective": contract,
                                        "compatibility_fallbacks": transitions}, **metadata}
                if parser_metadata:
                    record.update(parser_metadata)
                    record["evaluation_id"] = digest({"request_id": request_id,
                                                       "parser_contract_sha256": parser_metadata["parser_contract_sha256"]})
                if p.context_provenance is not None:
                    record["pretest_context_provenance"] = p.context_provenance
                write_json(path, record)
                with lock:
                    counts["completed"] += 1
                    counts["estimated_usd"] += cost
                    if counts["completed"] % 10 == 0:
                        print(json.dumps({"event": "progress", **counts,
                                          "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
                return
            except Exception as exc:
                error = {**p.job, "run_id": run_id, "attempt": attempt, "type": type(exc).__name__,
                         "status_code": getattr(exc, "status_code", None), "message": _redact(str(exc)),
                         "request_id": request_id, "transport": contract,
                         "created_at": datetime.now(timezone.utc).isoformat(),
                         "raw_response": raw, "response_metadata": metadata}
                if isinstance(exc, ResponseError):
                    error["raw_response_payload"] = exc.payload
                # SDK status exceptions occasionally expose usage in an error
                # body; copy only billing fields, never headers or credentials.
                body = getattr(exc, "body", None)
                if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                    error["response_metadata"].setdefault("usage", body["usage"])
                error.update(_failed_attempt_billing(error, model, provider, catalog_sha))
                write_json(study / "errors" / f"{path.stem}--{run_id}--{attempt}.json", error)
                fallback = (provider is not None and attempt < max_attempts and
                            effective_format != FORMATS[-1] and _format_incompatibility(exc))
                if fallback:
                    next_format = FORMATS[FORMATS.index(effective_format) + 1]
                    transitions.append({"from": effective_format, "to": next_format,
                                        "reason": error["message"], "status_code": error["status_code"],
                                        "attempt": attempt, "failed_request_id": request_id})
                    effective_format = next_format
                    with lock:
                        counts["compatibility_fallbacks"] += 1
                    continue
                if attempt < max_attempts and _transient(exc):
                    time.sleep(attempt * 3)
                    continue
                with lock:
                    counts["failed"] += 1
                    print(json.dumps({"event": "failed", **p.job, "attempt": attempt,
                                      "type": error["type"], "status_code": error["status_code"],
                                      "message": error["message"]}), flush=True)
                return

    print(json.dumps({"event": "start", "jobs": len(jobs), "new_jobs": len(pending),
                      "workers": workers, "per_model_limit": min(workers, per_model_limit),
                      "requested_per_model_limit": per_model_limit, "run_id": run_id}), flush=True)
    _dispatch_jobs(pending, perform, workers, per_model_limit=per_model_limit)
    print(json.dumps({"event": "finished", **counts, "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("jobs", type=Path)
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                     help=f"Total concurrent calls, 1–{MAX_CONCURRENCY} (default: {DEFAULT_WORKERS})")
    run.add_argument("--per-model-limit", type=int, default=DEFAULT_PER_MODEL_LIMIT,
                     help=f"Concurrent calls per exact model, 1–{MAX_CONCURRENCY} "
                          f"(default: {DEFAULT_PER_MODEL_LIMIT}; capped by --workers)")
    run.add_argument("--max-new-calls", type=int, default=1200)
    run.add_argument("--timeout", type=int, default=420)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--check-only", action="store_true", help="Validate all requests and caches without API calls")
    args = parser.parse_args()
    counts = run_jobs(args.study, json.loads(args.jobs.read_text()), args.workers, args.max_new_calls,
                      args.timeout, args.max_attempts, check_only=args.check_only,
                      per_model_limit=args.per_model_limit)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
