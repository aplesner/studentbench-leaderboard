"""Versioned lossless serialization compatibility for extension provider outputs.

Verdicts, evidence, criterion identities and semantic validation are unchanged.
This module is never used for native or imported-control responses.
"""

from __future__ import annotations

import hashlib
import copy
import json
import re

from .data import digest
from .judge import Task
from .structured_judge import parse_judge_response

PARSER_V1 = "extension-provider-serialization.v1"
PARSER_V2 = "extension-provider-serialization.v2"
PARSER_CONTRACT_V1 = {
    "version": "extension-provider-serialization.v1",
    "scope": "new-openrouter-responses-only",
    "outer_fence": "optional-one-complete-json-or-unlabelled-fence-no-surrounding-prose",
    "root": "exact-requested-keyed-object-or-exact-criteria-array-wrapper",
    "array_identifiers": "exactly-one-id-or-criterion_id-field-uniform-across-array",
    "criteria": "every-requested-id-exactly-once-no-extra-ids-or-fields",
    "entry_fields": ["evidence_a", "evidence_b", "reason", "winner", "strength"],
    "semantic_parser": "unchanged-structured_judge.parse_judge_response",
    "content_changes": "none",
}
PARSER_CONTRACT_V2 = {**PARSER_CONTRACT_V1, "version": PARSER_V2,
                      "root": "exact-requested-keyed-object-bare-array-or-exact-single-named-array-wrapper"}
PARSER_CONTRACTS = {PARSER_V1: PARSER_CONTRACT_V1, PARSER_V2: PARSER_CONTRACT_V2}
PARSER_CONTRACT = PARSER_CONTRACT_V2
PARSER_CONTRACT_SHA256 = digest(PARSER_CONTRACT)


def parse_provider_response(raw: str, task: Task, *, version: str = PARSER_V2) -> tuple[dict, dict]:
    """Return strict ratings and explicit serialization/provenance metadata."""
    if version not in PARSER_CONTRACTS:
        raise ValueError(f"Unknown provider parser version: {version}")
    contract = PARSER_CONTRACTS[version]
    if not isinstance(raw, str):
        raise ValueError("Provider response must be text")
    transformations = []
    text = raw.strip()
    if text.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*", text)
        if fence is None:
            raise ValueError("Only one complete outer JSON fence with no surrounding prose is accepted")
        text = fence.group(1)
        transformations.append("strip_outer_json_fence")

    def unique(items):
        value = {}
        for key, entry in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = entry
        return value

    result = json.loads(text, object_pairs_hook=unique)
    expected = {criterion["id"] for criterion in task.rubric}
    rows, wrapper = None, None
    if version == PARSER_V1 and isinstance(result, dict) and set(result) == {"criteria"}:
        rows, wrapper = result["criteria"], "criteria"
        if not isinstance(rows, list):
            raise ValueError("Criteria array must contain every requested criterion exactly once")
    elif version == PARSER_V2:
        if isinstance(result, list):
            rows = result
        elif isinstance(result, dict) and len(result) == 1 and isinstance(next(iter(result.values())), list):
            wrapper, rows = next(iter(result.items()))
    if rows is not None:
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError("Criteria array must contain every requested criterion exactly once")
        mapped, identifier_style = {}, None
        fields = set(contract["entry_fields"])
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each criteria array entry must be an object")
            identifiers = set(row) & {"id", "criterion_id"}
            if len(identifiers) != 1:
                raise ValueError("Each criteria array entry must use exactly one ID field")
            identifier = next(iter(identifiers))
            if identifier_style is not None and identifier != identifier_style:
                raise ValueError("Criteria array entries must use the same ID field style")
            identifier_style = identifier
            if set(row) != fields | {identifier}:
                raise ValueError("Criteria array entries must contain exactly the requested evidence/winner fields")
            key = row[identifier]
            if not isinstance(key, str) or key not in expected or key in mapped:
                raise ValueError("Criteria array IDs must be exact, requested and unique")
            mapped[key] = {field: value for field, value in row.items() if field != identifier}
        if set(mapped) != expected:
            raise ValueError("Criteria array is missing a requested ID")
        result = mapped
        if version == PARSER_V1:
            transformations.append(f"criteria_array_to_keyed_object:{identifier_style}")
        elif wrapper is None:
            transformations.append(f"bare_array_to_keyed_object:{identifier_style}")
        else:
            detail = json.dumps({"wrapper": wrapper, "id_field": identifier_style}, ensure_ascii=False,
                                sort_keys=True, separators=(",", ":"))
            transformations.append(f"named_array_to_keyed_object:{detail}")
    canonical = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    ratings = parse_judge_response(canonical, task)
    metadata = {"parser_contract": copy.deepcopy(contract), "parser_version": contract["version"],
                "parser_contract_sha256": digest(contract),
                "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "canonical_response": canonical, "canonical_response_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "transformations": transformations, "provider_schema_compliant": not transformations}
    return ratings, metadata
