"""Optional, verified pretest context; never infer absent question content.

The SSOT Quant CSV omits some quantities and stimuli. An explicitly supplied,
hash-pinned application bank can supplement it only after every shared item
field matches the immutable export. This module is not used by make_task.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

from .judge import PRETEST_FIELDS, Task


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inline_quantities(stem: str) -> dict:
    match = re.fullmatch(r"Quantity A:\s*(.*?)\nQuantity B:\s*(.*)", stem.strip(), re.DOTALL)
    return {"quantity_a": match[1].strip(), "quantity_b": match[2].strip()} if match else {}


def _exported_stem(item: dict) -> str:
    # This is the exact SSOT renderer fallback for the three stemless QC items.
    if item.get("stem", "").strip():
        return item["stem"]
    if item.get("quantity_a") and item.get("quantity_b"):
        return f"Quantity A:  {item['quantity_a']}\nQuantity B:  {item['quantity_b']}"
    return ""


def _compare_shared_fields(exported: dict, source: dict) -> None:
    fields = (("correct_answer", "correct_answer"), ("format_code", "format"),
              ("skill", "skill"), ("subskill", "subskill"), ("difficulty", "difficulty"))
    if exported["question"] != _exported_stem(source):
        raise ValueError(f"Source question stem differs from SSOT: {exported['internal_id']}")
    for exported_key, source_key in fields:
        if exported[exported_key] != source.get(source_key):
            raise ValueError(f"Source {source_key} differs from SSOT: {exported['internal_id']}")
    if json.loads(exported["options_json"]) != source.get("options", {}):
        raise ValueError(f"Source options differ from SSOT: {exported['internal_id']}")


@dataclass(frozen=True)
class PretestContextBank:
    items: dict[str, dict]
    aliases: dict[str, str]
    provenance: dict

    @classmethod
    def load(cls, ssot_root: Path, *, source_bank: Path | None = None,
             expected_source_sha256: str | None = None) -> PretestContextBank:
        """Load Quant item context without any human-review or author data.

        External context is opt-in and requires a pinned file hash. The hash
        identifies the actual supplement; matching shared fields does not prove
        that unexported assets were identical at every historical deployment.
        """
        bank_path = ssot_root / "question_bank_quant.csv"
        exported_hash = _sha256(bank_path)
        manifest = json.loads((ssot_root / "release_manifest.json").read_text())
        manifest_hash = next((row["sha256"] for row in manifest["files"]
                              if row["path"] == "question_bank_quant.csv"), None)
        if manifest_hash != exported_hash:
            raise ValueError("SSOT question-bank fingerprint differs from release manifest")
        with bank_path.open(newline="", encoding="utf-8") as stream:
            exported = list(csv.DictReader(stream))
        source_items = {}
        if source_bank is not None:
            if not expected_source_sha256 or _sha256(source_bank) != expected_source_sha256:
                raise ValueError("An external question bank requires its matching pinned SHA256")
            source = json.loads(source_bank.read_text())
            if source.get("domain") != "gre_quant":
                raise ValueError("Expected a GRE Quant application question bank")
            for phase in ("pretest", "posttest"):
                for item in source[phase]:
                    if item["id"] in source_items:
                        raise ValueError(f"Duplicate source question ID: {item['id']}")
                    source_items[item["id"]] = item
        elif expected_source_sha256 is not None:
            raise ValueError("A source hash needs a source question-bank file")
        if source_bank is not None and set(source_items) != {row["internal_id"] for row in exported}:
            raise ValueError("External and SSOT question banks must cover exactly the same item IDs")
        items, aliases = {}, {}
        for row in exported:
            internal = row["internal_id"]
            if internal in items:
                raise ValueError(f"Duplicate SSOT internal question ID: {internal}")
            context = _inline_quantities(row["question"])
            if source_bank is not None:
                if internal not in source_items:
                    raise ValueError(f"Question missing from external source bank: {internal}")
                original = source_items[internal]
                _compare_shared_fields(row, original)
                context.update({key: original[key] for key in ("quantity_a", "quantity_b")
                                if original.get(key)})
                if original.get("stimulus"):
                    context["shared_data"] = original["stimulus"]
                if original.get("asset"):
                    context["figure_status"] = "Original source figure is not rendered in this text payload."
                    context["figure_source_sha256"] = hashlib.sha256(original["asset"].encode()).hexdigest()
                context["source"] = "hash_pinned_application_bank_verified_against_ssot"
            else:
                context["source"] = "immutable_ssot_question_bank"
                context["export_limitation"] = "SSOT CSV does not preserve all source stimuli or figures."
            if row["format_code"] == "QC" and not (context.get("quantity_a") and context.get("quantity_b")):
                context["missing_evidence"] = ["Quantitative-comparison quantities are absent; do not reconstruct them."]
            if context.get("figure_status"):
                context.setdefault("missing_evidence", []).append(
                    "The source figure is unavailable here. Use only the supplied stem, quantities, and shared data; "
                    "do not infer geometry or values not stated.")
            items[internal] = {"exported": row, "context": context}
            for alias in (internal, row["question_id"]):
                if alias in aliases and aliases[alias] != internal:
                    raise ValueError(f"Ambiguous question ID: {alias}")
                aliases[alias] = internal
        return cls(items, aliases, {
            "instrument": "gre_quant", "ssot_bank_sha256": exported_hash,
            "source_bank_sha256": expected_source_sha256,
            "source_type": "application_supplement" if source_bank is not None else "ssot_only",
            "matched_items": len(items),
            "limitation": "Shared fields match exactly; unavailable historical stimulus fields cannot be "
                          "verified against the SSOT CSV. Source figures remain unrendered in text-only payloads.",
        })

    def enrich_pretest(self, pretest: list[dict]) -> list[dict]:
        result = []
        for row in pretest:
            if row["question_id"] not in self.aliases:
                raise ValueError(f"Unknown pretest question: {row['question_id']}")
            item = self.items[self.aliases[row["question_id"]]]
            exported = item["exported"]
            for field, expected in (("question_text", exported["question"]),
                                    ("correct_answer", exported["correct_answer"]),
                                    ("skill", exported["skill"]), ("subskill", exported["subskill"]),
                                    ("difficulty", exported["difficulty"])):
                if row.get(field) != expected:
                    raise ValueError(f"Pretest {field} does not match SSOT: {row['question_id']}")
            if (row.get("format") not in (exported["format"], exported["format_code"]) or
                    json.loads(row["options_json"]) != json.loads(exported["options_json"])):
                raise ValueError(f"Pretest format/options do not match SSOT: {row['question_id']}")
            # Only judge-visible input fields survive even if a caller supplies
            # a full CSV row with reviewer/subject/administrative metadata.
            kept = {key: copy.deepcopy(row[key]) for key in PRETEST_FIELDS if key in row}
            kept["question_context"] = copy.deepcopy(item["context"])
            result.append(kept)
        return result

    def enrich_task(self, task: Task) -> Task:
        if task.instrument != "gre_quant":
            raise ValueError("This verified question-context adapter currently supports GRE Quant only")
        return replace(task, pretest=self.enrich_pretest(task.pretest))


def enrich_pretest(pretest: list[dict], ssot_root: Path, source_bank: Path,
                   expected_source_sha256: str) -> tuple[list[dict], dict]:
    """Functional entry point for an explicitly opted-in study variant."""
    bank = PretestContextBank.load(ssot_root, source_bank=source_bank,
                                   expected_source_sha256=expected_source_sha256)
    return bank.enrich_pretest(pretest), copy.deepcopy(bank.provenance)
