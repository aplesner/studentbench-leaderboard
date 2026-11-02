"""Prepare Qwen and small, distinct-family panels without changing earlier runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from studentbench_leaderboard.data import digest
from studentbench_leaderboard.judge_config import load_definition
from studentbench_leaderboard.study_analysis import load_derived_judges

ORIGINAL = Path("results/ranking-study-20260910")
PREVIOUS = Path("results/ranking-extension-20260911")
STUDY = Path("results/ranking-followup-20260911")
SOURCE = Path("experiments/ranking-followup")
QWEN = "qwen38_max_high"
REUSE = ("enhanced_terra_high", "or_deepseek41_flash_high", "or_glm53_flash_high", "or_kimi3_high")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content.encode() if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Prepared input changed: {path}")
    else:
        path.write_bytes(raw)


def freeze_json(path, value):
    freeze(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def copy_tree(source, target):
    for path in source.rglob("*"):
        if path.is_file():
            freeze(target / path.relative_to(source), path.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-catalog", type=Path, required=True, metavar="PATH",
                        help="Path to the local provider catalog JSON used to prepare this follow-up")
    args = parser.parse_args(argv)
    catalog_source = args.provider_catalog
    if not catalog_source.is_file():
        parser.error(f"Provider catalog is not a readable local file: {catalog_source}")
    try:
        provider_catalog = load(catalog_source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"Cannot read provider catalog JSON from {catalog_source}: {exc}")
    qwen = next(row for row in provider_catalog["data"] if row["id"] == "qwen/qwen3.8-max-0902")
    assert "high" in qwen["reasoning"]["supported_efforts"]
    assert {"structured_outputs", "response_format"} <= set(qwen["supported_parameters"])
    freeze(STUDY / "provider_catalog.json", catalog_source.read_bytes())
    old = load(PREVIOUS / "manifest.json")
    manifest = {**old, "study_id": STUDY.name,
                "previous_manifest_sha256": sha(PREVIOUS / "manifest.json"),
                "protocol": {
                    "purpose": "Evaluate requested Qwen3.8 Max and three small panels on the same 120 validation plan pairs.",
                    "selection": "Panel members were chosen after observing earlier validation results. This is exploratory development on reused validation data, not an independent test or a replacement for the selected Terra system.",
                    "panel_selection": "Terra is the existing control; DeepSeek Flash had the highest previous point alignment; GLM Flash matched Terra at low recorded cost; Kimi supplies a different-family alternative. Qwen enters one prespecified quartet before its judgments are inspected.",
                    "panel_aggregation": "Each member has one equal vote per rubric question: A=1, tie=0.5, B=0. Average those votes. Every member must answer; missing answers are not replaced. Each panel includes at most one model per provider family.",
                    "qwen_request": "Exact OpenRouter qwen/qwen3.8-max-0902, high reasoning, original enriched context and response rubric, same seeded plan order; max_tokens explicitly null in variant metadata, so no application output cap is sent. Provider limits still apply.",
                    "execution": "One preflight followed by remaining requests with 128 total/per-model slots. No more than 120 requests exist in this Qwen schedule. Runner supports configuring up to 512 slots.",
                    "reuse": "480 earlier calls are copied byte-for-byte with source paths and hashes. They retain their original prompts, caps, catalog and billing provenance. Reused provider calls are verified in the original study, never reconstructed against the new catalog.",
                    "ranking": "Same 12 lesson plan generator systems, 120-pair human reference, eight rubric families, Bradley–Terry ridge1, equal plan-pair weights, student bootstrap500.",
                }}
    freeze_json(STUDY / "manifest.json", manifest)
    previous_variants = {v["variant"]: v for v in load(PREVIOUS / "variants.json")}
    variants = [previous_variants[name] for name in REUSE]
    variants[0] = {**variants[0], "family": "OpenAI"}
    baseline = PREVIOUS / "configs/enhanced_terra_high"
    qwen_variant = {**previous_variants["enhanced_terra_high"], "variant": QWEN,
                    "model": "openrouter/qwen/qwen3.8-max-0902", "reasoning_effort": "high",
                    "family": "Alibaba", "label": "Qwen3.8 Max · high reasoning",
                    "max_tokens": None}
    variants.append(qwen_variant)
    for name in REUSE:
        copy_tree(PREVIOUS / "configs" / name, STUDY / "configs" / name)
    config = ('schema_version = 1\njudge_id = "judge_5"\nstrategy = "single"\n'
              'response_style = "evidence_winner"\nplan_projection = "human_display"\n'
              'description = "Qwen3.8 Max with high reasoning and provider-default output limit"\n'
              '[prompts]\nsystem = "prompts/system.txt"\nuser = "prompts/user.txt"\n'
              '[[members]]\nname = "member_1"\nmodel = "openrouter/qwen/qwen3.8-max-0902"\n'
              'reasoning_effort = "high"\n')
    for parent in (STUDY, SOURCE):
        target = parent / "configs" / QWEN
        freeze(target / "judge_5.toml", config)
        for name in ("prompts/system.txt", "prompts/user.txt", "context_input.toml"):
            freeze(target / name, (baseline / name).read_bytes())
    freeze(STUDY / "configs" / QWEN / "question_bank.json", (baseline / "question_bank.json").read_bytes())
    for v in variants:
        load_definition(v["judge_id"], STUDY / "configs" / v["variant"])
    freeze_json(STUDY / "variants.json", variants)
    freeze_json(SOURCE / "variants.json", variants)
    ids = manifest["round2_validation_pairs"]
    reuse = {}
    native_imports = {}
    for name in REUSE:
        for pair_id in ids:
            filename = f"{name}--{pair_id}--0.json"
            source = PREVIOUS / "calls" / filename
            record = load(source)
            freeze(STUDY / "calls" / filename, source.read_bytes())
            reuse[filename] = {"source": str(source.resolve()), "sha256": sha(source),
                               "request_id": record["request_id"], "variant": name,
                               "pair_id": pair_id, "draw": 0}
            if "transport" not in record:
                native_imports[filename] = {"sha256": sha(source), "request_id": record["request_id"]}
    freeze_json(STUDY / "reused_calls.json", {"source_study": str(PREVIOUS.resolve()),
                "source_catalog_sha256": sha(PREVIOUS / "provider_catalog.json"), "calls": reuse})
    freeze_json(STUDY / "imported_calls.json", {"source_study": str(PREVIOUS.resolve()), "calls": native_imports})
    jobs = [{"variant": name, "pair_id": pair_id, "draw": 0} for pair_id in ids for name in (*REUSE, QWEN)]
    qwen_jobs = [j for j in jobs if j["variant"] == QWEN]
    freeze_json(STUDY / "focused_jobs.json", jobs)
    freeze_json(STUDY / "qwen_jobs.json", qwen_jobs)
    freeze_json(STUDY / "preflight_jobs.json", qwen_jobs[:1])
    recipes = [
        ("judge_14", "panel_trio", "Terra + DeepSeek Flash + GLM Flash", REUSE[:3]),
        ("judge_15", "panel_trio_kimi", "Terra + DeepSeek Flash + Kimi", (REUSE[0], REUSE[1], REUSE[3])),
        ("judge_16", "panel_quartet_qwen", "Terra + DeepSeek Flash + GLM Flash + Qwen", (*REUSE[:3], QWEN)),
    ]
    text = "schema_version = 1\n"
    for judge_id, name, label, members in recipes:
        text += (f'\n[[judges]]\njudge_id = "{judge_id}"\nvariant = "{name}"\n'
                 f'label = "{label}"\naggregation = "soft"\n'
                 'description = "Small panel selected using earlier validation evidence; one model per family, equal votes. Exploratory comparison on reused validation data."\n')
        for member in members:
            text += f'\n[[judges.members]]\nvariant = "{member}"\ndraw = 0\n'
    for parent in (SOURCE, STUDY):
        freeze(parent / "derived_judges.toml", text)
    freeze_json(STUDY / "derived_judges_manifest.json", {"file_sha256": sha(STUDY / "derived_judges.toml"),
                "selection_basis": manifest["protocol"]["panel_selection"],
                "selected_after_previous_validation": True, "selected_before_qwen_outcomes": True})
    definitions = load_derived_judges(STUDY, {v["variant"]: v for v in variants})
    for definition in definitions:
        families = [m["family"] for m in definition["weighted_members"]]
        assert len(set(families)) == len(families)
    freeze_json(STUDY / "report_metadata.json", {
        "title": "Qwen and small panels with one model per family",
        "protocol_note": "This follow-up uses the same 120 validation pairs. Panel choices use earlier validation results; Qwen was added afterward. These are exploratory results on reused data, not a new held-out test.",
        "analysis_files": ["focused_validation_analysis.json"],
        "examples": {v["variant"]: {"status": "none", "n_examples": 0} for v in variants},
        "provider_catalog_file": "provider_catalog.json",
    })
    freeze_json(STUDY / "config_inventory.json", {str(p.relative_to(STUDY)): digest(p.read_text())
                for p in (STUDY / "configs").rglob("*") if p.is_file()})
    freeze_json(STUDY / "prepared_inputs.json", {str(p.relative_to(STUDY)): sha(p)
                for p in STUDY.rglob("*") if p.is_file() and "calls" not in p.parts
                and p.name not in {"prepared_inputs.json", "summary.json"}})
    pointer_path = ORIGINAL / "report_extension.json"
    pointer = load(pointer_path)
    pointer["followup_study_dir"] = "../" + STUDY.name
    pointer_path.write_text(json.dumps(pointer, indent=2) + "\n")
    print(json.dumps({"scheduled_calls": len(jobs), "new_qwen_calls": len(qwen_jobs),
                      "reused_calls": len(reuse), "panels": len(definitions)}))


if __name__ == "__main__":
    main()
