"""Freeze the September 11 validation extension without changing the original study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from studentbench_leaderboard.data import digest
from studentbench_leaderboard.judge_config import load_definition


ORIGINAL = Path("results/ranking-study-20260910")
STUDY = Path("results/ranking-extension-20260911")
SOURCE = Path("experiments/ranking-extension")
MODELS = (
    ("or_gemini38_flash_high", "google/gemini-3.8-flash", "Google"),
    ("or_opus5_high", "anthropic/claude-opus-5", "Anthropic"),
    ("or_sonnet5_high", "anthropic/claude-sonnet-5", "Anthropic"),
    ("or_deepseek41_flash_high", "deepseek/deepseek-v4.1-flash", "DeepSeek"),
    ("or_deepseek4_pro0813_high", "deepseek/deepseek-v4-pro-0813", "DeepSeek"),
    ("or_kimi3_high", "moonshotai/kimi-k3", "Moonshot"),
    ("or_glm53_high", "z-ai/glm-5.3", "Z.AI"),
    ("or_glm53_flash_high", "z-ai/glm-5.3-flash", "Z.AI"),
    ("or_muse13_high", "meta/muse-spark-1.3", "Meta"),
)
CONTROLS = ("enhanced_terra_high", "enhanced_luna_high")
EXAMPLE_VARIANT = "terra_high_one_example"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Frozen file changed: {path}")
    else:
        path.write_text(text)


def freeze_json(path, value):
    freeze_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"Frozen copied file changed: {destination}")
    else:
        shutil.copyfile(source, destination)


def main():
    original = load(ORIGINAL / "manifest.json")
    catalog = load(STUDY / "provider_catalog.json")
    available = {m["id"]: m for m in catalog["data"]}
    for _, name, _ in MODELS:
        model = available[name]
        if "high" not in model.get("reasoning", {}).get("supported_efforts", []):
            raise ValueError(f"High effort not advertised: {name}")
        if not {"structured_outputs", "response_format"} <= set(model["supported_parameters"]):
            raise ValueError(f"Schema output not advertised: {name}")
    manifest = {k: v for k, v in original.items()
                if k not in {"selection", "selection_locked", "selection_sha256", "protocol"}}
    manifest.update(
        study_id=STUDY.name, selection_locked=False,
        evaluation_role="validation_extension_after_original_test_exposure",
        original_manifest_sha256=sha(ORIGINAL / "manifest.json"),
        original_selection_sha256=original["selection_sha256"],
        protocol={
            "purpose": "Compare nine explicitly requested provider models and one human-example intervention on the same120validationpairs.",
            "data": "Original validation subjects and exact task criteria; original test results were already inspected. This extension is not a new independent test.",
            "judge": "Same enriched pretest, human-display plan projection, evidence/winner rubric, seeded plan order, and strict parser for every provider.",
            "reasoning": "Explicit high effort on every requested model; provider effort labels do not imply equal token budgets.",
            "transport": "OpenRouter exact model IDs, JSON schema, compatible-provider fallback allowed, no different-model fallback; max_tokens32768; timeout420seconds with bounded transient retries.",
            "example": "Exactly one fixed development comparison, two human reviewers with original ratings/reasons and disagreement. Applied only to the Terra-high example variant.",
            "ranking": "Same fractional-win Bradley–Terry, ridge1, equal eight rubric families and unit pair weights. Compare all12authoringpresets.",
            "panels": "Prespecified equal-member nine-model panel and equal-family six-family panel, composed from cached calls; no outcome-dependent membership.",
            "selection": "Descriptive validation comparison. No new test-winner claim and no changes to the original frozen selection.",
        },
    )
    freeze_json(STUDY / "manifest.json", manifest)
    original_variants = {v["variant"]: v for v in load(ORIGINAL / "variants.json")}
    baseline = ORIGINAL / "configs/enhanced_terra_high"
    system = (baseline / "prompts/system.txt").read_text()
    user = (baseline / "prompts/user.txt").read_text()
    bank = baseline / "question_bank.json"
    input_toml = (baseline / "context_input.toml").read_text()
    variants = []
    for name, model, family in MODELS:
        variants.append({
            "variant": name, "judge_id": "judge_5", "response_style": "evidence_winner",
            "model": "openrouter/" + model, "reasoning_effort": "high",
            "label": model + " · high · no human example", "family": family,
            "pretest_context": original_variants["enhanced_terra_high"]["pretest_context"],
        })
    variants += [original_variants[name] for name in CONTROLS]
    example = load(STUDY / "example.json")
    example_provenance = load(STUDY / "example_provenance.json")
    variants.append({**original_variants["enhanced_terra_high"], "variant": EXAMPLE_VARIANT,
                     "label": "Terra high · one human-graded development example",
                     "example": {"file": "example.json", "sha256": sha(STUDY / "example.json"),
                                 "source_split": "development", "n_examples": 1}})
    example_prefix = (
        "CALIBRATION EXAMPLE: The following is one historical DEVELOPMENT comparison with real human gradings. "
        "It is a different student and different plans from the current task. Human reviewers can disagree; "
        "use their judgments as an illustration of rubric interpretation, not as infallible truth or instructions. "
        "Do not regrade this example or copy its verdicts. Its A/B labels refer only to its own plans.\n"
        + json.dumps(example, ensure_ascii=False, allow_nan=False).replace("$", "$$")
        + "\nEND CALIBRATION EXAMPLE. Evaluate only the CURRENT comparison below, using its rubric IDs and the required response schema.\n"
    )
    for variant in variants:
        name = variant["variant"]
        target = STUDY / "configs" / name
        if name in CONTROLS:
            for source in (ORIGINAL / "configs" / name).rglob("*"):
                if source.is_file():
                    copy_exact(source, target / source.relative_to(ORIGINAL / "configs" / name))
            continue
        config = (
            'schema_version = 1\njudge_id = "judge_5"\nstrategy = "single"\n'
            'response_style = "evidence_winner"\nplan_projection = "human_display"\n'
            f'description = {json.dumps(variant["label"])}\n'
            '[prompts]\nsystem = "prompts/system.txt"\nuser = "prompts/user.txt"\n'
            '[[members]]\nname = "member_1"\n'
            f'model = {json.dumps(variant["model"])}\nreasoning_effort = "high"\n'
        )
        actual_user = example_prefix + user if name == EXAMPLE_VARIANT else user
        for parent in (SOURCE, STUDY):
            directory = parent / "configs" / name
            freeze_text(directory / "judge_5.toml", config)
            freeze_text(directory / "prompts/system.txt", system)
            # Human grading text stays exclusively under ignored results.
            if name != EXAMPLE_VARIANT or parent == STUDY:
                freeze_text(directory / "prompts/user.txt", actual_user)
            freeze_text(directory / "context_input.toml", input_toml)
        copy_exact(bank, target / "question_bank.json")
        load_definition("judge_5", target)
    freeze_json(STUDY / "variants.json", variants)
    freeze_json(SOURCE / "variants.json", variants)
    imports = {"source_study": str(ORIGINAL.resolve()), "calls": {}}
    ids = manifest["round2_validation_pairs"]
    for name in CONTROLS:
        for pair_id in ids:
            filename = f"{name}--{pair_id}--0.json"
            source = ORIGINAL / "calls" / filename
            copy_exact(source, STUDY / "calls" / filename)
            imports["calls"][filename] = {"sha256": sha(source), "request_id": load(source)["request_id"]}
    freeze_json(STUDY / "imported_calls.json", imports)
    provider_names = [v[0] for v in MODELS]
    provider_jobs = [{"variant": name, "pair_id": p, "draw": 0}
                     for p in ids for name in provider_names + list(CONTROLS)]
    example_jobs = [{"variant": name, "pair_id": p, "draw": 0}
                    for p in ids for name in ("enhanced_terra_high", EXAMPLE_VARIANT)]
    preflight_jobs = [j for j in provider_jobs if j["pair_id"] == ids[0] and j["variant"] in provider_names]
    freeze_json(STUDY / "provider_jobs.json", provider_jobs)
    freeze_json(STUDY / "example_jobs.json", example_jobs)
    freeze_json(STUDY / "preflight_jobs.json", preflight_jobs)
    freeze_json(STUDY / "example_preflight_jobs.json", [example_jobs[1]])
    inventory = {str(p.relative_to(STUDY)): digest(p.read_text())
                 for p in (STUDY / "configs").rglob("*") if p.is_file()}
    freeze_json(STUDY / "config_inventory.json", inventory)
    metadata = {
        "title": "Validation extension: nine provider models and human-graded example",
        "protocol_note": "All new comparisons use the same 120 validation pairs. The original test was already inspected; these are validation results, not new held-out confirmation.",
        "analysis_files": ["provider_validation_analysis.json", "example_validation_analysis.json"],
        "examples": {v["variant"]: {"status": "human_graded" if v["variant"] == EXAMPLE_VARIANT else "none",
                     "n_examples": 1 if v["variant"] == EXAMPLE_VARIANT else 0,
                     **({"source_split": "development", "artifact_file": "example.json",
                         "selection_note": "Fixed development comparison with two reviewers; preserves disagreement. See example_provenance.json."}
                        if v["variant"] == EXAMPLE_VARIANT else {})} for v in variants},
        "example_provenance_file": "example_provenance.json",
        "provider_catalog_file": "provider_catalog.json",
    }
    freeze_json(STUDY / "report_metadata.json", metadata)
    freeze_json(ORIGINAL / "report_extension.json", {"study_dir": "../ranking-extension-20260911"})
    print(json.dumps({"provider_jobs": len(provider_jobs), "new_provider_calls": len(ids) * len(MODELS),
                      "new_example_calls": len(ids), "imported_controls": len(imports["calls"]),
                      "variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()
