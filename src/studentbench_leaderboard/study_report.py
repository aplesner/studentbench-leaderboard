"""Offline, self-contained HTML report for the model-ranking study.

The renderer reads analysis artifacts and frozen request/configuration snapshots.
It never opens raw SSOT tables, makes API requests, or advances selection/test
state. Explicitly declared development examples can be displayed from artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import tomllib

from .data import digest


def _load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _compact(value):
    """Keep summaries and provenance while omitting large repeated draw arrays."""
    omitted = {"runs", "draws", "partitions", "subject_ids", "left_reviewer_ids", "right_reviewer_ids",
               "messages", "raw_response", "response_schema", "human_rows", "judge_rows"}
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _local_text(directory: Path, filename: str | None) -> str | None:
    if not filename:
        return None
    path = (directory / filename).resolve()
    return path.read_text() if path.is_relative_to(directory.resolve()) and path.is_file() else None


def _configurations(study_dir: Path, variants: list[dict]) -> dict:
    result = {}
    for variant in variants:
        path = (study_dir / "configs" / variant["variant"] / f"{variant['judge_id']}.toml").resolve()
        if not path.is_relative_to(study_dir / "configs") or not path.is_file():
            continue
        text = path.read_text()
        config = tomllib.loads(text)
        config.update(config_text=text,
                      system_prompt=_local_text(path.parent, config.get("prompts", {}).get("system")),
                      user_template=_local_text(path.parent, config.get("prompts", {}).get("user")))
        result[variant["variant"]] = config
    return result


def _judge_catalog(study_dir: Path, variants: list[dict], rounds: list[dict], configurations: dict,
                   metadata: dict | None = None, prefix: str = "") -> list[dict]:
    """Expose frozen recipes and saved schemas without loading current SSOT labels."""
    metadata = metadata or {}
    examples = metadata.get("examples", {})
    catalog = {}
    for variant in variants:
        name = variant["variant"]
        config = configurations.get(name, {})
        catalog[name] = {
            "key": prefix + name, "variant": name, "label": variant.get("label", name),
            "group": ("Focused follow-up" if prefix == "followup:" else "Validation extension") if prefix else "Original study",
            "kind": "single", "judge_id": variant.get("judge_id"),
            "variant_configuration": variant,
            "response_style": variant.get("response_style", config.get("response_style", "score_reason")),
            "plan_projection": config.get("plan_projection", "as_stored"),
            "pretest_context": variant.get("pretest_context"),
            "configuration": config, "system_prompt": config.get("system_prompt"),
            "user_template": config.get("user_template"), "members": config.get("members", []),
            "member_calls": [{**member, "draw": 0} for member in config.get("members", [])],
            "examples": examples.get(name, {"status": "none", "n_examples": 0}),
            "availability": metadata.get("availability", {}).get(name),
        }
    # A saved schema is an actual request schema; its criterion IDs can differ across pairs.
    remaining = {name for name, entry in catalog.items()
                 if (entry.get("availability") or {}).get("status") != "unavailable"}
    for path in sorted((study_dir / "calls").glob("*.json")):
        if not remaining:
            break
        call = _load(path, {})
        name = call.get("variant")
        if name not in remaining:
            continue
        entry = catalog[name]
        saved = call.get("judge", {})
        entry.update(output_schema=call.get("response_schema"), version=saved.get("version"),
                     saved_structure=saved.get("structure"), schema_source=f"calls/{path.name}",
                     returned_model=call.get("response_model"),
                     prompt_snapshot_matches=(entry.get("system_prompt") == saved.get("system_prompt")
                                              and entry.get("user_template") == saved.get("user_template")))
        entry.update(parser_contract=call.get("parser_contract"), parser_version=call.get("parser_version"),
                     parser_contract_sha256=call.get("parser_contract_sha256"), transformations=call.get("transformations"),
                     provider_schema_compliant=call.get("provider_schema_compliant"),
                     serialization_summary=metadata.get("serialization_by_variant", {}).get(name),
                     cost_usage={key: value for key, value in (call.get("usage") or {}).items()
                                 if key in {"cost", "cost_details", "is_byok"}})
        entry["system_prompt"] = saved.get("system_prompt", entry.get("system_prompt"))
        entry["user_template"] = saved.get("user_template", entry.get("user_template"))
        if call.get("transport"):
            transport = {}
            for phase in ("requested", "effective"):
                original = call["transport"].get(phase, {})
                transport[phase] = {key: value for key, value in original.items() if key in {
                    "version", "adapter", "endpoint", "timeout_seconds", "sdk_retries", "response_format",
                    "semantic_schema_sha256"}}
                transport[phase]["body_settings"] = {key: value for key, value in original.get("body", {}).items()
                    if key in {"model", "max_tokens", "provider", "reasoning", "reasoning_effort", "response_format",
                               "temperature", "top_p", "seed"}}
            transport["compatibility_fallbacks"] = call["transport"].get("compatibility_fallbacks", [])
            entry.update(transport=transport, response_provider=call.get("response_provider"),
                         requested_response_format=call.get("requested_response_format"),
                         effective_response_format=call.get("effective_response_format"),
                         cost_provenance=call.get("cost_provenance"))
        # If adaptation occurred, expose an adapted representative rather than an
        # earlier direct-shape preflight; aggregate counts still describe all calls.
        needs_adapted = (entry.get("serialization_summary") or {}).get("adapted_calls", 0) > 0
        if not needs_adapted or call.get("transformations"):
            remaining.remove(name)
    catalog_rounds = list(rounds)
    if (study_dir / "derived_judges.toml").is_file():
        document = tomllib.loads((study_dir / "derived_judges.toml").read_text())
        # The original study's descriptive TOML predates the executable recipe schema;
        # its concrete members are recovered from the completed analysis above.
        if prefix or "schema_version" in document:
            from .study_analysis import load_derived_judges
            definitions = load_derived_judges(study_dir, {v["variant"]: v for v in variants})
            catalog_rounds.append({"name": None, "judges": definitions})
    for round_ in catalog_rounds:
        for judge in round_.get("judges", []):
            name = judge["variant"]
            if name in catalog:
                if round_.get("name"):
                    catalog[name].setdefault("evaluated_rounds", []).append(round_["name"])
                continue
            if not judge.get("members"):
                continue
            member_calls = []
            for member, draw in judge["members"]:
                base = catalog.get(member, {})
                configured = (base.get("members") or [{}])[0]
                weights = next((item for item in judge.get("weighted_members", [])
                                if item.get("variant") == member and item.get("draw") == draw), {})
                member_calls.append({"variant": member, "catalog_key": prefix + member, "draw": draw,
                                     "weight": weights.get("weight", 1.0),
                                     "normalized_weight": weights.get("normalized_weight", 1 / len(judge["members"])),
                                     "family": weights.get("family"), **configured})
            catalog[name] = {"key": prefix + name, "variant": name, "label": judge.get("label", name),
                             "group": ("Focused follow-up" if prefix == "followup:" else "Validation extension") if prefix else "Original study",
                             "kind": "derived", "aggregation": judge.get("aggregation"),
                             "member_calls": member_calls, "evaluated_rounds": [round_["name"]] if round_.get("name") else [],
                             "judge_id": judge.get("judge_id"), "version": judge.get("judge_version"),
                             "derived_definition": judge.get("derived_definition"),
                             "definition_sha256": judge.get("definition_sha256"),
                             "definition_file_sha256": judge.get("definition_file_sha256"),
                             "examples": {"status": "inherited", "note": "Uses the exact member calls; no additional model prompt."}}
            unavailable = [member for member, _draw in judge["members"]
                           if (catalog.get(member, {}).get("availability") or {}).get("status") == "unavailable"]
            if unavailable:
                catalog[name]["availability"] = {"status": "unavailable", "basis": "required_member_unavailable",
                    "reason": "Required member unavailable: " + ", ".join(unavailable) +
                    ". The frozen recipe is not renormalized around missing members."}
    if any(round_.get("name") == "legacy_development" for round_ in rounds):
        legacy_path = study_dir.parent / "quant-100-20260910/judgments.jsonl"
        wanted = {j["variant"] for r in rounds if r.get("name") == "legacy_development" for j in r.get("judges", [])}
        if legacy_path.is_file():
            with legacy_path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    name = row.get("judge_id")
                    if name not in wanted:
                        continue
                    saved, calls = row.get("judge", {}), row.get("calls", [])
                    catalog[name] = {"key": prefix + name, "variant": name, "label": "Legacy " + name,
                        "group": "Legacy development", "kind": "legacy", "judge_id": name,
                        "version": saved.get("version"), "configuration": saved,
                        "saved_structure": saved.get("structure"), "system_prompt": saved.get("system_prompt"),
                        "user_template": saved.get("user_template"), "members": saved.get("members", []),
                        "member_calls": [{**c.get("member", {}), "order": c.get("order"), "call_index": c.get("call_index")} for c in calls],
                        "output_schema": calls[0].get("response_schema") if calls else None,
                        "schema_source": "Historical saved request; criterion IDs vary by pair.",
                        "response_style": "score_reason", "plan_projection": "as_stored",
                        "examples": {"status": "none", "n_examples": 0}, "evaluated_rounds": ["legacy_development"]}
                    wanted.remove(name)
                    if not wanted:
                        break
    private_keys = {"author", "author_model", "source_model", "model", "models", "model_a", "model_b",
                    "model_id", "student_id", "subject", "subject_id", "reviewer_id", "reviewer_name",
                    "tutor_id", "plan_id", "plan_ids", "grading_id", "session_id"}
    def masked(value):
        if isinstance(value, dict):
            return {key: masked(item) for key, item in value.items() if key not in private_keys}
        if isinstance(value, list):
            return [masked(item) for item in value]
        return value
    for entry in catalog.values():
        example = dict(entry.get("examples", {}))
        text = _local_text(study_dir, example.get("artifact_file"))
        if text:
            try:
                example["display_artifact"] = masked(json.loads(text))
                example["display_note"] = "Identity metadata removed from this display; the exact prompt template is available in the Prompts tab."
            except json.JSONDecodeError:
                example["display_note"] = "The example artifact is not JSON; consult the exact saved prompt template."
        entry["examples"] = example
    return list(catalog.values())


def _extension(study_dir: Path, pointer_key: str = "study_dir", catalog_prefix: str = "extension:") -> dict | None:
    pointer = _load(study_dir / "report_extension.json", {})
    if not pointer.get(pointer_key):
        return None
    root = (study_dir / pointer[pointer_key]).resolve()
    if not root.is_relative_to(study_dir.parent) or not root.is_dir():
        return None
    metadata = _load(root / "report_metadata.json", {})
    summary = _load(root / "summary.json", {})
    rounds = []
    for filename in metadata.get("analysis_files", []):
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            continue
        result = _load(path)
        if result:
            result.setdefault("name", path.stem.removesuffix("_analysis"))
            result["source_file"] = filename
            rounds.append(result)
    variants = _load(root / "variants.json", [])
    configurations = _configurations(root, variants)
    artifact_labels = {
        "summary.json": "Extension · runtime and cost summary",
        "verification.json": "Extension · verification",
        "source_snapshot.json": "Extension · source checksums",
        "source_snapshot.zip": "Extension · source archive (ZIP)",
        "paired_deltas.json": "Extension · paired metric changes",
        "provider_validation_analysis.json": "Extension · provider analysis",
        "example_validation_analysis.json": "Extension · example analysis",
        "terminal_failures.json": "Extension · missing-request ledger",
        "execution_interruptions.json": "Extension · interruption ledger",
    }
    for path in sorted((root / "exports").glob("*.csv")):
        artifact_labels[str(path.relative_to(root))] = "Extension · " + path.stem.replace("_", " ") + " (CSV)"
    for filename in metadata.get("analysis_files", []):
        artifact_labels.setdefault(filename, "Extension · " + Path(filename).stem.replace("_", " "))
    if catalog_prefix == "followup:":
        artifact_labels = {key: value.replace("Extension · ", "Focused follow-up · ")
                           for key, value in artifact_labels.items()}
    # Normalize links to a sibling-relative path, and exclude absent artifacts or
    # symlinks whose targets escape the validated extension root.
    directory_link = "../" + root.relative_to(study_dir.parent).as_posix()
    artifacts = [{"path": directory_link + "/" + filename, "label": label}
                 for filename, label in artifact_labels.items()
                 if (root / filename).resolve().is_relative_to(root) and (root / filename).is_file()]
    example_audit = _load(root / "example_audit.json")
    if example_audit:
        # Display numbered illustrations, retaining the complete provenance only
        # in the linked local artifact rather than exposing student/plan IDs.
        example_audit = {
            "sample": {key: value for key, value in example_audit.get("sample", {}).items()
                       if key in {"n_pairs", "n_criteria", "selection", "representative"}},
            "summary": example_audit.get("summary", {}),
            "interpretation": example_audit.get("interpretation"),
            "copying_check": {key: value for key, value in example_audit.get("copying_check", {}).items()
                              if key in {"result", "method", "limitations"}},
            "findings": [{**{key: value for key, value in finding.items()
                              if key in {"title", "pair_number", "criterion", "observation"}},
                          "snippets": [{key: value for key, value in snippet.items()
                                        if key in {"source", "field", "text"}}
                                       for snippet in finding.get("snippets", [])]}
                         for finding in example_audit.get("findings", [])]}
    return {"directory": directory_link, "catalog_prefix": catalog_prefix, "metadata": metadata, "summary": summary, "rounds": rounds,
            "artifacts": artifacts,
            "paired_deltas": _load(root / "paired_deltas.json"),
            "example_audit": example_audit,
            "variants": variants, "configurations": configurations,
            "catalog": _judge_catalog(root, variants, rounds, configurations,
                                      {**metadata, "availability": summary.get("availability", {}),
                                       "serialization_by_variant": summary.get("serialization_by_variant", {})}, catalog_prefix)}


def _public_leaderboard(study_dir: Path) -> dict | None:
    """Read only aggregate fields from a bounded, explicitly declared artifact."""
    pointer_path = study_dir / "report_leaderboard.json"
    if not pointer_path.exists():
        return None
    pending = {"status": "pending"}
    try:
        filename = _load(pointer_path, {}).get("evaluation_file")
        if not isinstance(filename, str) or not filename:
            return pending
        path = (study_dir / filename).resolve()
        if not path.is_relative_to(study_dir.parent) or not path.is_file():
            return pending
        value = _load(path)
        if not isinstance(value, dict):
            return pending
    except (OSError, ValueError, TypeError, AttributeError):
        return pending

    def scalar(value):
        return value if isinstance(value, (str, int, bool)) or isinstance(value, float) and math.isfinite(value) else None

    def fit(source):
        if not isinstance(source, dict):
            return {}
        rows = source.get("leaderboard", [])
        if not isinstance(rows, list):
            rows = []
        return {"connected": source.get("connected") is True, "converged": source.get("converged") is True,
                "models": [m for m in source.get("models", []) if isinstance(m, str)] if isinstance(source.get("models"), list) else [],
                "leaderboard": [{k: scalar(row.get(k)) for k in ("model", "rank", "score", "n_comparisons", "n_subjects")}
                                for row in rows if isinstance(row, dict)]}

    coverage = value.get("coverage") if isinstance(value.get("coverage"), dict) else {}
    intervals = value.get("rank_intervals") if isinstance(value.get("rank_intervals"), dict) else {}
    rubrics = value.get("per_rubric") if isinstance(value.get("per_rubric"), dict) else {}
    generation = value.get("generation_coverage") if isinstance(value.get("generation_coverage"), list) else []
    conditions = value.get("evaluation_conditions") if isinstance(value.get("evaluation_conditions"), dict) else {}
    bootstrap = value.get("bootstrap") if isinstance(value.get("bootstrap"), dict) else {}
    judge = conditions.get("judge") if isinstance(conditions.get("judge"), dict) else {}
    members = judge.get("members") if isinstance(judge.get("members"), list) else []
    routes = conditions.get("routes") if isinstance(conditions.get("routes"), list) else []
    requested = value.get("requested_models") if isinstance(value.get("requested_models"), list) else [row.get("model") for row in generation if isinstance(row, dict)]
    return {"status": "complete" if value.get("complete") is True else "pending",
            "requested_models": [model for model in requested if isinstance(model, str)],
            "deferred_models": [{k: scalar(row.get(k)) for k in ("model", "label", "status")} for row in value.get("deferred_models", []) if isinstance(row, dict)] if isinstance(value.get("deferred_models"), list) else [],
            "ranking": fit(value.get("ranking")),
            "per_rubric": {str(k): fit(v) for k, v in rubrics.items()},
            "rank_intervals": {str(k): {field: scalar(v.get(field)) for field in ("low", "high", "n_valid")}
                               for k, v in intervals.items() if isinstance(v, dict)},
            "coverage": {k: scalar(coverage.get(k)) for k in ("scheduled_comparisons", "completed_comparisons", "missing_comparisons", "scheduled_calls", "completed_calls", "missing_calls", "requested_criteria", "scored_criteria", "missing_criteria", "abstained_criteria")},
            "bootstrap": {k: scalar(bootstrap.get(k)) for k in ("requested", "valid", "method")},
            "conditions": {**{k: scalar(conditions.get(k)) for k in ("seed", "comparisons_per_student", "include_new_pairs")},
                           "judge": {**{k: scalar(judge.get(k)) for k in ("adapter", "model", "reasoning_effort", "judge_id", "strategy", "version") if k in judge}, "members": [{k: scalar(m.get(k)) for k in ("name", "model", "reasoning_effort")} for m in members if isinstance(m, dict)]},
                           "routes": [{"model": scalar(r.get("model")), "reasoning_effort": scalar(r.get("reasoning_effort")),
                                       "settings": {k: scalar(r["settings"].get(k)) for k in ("max_tokens", "max_output_tokens", "max_completion_tokens", "temperature", "top_p") if k in r["settings"]} if isinstance(r.get("settings"), dict) else {}}
                                      for r in routes if isinstance(r, dict)]},
            "generation_coverage": [{"model": scalar(row.get("model")), "label": scalar(row.get("label")),
                "reasoning_effort": scalar(row.get("reasoning_effort")), "status": scalar(row.get("status")),
                "settings": {k: scalar(row.get("settings", {}).get(k)) for k in ("reasoning_effort", "temperature", "top_p", "max_output_tokens", "max_tokens", "max_completion_tokens") if k in row.get("settings", {})} if isinstance(row.get("settings"), dict) else {},
                **{k: len(row[k]) if isinstance(row.get(k), list) else scalar(row.get(k)) for k in ("selected_subjects", "completed_subjects", "pending_subjects")}}
                for row in generation if isinstance(row, dict)]}


def render_report(study_dir: Path) -> Path:
    """Render available analyses, judge snapshots, uncertainty, density and audit."""
    study_dir = Path(study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load(study_dir / "manifest.json", {})
    split_counts = {key: len(value) for key, value in manifest.get("splits", {}).items()}
    pair_counts = {key: {"available": len(rows), "human_reviewed": sum(bool(r.get("human_reviewed")) for r in rows)}
                   for key, rows in manifest.get("pair_catalog", {}).items()}
    manifest = {key: value for key, value in manifest.items() if key not in ("pair_catalog", "splits")}
    manifest.update(split_subject_counts=split_counts, pair_counts=pair_counts)
    rounds = []
    for filename in ("legacy_development_analysis.json", "round1_analysis.json", "round2_analysis.json", "test_analysis.json", "dense_analysis.json"):
        value = _load(study_dir / filename)
        if value:
            value.setdefault("name", filename.removesuffix("_analysis.json"))
            value["source_file"] = filename
            rounds.append(value)
    final = _load(study_dir / "analysis.json", {})
    selection = final.get("selected_judge") or _load(study_dir / "selection.json") or manifest.get("selection")
    variants = _load(study_dir / "variants.json", [])
    configurations = _configurations(study_dir, variants)
    context_coverage = _load(study_dir / "context/coverage.json", {})
    context_provenance = _load(study_dir / "context/provenance.json") or context_coverage.get("provenance", {})
    input_ablation = _load(study_dir / "input_ablation_analysis.json")
    input_jobs = _load(study_dir / "input_ablation_jobs.json")
    input_cohort = {"verified": False, "reason": "Matched-cohort analysis and pair identities are pending."}
    if input_ablation:
        expected_pairs = set(manifest.get("round1_validation_pairs", []))
        if input_jobs is not None:
            actual_pairs = {job["pair_id"] for job in input_jobs}
            hash_matches = digest(input_jobs) == input_ablation.get("jobs_sha256")
            expected_variants = {"luna_high", "terra_high", "enhanced_luna_high", "enhanced_terra_high"}
            variant_pairs = {variant: {job["pair_id"] for job in input_jobs if job["variant"] == variant}
                             for variant in expected_variants}
            variant_cohorts_match = all(pairs == expected_pairs for pairs in variant_pairs.values())
            verified = bool(expected_pairs) and actual_pairs == expected_pairs and hash_matches and variant_cohorts_match
            input_cohort = {"verified": verified, "n_pairs": len(actual_pairs), "jobs_hash_matches": hash_matches,
                            "each_variant_matches_screen": variant_cohorts_match,
                            "reason": "Pair identities and job fingerprint match the original screening cohort." if verified else
                            "Pair identities or job fingerprint differ from the original screening cohort."}
        elif isinstance(input_ablation.get("pair_ids"), list):
            actual_pairs = set(input_ablation["pair_ids"])
            verified = bool(expected_pairs) and actual_pairs == expected_pairs
            input_cohort = {"verified": verified, "n_pairs": len(actual_pairs),
                            "reason": "Analysis pair identities match the original screening cohort." if verified else
                            "Analysis pair identities differ from the original screening cohort."}
    # The orchestrator may keep round artifacts nested in its final summary.
    for name in ("round1", "round2", "test", "dense"):
        value = final.get(name)
        if isinstance(value, dict) and value.get("judges") and not any(r["name"] == name for r in rounds):
            rounds.append({"name": name, **value})
    metadata = _load(study_dir / "report_metadata.json", {})
    extension = _extension(study_dir)
    followup = _extension(study_dir, "followup_study_dir", "followup:")
    revision = None
    revision_source = None
    human_comparator = None
    human_comparator_source = None
    revision_pointer = _load(study_dir / "report_extension.json", {}).get("revision_statistics_file")
    if revision_pointer:
        revision_path = (study_dir / revision_pointer).resolve()
        if revision_path.is_relative_to(study_dir.parent) and revision_path.is_file():
            revision = _load(revision_path)
            revision_source = "../" + revision_path.relative_to(study_dir.parent).as_posix()
            if isinstance(revision.get("reasoning_effort"), dict):
                revision["reasoning_effort"].pop("input_call_sha256", None)
            comparator_path = revision_path.parent / "human_comparator.json"
            if comparator_path.is_file() and comparator_path.resolve().is_relative_to(study_dir.parent):
                human_comparator = _load(comparator_path)
                human_comparator_source = "../" + comparator_path.relative_to(study_dir.parent).as_posix()
    catalog = _judge_catalog(study_dir, variants, rounds, configurations, metadata)
    if extension:
        catalog.extend(extension.pop("catalog"))
    if followup:
        catalog.extend(followup.pop("catalog"))
    sources = [str(path.relative_to(study_dir)) for path in sorted(study_dir.rglob("*.json"))
               if (path.parent in (study_dir, study_dir / "human", study_dir / "audit")
                   or path.parent == study_dir / "context" and path.name in ("coverage.json", "provenance.json"))
               and not path.name.endswith("_jobs.json") and path.name not in ("manual_sample.json", "screening_candidates.json")]
    sources.extend(name for name in (
        "exports/human_test_leaderboard.csv", "exports/selected_ai_reviewed_leaderboard.csv",
        "exports/selected_ai_dense_leaderboard.csv", "report-before-extension-20260911.html")
        if (study_dir / name).exists())
    sources.sort()
    data = _compact({
        "generated_at": datetime.now(timezone.utc).isoformat(), "study_name": study_dir.name,
        "manifest": manifest, "variants": variants, "configurations": configurations,
        "judge_catalog": catalog, "extension": extension, "followup": followup,
        "revision": revision, "revision_source": revision_source,
        "human_comparator": human_comparator, "human_comparator_source": human_comparator_source,
        "selection": selection, "selection_rule": _load(study_dir / "selection_rule.json", {}),
        "context": {"coverage": context_coverage, "provenance": context_provenance},
        "input_ablation": input_ablation, "input_ablation_cohort": input_cohort,
        "rounds": rounds, "final": final,
        "human": {name: _load(study_dir / "human" / f"{name}.json") for name in ("validation", "test")},
        "human_cross_split": _load(study_dir / "human/validation_test_human_alignment.json"),
        "audit": {"summary": _load(study_dir / "audit/summary.json", {}),
                  "findings": _load(study_dir / "audit/findings.json", [])},
        "sources": sources,
    })
    public = _public_leaderboard(study_dir)
    if public is not None:
        public["baseline_models"] = _public_baselines(study_dir)
        data["public_leaderboard"] = public
    # Escaping '<' also prevents a saved explanation from closing a script tag.
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    document = TEMPLATE.replace("__STUDY_DATA__", serialized).replace("__STUDY_TITLE__", html.escape(study_dir.name))
    if public is not None:
        document = _inject_public_leaderboard(document)
    target = study_dir / "report.html"
    target.write_text(document)
    return target


def _public_baselines(study_dir: Path) -> list[str]:
    analysis = _load(study_dir / "test_analysis.json", {})
    models = analysis.get("models", []) if isinstance(analysis, dict) else []
    return [model for model in models if isinstance(model, str)] if isinstance(models, list) else []


def _public_validation(study_dir: Path) -> dict:
    selection = _load(study_dir / "selection.json", {})
    test = _load(study_dir / "test_analysis.json", {})
    if not isinstance(selection, dict) or not isinstance(test, dict):
        return {}
    variant = selection.get("variant")
    judge = next((j for j in test.get("judges", []) if isinstance(j, dict) and j.get("variant") == variant), {})
    metrics = judge.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {"variant": variant, "n_pairs": test.get("n_pairs"), "n_subjects": test.get("n_subjects"),
            "metrics": {k: metrics.get(k) for k in ("kendall_tau_b", "spearman", "pair_order_agreement")}}


def render_public_report(study_dir: Path) -> Path:
    """Render only the aggregate first-release section, without validation data."""
    study_dir = Path(study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    public = _public_leaderboard(study_dir) or {"status": "pending"}
    public["baseline_models"] = _public_baselines(study_dir)
    validation = _public_validation(study_dir)
    serialized = json.dumps({"public_leaderboard": public, "validation": validation, "human": {}}, ensure_ascii=False, allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    style = TEMPLATE.split("<style>", 1)[1].split("</style>", 1)[0]
    helpers = TEMPLATE.split("const D=", 1)[1].split("\nlet activeCriterion=", 1)[0]
    # The selector uses the same model names and criterion labels as the report.
    names = TEMPLATE.split("function modelName(raw){", 1)[1].split("function judgeName(", 1)[0]
    labels = TEMPLATE.split("const criterionLabel=", 1)[1].split("\n", 1)[0]
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>StudentBench · AI-judged leaderboard</title><style>' + style + '</style></head><body>'
                '<header><div class="eyebrow">StudentBench / first public leaderboard</div><h1>AI-judged lesson-plan quality leaderboard</h1></header>'
                '<main>' + PUBLIC_SECTION + PUBLIC_VALIDATION + '</main><script type="application/json" id="study-data">' + serialized + '</script>'
                '<script>const D=' + helpers + '\nfunction modelName(raw){' + names + '\nconst criterionLabel=' + labels + '\n' + PUBLIC_SCRIPT + PUBLIC_VALIDATION_SCRIPT + '</script></body></html>')
    target = study_dir / "public-report.html"
    target.write_text(document)
    return target


def _inject_public_leaderboard(document: str) -> str:
    document = document.replace('<title>StudentBench · Model ranking validation</title>', '<title>StudentBench · AI-judged leaderboard and validation</title>')
    document = document.replace('StudentBench / judge validation study</div><h1>Do AI judges recover the human ranking of lesson-plan generator systems?</h1><p>We compare human and AI rankings of systems that generate lesson plans, then test which judge designs and comparison schedules produce a similar leaderboard.</p>', 'StudentBench / first public leaderboard</div><h1>AI-judged lesson-plan quality leaderboard</h1><p>Compare the requested generator systems, then explore the judge-validation studies below.</p>')
    document = document.replace('<nav aria-label="Report sections">', '<nav aria-label="Report sections"><a href="#public-leaderboard">Public leaderboard</a>')
    document = document.replace('<section id="overview">', PUBLIC_SECTION + '<section id="overview">')
    return document.replace("$('print-report').addEventListener", PUBLIC_SCRIPT + "\n$('print-report').addEventListener")


PUBLIC_SECTION = r'''<section id="public-leaderboard"><div class="section-head"><h2>First public leaderboard · AI-judged quality</h2><span class="badge" id="public-status">Pending</span></div><p class="lede">This ranking compares lesson-plan generator systems on GRE Quant for 83 frozen benchmark student contexts: the original 12 systems and the requested new generators. A system includes its model and generation settings. Ranks measure delivered schema-valid plans. Historical baseline plans can include answer-verifier edits; new plans use the released prompt, reconstructed context renderer, and schema/options normalization without that historical answer verifier. The generation pipelines are not historically equivalent, so these rankings describe the recorded generator systems rather than a pure comparison of models. Provider and validation failures are retained in the audit. The AI judge evaluates plan quality; this is not a human grade or a measure of student learning.</p><div id="public-coverage"></div><div class="toolbar"><label class="control">Quality criterion<select id="public-criterion"></select></label></div><div id="public-ranking" class="scroll"></div><p class="caption">Rank 1 is highest estimated quality. Bradley–Terry scores summarize pairwise preferences. Overall rank ranges are 95% student-context bootstrap percentile intervals, conditional on connected, converged fits; they do not establish a unique winner. Missing or abstained judgments are not ties. Criterion-specific fits need not share the overall ordering.</p><div id="public-model-settings"></div><div id="public-deferred"></div></section>'''

PUBLIC_SCRIPT = r'''
function publicLeaderboard(){
 const p=D.public_leaderboard;if(!p)return;const status=$('public-status'),select=$('public-criterion');
 const labels=new Map((p.generation_coverage||[]).filter(x=>x.model&&x.label).map(x=>[x.model,x.label]));const keys=['overall',...Object.keys(p.per_rubric||{}).sort()];select.innerHTML=keys.map(k=>'<option value="'+esc(k)+'">'+esc(criterionLabel(k))+'</option>').join('');
 const c=p.coverage||{},cond=p.conditions||{};$('public-coverage').innerHTML='<p class="small">Comparisons completed: '+integer(c.completed_comparisons)+' / '+integer(c.scheduled_comparisons)+'; missing: '+integer(c.missing_comparisons)+'. Usable rubric answers: '+integer(c.scored_criteria)+' / '+integer(c.requested_criteria)+'; missing: '+integer(c.missing_criteria)+'; abstained: '+integer(c.abstained_criteria)+'. Bootstrap fits: '+integer(p.bootstrap?.valid)+' / '+integer(p.bootstrap?.requested)+'. Comparisons per student: '+integer(cond.comparisons_per_student)+'; new-to-new comparisons: '+esc(cond.include_new_pairs==null?'not recorded':cond.include_new_pairs?'included':'excluded')+'. Judge: '+esc(Object.entries(cond.judge||{}).filter(([k])=>k!=='members').map(([k,v])=>k.replaceAll('_',' ')+': '+v).concat((cond.judge?.members||[]).map(m=>[m.model,m.reasoning_effort].filter(Boolean).join(' · '))).join('; ')||'not recorded')+'. Judge route settings: '+esc((cond.routes||[]).map(r=>[r.model,r.reasoning_effort,...Object.entries(r.settings||{}).map(([k,v])=>k.replaceAll('_',' ')+': '+v)].filter(v=>v!=null&&v!=='').join(' · ')).filter(Boolean).join('; ')||'not recorded')+'.</p>';
 function draw(){const key=select.value,fit=key==='overall'?p.ranking:p.per_rubric?.[key],rows=fitRows(fit),baseline=p.baseline_models||[],newModels=p.requested_models||[],generationModels=(p.generation_coverage||[]).map(x=>x.model),roster=[...baseline,...newModels],expected=new Set(roster),models=new Set(rows.map(r=>r.model)),fitModels=Array.isArray(fit?.models)?fit.models:[],sameRoster=ids=>ids.length===roster.length&&new Set(ids).size===roster.length&&ids.every(id=>typeof id==='string'&&expected.has(id));const ready=p.status==='complete'&&baseline.length===12&&newModels.length>0&&new Set(newModels).size===newModels.length&&generationModels.length===newModels.length&&new Set(generationModels).size===newModels.length&&generationModels.every(id=>newModels.includes(id))&&expected.size===roster.length&&sameRoster(fitModels)&&sameRoster(rows.map(r=>r.model))&&fit?.connected===true&&fit?.converged===true&&rows.every(r=>valid(r.rank));status.textContent=ready?'Complete':'Pending / unavailable';status.classList.toggle('pending',!ready);if(!ready){$('public-ranking').innerHTML=empty('A complete, connected requested-system '+criterionLabel(key)+' ranking is not available yet. No ranks are inferred from the earlier 12-system study.');const base=(p.baseline_models||[]).map(model=>[esc(modelName(model)),'—','Original system · pending global ranking']);const newer=(p.generation_coverage||[]).map(x=>[esc(x.label||modelName(x.model)),'—',esc(x.status||((x.pending_subjects||0)>0?'Generation pending':'Awaiting evaluation'))]);$('public-ranking').innerHTML+=table(['Generator system','Rank','Status'],[...base,...newer]);return}$('public-ranking').innerHTML=table(['Generator system','Rank','95% student bootstrap rank range','BT score','Comparisons','Students'],rows.map(r=>['<strong>'+esc(labels.get(r.model)||modelName(r.model))+'</strong>',num(r.rank,0),key==='overall'?interval(p.rank_intervals?.[r.model],0):'—',num(r.score),integer(r.n_comparisons),integer(r.n_subjects)]));}
 select.addEventListener('change',draw);draw();const deferred=p.deferred_models||[];if(deferred.length)$('public-deferred').innerHTML='<p class="small">Deferred from this release: '+deferred.map(x=>esc(x.label||modelName(x.model))).join(', ')+'.</p>';const configs=p.generation_coverage||[];if(configs.length)$('public-model-settings').innerHTML='<details><summary>New generator models and settings</summary>'+table(['Generator system','Generation settings','Completed / selected students'],configs.map(x=>[esc(x.label||modelName(x.model)),esc([...(x.reasoning_effort!=null?['reasoning effort: '+x.reasoning_effort]:[]),...Object.entries(x.settings||{}).map(([k,v])=>k.replaceAll('_',' ')+': '+v)].join('; ')||'Not recorded'),integer(x.completed_subjects)+' / '+integer(x.selected_subjects)]))+'</details>';
}
publicLeaderboard();
'''


PUBLIC_VALIDATION = r'''<section id="public-validation"><h2>Judge validation and methods</h2><div class="card"><p>The judge was selected on validation students before examining the held-out test. The held-out comparison below concerns the original 12 generator systems; it is evidence about the judge, not a human grade for new generators.</p><div id="public-validation-stats"></div><p>Pairwise rubric preferences are combined into a Bradley–Terry quality ranking. Ties count as half wins; missing and abstained judgments do not. Student contexts are the resampling unit. Evaluation schedules and model generation settings affect the estimates.</p><p>Density analyses are sensitive to opponent schedules, student weights and the fixed Bradley–Terry ridge penalty; stronger relative shrinkage at lower density complicates comparisons. Validation agreement does not establish a certain winner or measure student learning gains.</p></div></section>'''
PUBLIC_VALIDATION_SCRIPT = r'''const v=D.validation||{},m=v.metrics||{};document.getElementById('public-validation-stats').innerHTML='<p>Frozen selected judge: '+esc(v.variant||'pending')+'. Held-out test: '+integer(v.n_pairs)+' human-reviewed pairs across '+integer(v.n_subjects)+' students. Against the human reference, Kendall τ-b: '+num(m.kendall_tau_b)+', Spearman ρ: '+num(m.spearman)+', model-order agreement: '+pct(m.pair_order_agreement)+'.</p>';'''


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>StudentBench · Model ranking validation</title>
<style>
:root{--ink:#192b36;--muted:#59707d;--line:#dbe3e6;--paper:#f5f7f7;--teal:#087f83;--blue:#5571c4;--amber:#8f590d;--red:#ab4848;--green:#28765c}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--teal)}button,select{font:inherit;color:var(--ink)}button,select{border:1px solid #c4d1d7;border-radius:7px;background:#fff;padding:9px 12px;cursor:pointer}button:hover,select:hover{border-color:var(--teal)}button.active{background:var(--teal);border-color:var(--teal);color:#fff}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #d59c38;outline-offset:3px}
header{background:#102f3c;color:#fff;padding:48px max(24px,calc((100vw - 1240px)/2)) 36px;border-bottom:5px solid #45bdba}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.15em;font-weight:750;color:#aadbdc}h1{font-size:clamp(30px,4vw,49px);line-height:1.13;letter-spacing:-.035em;max-width:930px;margin:13px 0 18px}header p{color:#d3e2e7;max-width:870px;font-size:17px}header .meta{font-size:12px;color:#b2c5ce;display:flex;gap:22px;flex-wrap:wrap}
nav{position:sticky;top:0;z-index:4;background:#ffffffef;backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:12px 24px;display:flex;justify-content:flex-start;gap:20px;overflow:auto}nav>*{flex-shrink:0}nav a{font-size:13px;white-space:nowrap;text-decoration:none;color:var(--ink);font-weight:650}main{max-width:1288px;margin:auto;padding:28px 24px 70px}section{scroll-margin-top:70px;margin-bottom:34px}.section-head{display:flex;justify-content:space-between;align-items:baseline;gap:16px}h2{font-size:25px;letter-spacing:-.02em;margin:0 0 10px}h3{font-size:17px;margin:0 0 12px}h4{font-size:15px;margin:15px 0 7px}p{margin:0 0 14px}.lede{color:var(--muted);max-width:1010px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:23px;box-shadow:0 3px 13px #17344205}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:19px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:20px}.stat{background:white;border:1px solid var(--line);border-radius:10px;padding:17px 19px}.stat .value{font-size:31px;font-weight:720;letter-spacing:-.03em;color:var(--teal);line-height:1.2}.stat .label{font-size:12px;font-weight:650;margin-top:8px}.stat .note{font-size:11px;color:var(--muted);margin-top:5px}.notice{border-left:4px solid #b87f2e;background:#fff7e9;border-radius:5px;padding:13px 16px;font-size:13px;margin:14px 0}.notice.good{border-left-color:var(--teal);background:#eef8f6}.small,.caption{font-size:12px;color:var(--muted)}.caption{margin-top:12px}.empty{padding:28px;border:1px dashed #b7c7d0;border-radius:8px;color:var(--muted);background:#fcfdfd}.toolbar{display:flex;align-items:end;gap:14px;flex-wrap:wrap;margin:18px 0}.control{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}.control select{font-weight:450;font-size:14px;max-width:440px}.tabs{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:18px}.tabs button{font-size:12px;padding:7px 10px}.legend{font-size:12px;display:flex;gap:20px;align-items:center;color:var(--muted);margin:7px 0 17px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;background:var(--teal)}.dot.ai{background:var(--blue)}.badge{display:inline-block;font-size:11px;border-radius:20px;padding:3px 9px;background:#eaf3f3;color:#216d72;font-weight:650}.badge.pending{background:#fff3d9;color:#8b5d14}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}th{text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.025em;font-weight:750;color:var(--muted);padding:11px 12px;background:#f3f6f7;border-bottom:1px solid var(--line);white-space:nowrap}td{text-align:right;padding:11px 12px;border-bottom:1px solid #ebeff1;vertical-align:top}th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:none}tbody tr:hover{background:#f5faf9}td strong{font-weight:650}td .sub{display:block;font-size:11px;color:var(--muted);margin-top:3px}.positive{color:var(--green)}.negative{color:var(--red)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}.ranked{background:#f0faf7}details{border:1px solid var(--line);border-radius:8px;padding:12px 15px;background:white;margin-top:11px}summary{cursor:pointer;font-size:13px;font-weight:650}details p{font-size:13px;margin:12px 0 0}pre{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f3f6f7;padding:15px;border-radius:5px;overflow:auto;max-height:440px;white-space:pre-wrap;overflow-wrap:anywhere}blockquote{border-left:3px solid #9dbbbf;padding-left:13px;margin:12px 0;color:#364e5c;font-size:13px}.audit-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.audit-item{border-top:3px solid #a4bfc5}.audit-item h3{margin-top:3px}.audit-item p{font-size:13px}.source-links{display:flex;gap:12px 22px;flex-wrap:wrap}.source-links a{font-size:12px}.chart svg{width:100%;height:auto;display:block}.chart text{font-family:inherit;fill:var(--muted);font-size:11px}.chart .axis-title{font-size:12px;font-weight:650;fill:var(--ink)}.chart .gridline{stroke:#e5eaed;stroke-width:1}.chart .plot-line{fill:none;stroke-width:2.5}.density-controls{display:flex;gap:12px;flex-wrap:wrap}.density-controls select{max-width:100%;font-size:13px}.row-title{margin-top:28px}.protocol-list{padding-left:19px;color:var(--muted);font-size:13px}.protocol-list li{margin:8px 0}.footer{font-size:12px;color:var(--muted);padding-top:23px;border-top:1px solid var(--line)}.print-button{white-space:nowrap;margin-left:auto;font-size:12px;padding:3px 10px}noscript{display:block;padding:30px;color:#8b3d37}
@media(max-width:800px){header{padding:32px 23px}.grid,.audit-grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}nav{justify-content:flex-start;gap:20px}main{padding:24px 15px}.card{padding:17px}.control select{max-width:85vw}.section-head{display:block}.stat .value{font-size:26px}}
@media print{nav,.toolbar,.tabs,.print-button,.density-controls{display:none}body{background:white;font-size:11px}header{padding:20px;background:white;color:var(--ink);border-bottom:2px solid var(--teal)}header p,header .meta,.eyebrow{color:var(--muted)}main{padding:15px 0;max-width:none}.card,.stat{box-shadow:none;break-inside:avoid}.grid{display:block}.grid>.card{margin-bottom:15px}section{margin-bottom:22px}.audit-grid{display:block}.audit-item{margin-bottom:12px}h1{font-size:30px}.stats{gap:8px}a{color:inherit}details:not([open]){display:none}.chart{break-inside:avoid}}

.table-block{min-width:0}.table-tools{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:6px 0 9px}.table-tools .reset-sort{font-size:11px;padding:4px 9px}.reset-sort:disabled{opacity:.5;cursor:default}thead .sort-column{border:0;background:none;border-radius:3px;padding:0;font:inherit;color:inherit;text-transform:inherit;letter-spacing:inherit;text-align:inherit}.sort-arrow{color:var(--teal)}tbody th{background:transparent;text-transform:none;letter-spacing:normal;font-size:13px;font-weight:450;white-space:normal;color:var(--ink);border-bottom:1px solid #ebeff1;vertical-align:top;text-align:left}tbody tr:last-child th{border-bottom:0}caption{text-align:left;caption-side:bottom;color:var(--muted);font-size:12px;padding:12px 0}.visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.experiment-intro{max-width:970px;font-size:15px;line-height:1.7;margin-bottom:20px}.experiment{margin:24px 0 32px}.experiment h3{font-size:22px;letter-spacing:-.02em}.experiment h4{font-size:17px;margin-top:26px}.experiment .card h4{margin-top:0}.table-section{margin:18px 0 26px}.metric-definition{max-width:970px}.panel-members{font-size:12px;color:var(--muted)}@media print{.table-tools{display:none}thead .sort-column{padding:0}.sort-arrow{display:none}}
</style></head><body>
<header><div class="eyebrow">StudentBench / judge validation study</div><h1>Do AI judges recover the human ranking of lesson-plan generator systems?</h1><p>We compare human and AI rankings of systems that generate lesson plans, then test which judge designs and comparison schedules produce a similar leaderboard.</p><div class="meta"><span>GRE Quant · pairwise lesson-plan judgments</span><span id="study-meta"></span><span>Local report · no external requests</span></div></header>
<nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#study-design">Study design</a><a href="#rounds">Experiments</a><a href="#leaderboard">Held-out leaderboard</a><a href="#extension">Other providers</a><a href="#followup">Smaller panels</a><a href="#judge-design">Sensitivity checks</a><a href="#humans">Human consistency</a><a href="#density">Density</a><a href="#audit">Explanation audit</a><a href="#context">Input context</a><a href="#judge-systems">Judge prompts</a><a href="#protocol">Reproducibility</a><button class="print-button" id="print-report">Print / PDF</button></nav>
<main><noscript>This interactive report needs JavaScript. The linked JSON artifacts contain the complete numerical results.</noscript>
<section id="overview"><div class="section-head"><h2>The question: can an AI judge recover the human leaderboard?</h2><span class="badge" id="study-status"></span></div><p class="lede">Agreement on individual rubric answers and agreement on the final model order are different measurements. All rankings below refer to lesson-plan generator systems: a model together with its generation settings.</p><div id="reading-guide" class="grid" style="margin-bottom:20px"></div><div id="selection-note"></div><div class="notice" id="context-summary-note">The estimates describe the stored generation pipeline. Some practice keys were altered by an upstream answer verifier, and the baseline exported pretest lacks some figures and comparison quantities. These limitations affect interpretation of generator-system quality.</div></section>
<section id="study-design"><h2>What is being compared?</h2><p class="lede">A student completes a pretest, and several generator systems write a lesson plan for that student. A comparison presents two of those plans. Human reviewers and AI judges answer the same quality questions. We aggregate these comparisons to rank the generator systems, rather than rank individual plans.</p><div id="overview-stats" class="stats"></div><div id="study-design-content"></div></section>
<section id="rounds"><h2>Experiments: screen the judge, refine it, then test it</h2><p class="lede">Round 1 asks which model, reasoning setting, and response design are worth pursuing. Round 2 tests a smaller group with more complete input context. We then freeze one choice and evaluate it on the held-out students. Each step below states its comparison schedule and what changed.</p><div id="round-results"></div></section>
<section id="leaderboard"><h2>Experiment 3 · Does the selected judge generalize to held-out students?</h2><p class="lede">The human reference is the leaderboard estimated from reviewers’ pairwise answers. The rubric groups quality questions into eight families, such as relevance and answer accuracy. The default view uses the selected judge on students excluded from its development and selection. The Overall tab combines the eight quality families. Each other tab recomputes both rankings using only that family: Difficulty, for example, ranks generators only on how appropriate the practice difficulty was. These criterion-specific rankings answer narrower questions than the overall leaderboard.</p><div class="toolbar"><label class="control">Evaluation round<select id="round-select"></select></label><label class="control">Judge system<select id="judge-select"></select></label><label class="control">Compare with<select id="compare-select"></select></label></div><p class="small" id="criterion-guide">Choose Overall for the combined quality ranking, or a rubric family to compare rankings on that quality dimension alone.</p><div id="criterion-tabs" class="tabs"></div><div id="ranking-metrics" class="stats"></div><div class="card"><div class="section-head"><h3 id="leaderboard-title">Model ranking</h3><span id="ranking-coverage" class="small"></span></div><div id="ranking-warning"></div><div class="legend"><span><span class="dot"></span><span id="reference-legend">Human reference</span></span><span><span class="dot ai"></span><span id="judge-legend">AI judge</span></span><span>Rank 1 = highest estimated quality</span></div><div id="rank-chart" class="chart"></div><div id="leaderboard-table" class="scroll"></div><p class="caption">Each row ranks a lesson-plan lesson-plan generator system, not the judge model. BT scores summarize pairwise performance after accounting for opponents; score differences determine fitted win probabilities. Model-order agreement compares all 66 unordered pairs among the 12 generator-system ranks, not individual human rubric answers. Rank ranges resample student contexts and are conditional on connected fits; they do not prove a unique winner. “—” means unavailable, not zero.</p></div></section>
<section id="extension" hidden><h2>Experiments 4–5 · Other providers and a human example</h2><p id="extension-protocol" class="lede"></p><p class="lede">We next broaden the judge models and ask whether one worked human comparison helps Terra. These experiments reuse the same 120 validation pairs after the original test was inspected. Their results are exploratory; they do not replace the selected judge’s held-out evaluation.</p><div class="toolbar"><label class="control">Experiment<select id="extension-round-select"></select></label></div><div id="extension-results"></div><div id="extension-deltas"></div><div id="extension-example-audit"></div><details><summary>Provider membership, execution, formatting, and cost details</summary><div id="extension-plan"></div></details></section>
<section id="followup" hidden><h2>Experiment 6 · Can a smaller panel retain the alignment?</h2><div id="followup-results"></div></section>
<section id="judge-design"><h2>Do the conclusions depend on how the leaderboard is built?</h2><p class="lede">The main checks below reuse the selected Terra judge’s held-out answers. They change the amount of influence each student receives, the human answers used as a reference, or the quality questions included. They do not alter the judge or generate new answers.</p><div id="statistical-guide"></div><div id="selected-sensitivities"></div><details><summary>Additional checks for other judges and earlier rounds</summary><div id="extra-sensitivities"></div></details></section>
<section id="humans"><h2>How consistently do humans rank the generator systems?</h2><p class="lede">Different reviewers can disagree on individual answers while producing similar generator rankings. An AI comparison against the full human aggregate can score higher than two smaller human groups compared with each other: the reference and amount of data are different. That does not establish that the AI is better than a human. The comparisons below measure different sources of variation, rather than a universal upper bound for AI performance.</p><div id="human-comparator"></div><div class="toolbar"><label class="control">Human-analysis cohort<select id="human-split-select"></select></label><span id="human-split-note" class="small"></span></div><div id="human-stats" class="stats"></div><div class="grid"><div class="card"><h3>Reviewer-disjoint ranking comparisons</h3><div id="human-consistency"></div></div><div class="card"><h3 id="human-reference-title">Full human reference and rank uncertainty</h3><div id="human-leaderboard" class="scroll"></div></div></div></section>
<section id="density"><h2>How much judging is enough?</h2><p class="lede">Density changes precision and the comparison schedule. Human resampling and AI augmentation are reported separately; adding AI comparisons does not improve the uncertainty of the human reference itself.</p><div class="grid"><div class="card"><h3>Human density and independent reviewers</h3><div class="density-controls"><select id="human-density-ref" aria-label="Human density reference"><option value="density_independent_reviewers">Independent reviewer reference</option><option value="density_self_overlap">Full reference (self-overlap)</option></select><select id="human-density-axis" aria-label="Human density axis"></select></div><div id="human-density-chart" class="chart"></div><div id="human-density-table" class="scroll"></div><p id="human-density-note" class="caption"></p></div><div class="card"><h3>AI density and unseen pair augmentation</h3><div id="ai-density-chart" class="chart"></div><div id="ai-density-table" class="scroll"></div><p class="caption">Curves describe repeated subsampling of completed judgments. Reviewed-pair downsampling and additions without human pair labels answer different questions and must remain separate.</p><p class="caption" id="ai-density-note"></p></div></div></section>
<section id="audit"><h2>What the explanations reveal</h2><p class="lede">The audit used only the previous development run. Saved explanations are concise requested justifications, not internal reasoning traces. Counts from heuristic text screens and the small manual sample are exploratory.</p><div id="audit-stats" class="stats"></div><div id="audit-findings" class="audit-grid"></div></section>
<section id="context"><h2>Input context: restored evidence and remaining gaps</h2><p class="lede">The missing quantities and shared chart text originate in the frozen CSV export. The original application contains compatible source fields. Enhanced variants use a separately pinned supplement; baseline results retain their original inputs.</p><div id="context-stats" class="stats"></div><div class="grid"><div class="card"><h3>What changes in enhanced variants?</h3><div id="context-changes"></div></div><div class="card"><h3>Provenance and limits</h3><div id="context-provenance"></div></div></div></section>
<section id="judge-systems"><h2>Judge systems: recipes, prompts, and examples</h2><p class="lede">A judge system consists of a model call or an explicit combination of calls. Inspect the frozen instructions and requested settings here. The four-model panels use Luna, Terra, Sol, and Astra, with all members at the named effort level.</p><div class="toolbar"><label class="control">Judge system<select id="system-select"></select></label><span id="system-version" class="small mono"></span></div><div id="system-tabs" class="tabs" role="tablist" aria-label="Judge system information"><button data-system-tab="recipe" class="active" role="tab" aria-selected="true">Configuration and members</button><button data-system-tab="prompts" role="tab" aria-selected="false">Exact prompts</button><button data-system-tab="schema" role="tab" aria-selected="false">Response schema</button><button data-system-tab="examples" role="tab" aria-selected="false">Human examples and context</button></div><div id="system-content" class="card" role="tabpanel"></div></section>
<section id="protocol"><h2>Protocol, provenance, and limitations</h2><div class="grid"><div class="card"><h3>Frozen study design</h3><div id="protocol-content"></div></div><div class="card"><h3>Read the results responsibly</h3><ul class="protocol-list"><li>The ranking target is lesson-plan generator systems, not individual lesson plans. Each system combines a model with its generation reasoning settings.</li><li>Ties contribute half a win. Preference strength is omitted in the primary analysis; rubric families receive equal weight by design.</li><li>Disconnected comparison graphs do not support one global leaderboard. Ridge regularization cannot supply missing comparisons.</li><li>Student-context bootstraps retain correlated criteria and pair judgments together. Repeated reviewer effects can still remain.</li><li>Validation selection is adaptive. Report test performance for the frozen system separately from exploratory ablations and sensitivity analyses.</li><li>The study covers GRE Quant and the available generator systems, students, and schedules. It does not measure subsequent student learning gains.</li></ul></div></div><details><summary>Completed artifact summaries</summary><div id="artifact-details"></div></details><h3 class="row-title">Reproducibility files</h3><div id="source-links" class="source-links"></div></section>
<div class="footer">Generated from local analysis artifacts · <span id="generated-at"></span> · <span>__STUDY_TITLE__</span>. No data is fetched by this page. Rendering does not run judges or change the selection state.</div></main>
<script type="application/json" id="study-data">__STUDY_DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('study-data').textContent);
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const valid=x=>typeof x==='number'&&Number.isFinite(x), num=(x,d=3)=>valid(x)?x.toFixed(d):'—', pct=x=>valid(x)?(100*x).toFixed(1)+'%':'—', integer=x=>valid(x)?x.toLocaleString('en-US',{maximumFractionDigits:0}):'—';
const pick=(o,...keys)=>{for(const k of keys)if(o&&o[k]!=null)return o[k];return null}, fitRows=f=>Array.isArray(f)?f:(f?.leaderboard||[]);
let human=D.human.test||D.human.validation;
let tableSequence=0;
function table(headers,rows,caption=''){
 const id='report-table-'+(++tableSequence);
 return '<div class="table-block"><div class="table-tools"><span class="small">Select a column heading to sort.</span><button type="button" class="reset-sort" aria-controls="'+id+'" disabled>Reset order</button><span class="sort-status visually-hidden" role="status"></span></div><table class="sortable" id="'+id+'">'+(caption?'<caption>'+esc(caption)+'</caption>':'')+'<thead><tr>'+headers.map((x,i)=>'<th scope="col" aria-sort="none"><button type="button" class="sort-column" data-column="'+i+'">'+esc(x)+'<span aria-hidden="true" class="sort-arrow"> ↕</span></button></th>').join('')+'</tr></thead><tbody>'+rows.map((r,i)=>'<tr data-original-order="'+i+'">'+r.map((c,k)=>(k===0?'<th scope="row">':'<td>')+c+(k===0?'</th>':'</td>')).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function sortableValue(cell){
 const copy=cell.cloneNode(true);copy.querySelectorAll('.sub,details').forEach(node=>node.remove());const text=copy.textContent.trim(),match=text.replaceAll(',','').replaceAll('−','-').match(/^(?:\[\s*)?[+$]?(-?\d+(?:\.\d+)?)(?:%|\s|$|[–/])/);
 return {text,missing:!text||/^[—–-]$/.test(text),number:match?Number(match[1]):null};
}
function sortReportTable(table,column,descending){
 const rows=[...table.tBodies[0].rows],values=new Map(rows.map(row=>[row,sortableValue(row.cells[column])])),numeric=rows.some(row=>values.get(row).number!==null)&&rows.every(row=>values.get(row).missing||values.get(row).number!==null);
 rows.sort((a,b)=>{const x=values.get(a),y=values.get(b);if(x.missing!==y.missing)return x.missing?1:-1;const comparison=numeric?(x.number-y.number):x.text.localeCompare(y.text,undefined,{numeric:true,sensitivity:'base'});return (descending?-comparison:comparison)||Number(a.dataset.originalOrder)-Number(b.dataset.originalOrder)});
 rows.forEach(row=>table.tBodies[0].appendChild(row));table.querySelectorAll('thead th').forEach((th,i)=>{th.setAttribute('aria-sort',i===column?(descending?'descending':'ascending'):'none');th.querySelector('.sort-arrow').textContent=i===column?(descending?' ↓':' ↑'):' ↕';});const block=table.closest('.table-block');block.querySelector('.reset-sort').disabled=false;block.querySelector('.sort-status').textContent='Sorted by '+table.tHead.rows[0].cells[column].querySelector('button').textContent.replace(/[↕↑↓]/g,'').trim()+', '+(descending?'descending':'ascending')+'.';
}
document.addEventListener('click',event=>{const sort=event.target.closest('.sort-column'),reset=event.target.closest('.reset-sort');if(sort){const th=sort.closest('th');sortReportTable(sort.closest('table'),Number(sort.dataset.column),th.getAttribute('aria-sort')==='ascending')}else if(reset){const block=reset.closest('.table-block'),table=block.querySelector('table');[...table.tBodies[0].rows].sort((a,b)=>Number(a.dataset.originalOrder)-Number(b.dataset.originalOrder)).forEach(row=>table.tBodies[0].appendChild(row));table.querySelectorAll('thead th').forEach(th=>{th.setAttribute('aria-sort','none');th.querySelector('.sort-arrow').textContent=' ↕'});reset.disabled=true;block.querySelector('.sort-status').textContent='Original row order restored.';}});
const empty=msg=>'<div class="empty">'+esc(msg)+'</div>', stat=(value,label,note='')=>'<div class="stat"><div class="value">'+esc(value)+'</div><div class="label">'+esc(label)+'</div><div class="note">'+esc(note)+'</div></div>';
const interval=(v,d=3)=>v&&valid(pick(v,'low','p025'))?'['+num(pick(v,'low','p025'),d)+', '+num(pick(v,'high','p975'),d)+']':'—';
const percentInterval=v=>v&&valid(pick(v,'low','p025'))?pct(pick(v,'low','p025'))+'–'+pct(pick(v,'high','p975')):'—';
const brief=(value)=>'<pre>'+esc(JSON.stringify(value,null,2))+'</pre>';
let activeCriterion='overall';
const roundLabel=x=>({legacy_development:'Legacy judges · development',round1:'Round 1 · validation screening',round2:'Round 2 · validation refinement',test:'Held-out test',dense:'Test · denser AI comparisons'}[x]||x);
function modelName(raw){
 const name=String(raw||'Model not recorded').replace(/^openrouter\//,'').replace(/^(openai|anthropic|google|deepseek|moonshotai|z-ai|meta|qwen)\//,'');
 const names={'gpt-5.6-luna':'GPT 5.6 Luna','gpt-5.6-terra':'GPT 5.6 Terra','gpt-5.6-sol':'GPT 5.6 Sol','gpt-6-astra':'GPT 6 Astra','gemini-3.8-flash':'Gemini 3.8 Flash','claude-opus-5':'Claude Opus 5','claude-sonnet-5':'Claude Sonnet 5','deepseek-v4.1-flash':'DeepSeek 4.1 Flash','deepseek-v4-pro-0813':'DeepSeek 4 Pro (August 13)','kimi-k3':'Kimi K3','glm-5.3':'GLM 5.3','glm-5.3-flash':'GLM 5.3 Flash','muse-spark-1.3':'Muse Spark 1.3'};
 return names[name]||name.replace(/gpt-/g,'GPT ').replace(/claude-/g,'Claude ').replace(/gemini-/g,'Gemini ').replace(/^qwen-?/g,'Qwen ').replace(/opus-/g,'Opus ').replace(/sonnet-/g,'Sonnet ').replace(/haiku-/g,'Haiku ').replace(/-/g,' ').replace(/\bmed\b/g,'medium');
}
function judgeName(value){
 const name=typeof value==='string'?value:(value?.variant||value?.key||value?.judge_id||''),key=name.replace(/^(extension|followup):/,''),entry=(D.judge_catalog||[]).find(e=>e.key===name)|| (D.judge_catalog||[]).find(e=>e.variant===key),meta=entry?.variant_configuration||(D.variants||[]).find(v=>v.variant===key)||{},model=meta.model||entry?.member_calls?.[0]?.model;
 const named={panel_low:'Four-model panel · low reasoning · average votes',panel_low_strict:'Four-model panel · low reasoning · majority vote',panel_high:'Four-model panel · high reasoning · average votes',panel_high_strict:'Four-model panel · high reasoning · majority vote',enhanced_luna_high_balanced:'GPT 5.6 Luna · high reasoning · both plan orders averaged',enhanced_luna_high_balanced_strict:'GPT 5.6 Luna · high reasoning · both orders must agree',or_panel_nine_high:'Nine-model panel · equal model weights',or_panel_family_balanced_high:'Nine-model panel · equal weights for six families',or_panel_eight_available_high:'Eight-model panel · equal model weights',or_panel_five_families_available_high:'Eight-model panel · equal weights for five families',judge_1:'Original single judge',judge_2:'Original judge · both plan orders',judge_3:'Original mixed-model panel',judge_4:'Original judge · five repeated comparisons'};
 if(named[key])return named[key];
 if(entry?.kind==='derived')return (entry.display_label||entry.label||key.replaceAll('_',' '));
 if(model){let label=modelName(model),effort=meta.reasoning_effort||entry?.member_calls?.[0]?.reasoning_effort;if(effort)label+=' · '+effort+' reasoning';if(key==='legacy_luna_low')label+=' · signed-score prompt';else if(key==='absolute_luna_high')label+=' · separate plan ratings';else if(key==='terra_high_one_example')label+=' · one human example';else if(key.startsWith('enhanced_'))label+=' · restored context';return label;}
 return (typeof value==='object'?value.label:null)||entry?.label||key.replaceAll('_',' ')||'AI judge';
}
const coverageText=j=>pct(j?.coverage?.criterion_coverage)+(valid(j?.coverage?.scored_criteria)&&valid(j?.coverage?.requested_criteria)?'<span class="sub">'+integer(j.coverage.scored_criteria)+' / '+integer(j.coverage.requested_criteria)+' answered</span>':'');
const extensionRoundLabel=x=>({provider_validation:'Experiment 4 · Compare provider models and panels',example_validation:'Experiment 5 · Add one human comparison example',focused_validation:'Experiment 6 · Compare smaller panels'}[x]||x.replaceAll('_',' '));
const criterionLabel=x=>{if(x==='overall')return 'Overall';const m=x.match(/q(\d+)/);return ({'01':'Concept relevance','02':'Grouping','03':'Priority','04':'Time allocation','05':'Practice alignment','06':'Difficulty','07':'Answer accuracy','08':'Strategies / formats'}[m?.[1]]||x.replaceAll('_',' '));};
function overview(){
 const splits=D.manifest.split_subject_counts||{}, test=D.rounds.find(r=>r.name==='test');
 $('study-meta').textContent=(human?.coverage?.models?.length??12)+' lesson-plan generator systems · students held out for the original test';
 $('study-status').textContent=test&&test.complete!==false?'Test results available':D.rounds.some(r=>r.name.startsWith('round'))?'Validation results available':'Study in progress';
 $('overview-stats').innerHTML=stat(integer(splits.development),'Development students','Previously inspected comparisons')+stat(integer(splits.validation),'Validation students','Judge design and selection')+stat(integer(splits.test),'Held-out test students','All plans stay within a student split')+stat(integer(D.manifest.pair_counts?.test?.human_reviewed),'Human-reviewed test pairs','Several comparisons can share one student');
 const note=D.final.headline||D.final.summary||D.final.conclusion;
 $('selection-note').innerHTML=(typeof note==='string'?'<div class="notice good">'+esc(note)+'</div>':'');
 selectionCard();
 $('generated-at').textContent=D.generated_at.replace('T',' ').replace(/\.\d+\+00:00/,' UTC');
}
function readingGuide(){
 $('reading-guide').innerHTML='<div class="card"><h3>What does success mean?</h3><p>Human and AI verdicts need not agree on every lesson-plan comparison. We care primarily about whether they produce the same order of the 12 generator systems. A generator system includes the model and the settings used to write its plans.</p><p class="small">A high rank correlation can coexist with disagreement about the best system. It also does not establish that either ranking predicts student learning gains.</p></div><div class="card"><h3>What did the judges see?</h3><p><strong>The original judges received no human examples.</strong> Each request contained the student’s pretest, two anonymous lesson plans, and the quality rubric. Human lesson-plan verdicts were used for evaluation, outside these prompts.</p><p class="small">The original four-model panels combined GPT 5.6 Luna, Terra, Sol, and GPT 6 Astra. All four used either low or high reasoning. Averaging and majority-vote panels reused the same saved member answers.</p><a href="#judge-systems">Inspect the exact judge prompts and settings</a></div>'+extensionHeadline();
 const shared=D.human_comparator?.summary?.kendall_tau_b;if(shared)$('reading-guide').insertAdjacentHTML('beforeend','<p class="small" style="grid-column:1/-1"><strong>Can AI agree with humans more than humans agree with each other?</strong> In a comparison using the same questions and the same independent human reference, Terra has mean Kendall τ-b '+num(shared.ai_vs_reference.mean)+' versus '+num(shared.human_vs_reference.mean)+' for another human group. The difference varies across reviewer splits, including reversals; this does not prove general AI superiority. <a href="#human-comparator">See the matched comparison and its limits</a>.</p>');
}

function studyDesign(){
 const m=D.manifest,rows=Object.entries(m.split_subject_counts||{}).map(([key,n])=>[esc({development:'Development',validation:'Validation',test:'Held-out test'}[key]||key),integer(n),integer(m.pair_counts?.[key]?.human_reviewed),integer(m.pair_counts?.[key]?.available)]);
 $('study-design-content').innerHTML='<div class="card"><h3>Students are the split unit; plan pairs are the comparison unit</h3><p>All plans for one student stay in the same split. One student can have several plans and therefore several possible comparisons. For example, three plans allow three pairs: A–B, A–C, and B–C. Counting extra pairs does not create extra independent students.</p>'+table(['Student split','Students','Pairs reviewed by humans','All available plan pairs'],rows)+'<p class="caption">Counts describe the stored pair catalog, before an experiment selects its schedule. Development contains the previously inspected examples. Validation supports judge choices. Test students and plans were excluded from those choices until one judge was locked; reviewers and pretest questions can occur in more than one split.</p><p>Round 1 samples 60 reviewed pairs from 60 validation students. Round 2 uses 120 reviewed pairs from all 82 validation students. The held-out comparison uses all 155 reviewed pairs from 83 test students; dense AI judging expands this to 293 available pairs, including 138 pairs without human reviews.</p></div><div class="card" style="margin-top:18px"><h3>From pairwise verdicts to a generator leaderboard</h3><p>For each quality question, a preference for a plan counts as a win, a tie as half a win, and a preference for its opponent as a loss. Multiple human reviews are averaged before quality families are combined. The overall analysis gives each plan pair equal total weight and fits a Bradley–Terry model, which accounts for which opponents each generator faced.</p><p><strong>Twelve generator systems imply 66 distinct generator matchups:</strong> 12 × 11 ÷ 2. The 155 or 293 lesson-plan comparisons repeat some of these matchups on different students. They are not 155 or 293 independently ranked generators. Averaged across all 66 possible matchups, 155/66 is 2.35 plan comparisons and 293/66 is 4.44. The reviewed test covers 64 of the 66 matchups, and the dense schedule covers 65; their comparison graphs still connect all 12 systems.</p><p class="caption">The schedules are uneven: among observed generator matchups, the reviewed test has 1–5 plan comparisons per matchup (median 2), while the dense schedule has 1–14 (median 4). Dividing 155 or 293 by 66 hides missing and heavily sampled matchups. The fitted leaderboard can order systems linked through shared opponents, although direct evidence for each matchup differs.</p><details><summary>How to read the alignment metrics</summary><p><strong>Kendall τ-b</strong> measures whether pairs of generators have the same order, with a correction for tied ranks. It runs from −1 for reversed rankings to 1 for identical rankings; values near 0 indicate little ordering association. <strong>Spearman ρ</strong> is the correlation between the two lists of rank positions. It has the same −1 to 1 range and penalizes larger rank displacements more strongly.</p><p><strong>Model-order agreement</strong> is the share of the 66 generator matchups whose fitted order agrees. <strong>Rank MAE</strong> is the average absolute difference in rank position: 1.0 means a generator is one position away on average. Lower is better. These metrics compare fitted generator rankings, rather than individual lesson-plan verdicts.</p><p><strong>Answer coverage</strong> is the number of requested rubric comparisons that received a usable AI answer divided by the number requested. A tie is an answer; insufficient evidence or a missing request is not. Panels can lose coverage when required members do not answer or a strict vote does not resolve the question.</p></details></div>';
}
function extensionHeadline(){
 const extension=D.extension;if(!extension)return '';let body='<strong>New extension · exploratory validation.</strong> Provider and human-example comparisons reuse the same 120 validation pairs after the original test results were exposed. ';
 const candidates=(extension.rounds||[]).flatMap(round=>(round.judges||[]).filter(j=>j.variant.startsWith('or_')&&catalogEntry('extension:'+j.variant)?.kind==='single'&&(j.n_calls>=round.n_pairs||j.execution_status==='completed_with_terminal_failures')&&valid(j.metrics?.kendall_tau_b)&&catalogEntry('extension:'+j.variant)?.availability?.status!=='unavailable').map(j=>({judge:j,round}))).sort((a,b)=>b.judge.metrics.kendall_tau_b-a.judge.metrics.kendall_tau_b);
 if(candidates.length){const best=candidates[0],ties=candidates.filter(c=>Math.abs(c.judge.metrics.kendall_tau_b-best.judge.metrics.kendall_tau_b)<1e-12),control=best.round.judges.find(j=>j.variant==='enhanced_terra_high');body+='<strong>'+esc(ties.map(c=>judgeName(c.judge)).join(' and '))+'</strong> '+(ties.length===1?'has':'share')+' the highest point Kendall τ-b ('+num(best.judge.metrics.kendall_tau_b)+') among '+integer(candidates.length)+' completed new single-provider candidates'+(valid(control?.metrics?.kendall_tau_b)?'; cached Terra-high is '+num(control.metrics.kendall_tau_b):'')+'. ';const paired=extension.paired_deltas?.comparisons?.[best.judge.variant]?.metrics?.kendall_tau_b;if(valid(paired?.delta))body+='The paired change versus cached Terra'+(ties.length>1?' for '+esc(judgeName(best.judge)):'')+' is Δ '+(paired.delta>0?'+':'')+num(paired.delta)+' (95% range '+interval(paired,3)+')'+(paired.low<=0&&paired.high>=0?', including no change':'')+'. ';}
 if(!candidates.length)body+=(extension.rounds||[]).some(r=>r.name==='provider_validation')?'No completed connected provider estimate is available. ':'Provider-model analyses are pending. ';
 if(extension.summary?.counts?.terminal_failed_calls>0)body+=integer(extension.summary.counts.terminal_failed_calls)+' scheduled requests remain missing after retries; compare coverage in the extension details. ';
 const primary=Object.values(extension.paired_deltas?.comparisons||{}).find(c=>c.role==='primary_example'),delta=primary?.metrics?.kendall_tau_b;if(valid(delta?.delta)){body+='Adding one human example '+(delta.delta>0?'increases':delta.delta<0?'decreases':'does not change')+' Terra’s point τ-b'+(delta.delta===0?'':(' by '+num(Math.abs(delta.delta))))+' (candidate minus control: '+num(delta.delta)+'; 95% paired range '+interval(delta)+'). ';if(delta.low<=0&&delta.high>=0)body+='The paired range includes no change; a ranking benefit is not demonstrated. ';}
 else{const exampleRound=(extension.rounds||[]).find(r=>r.judges?.some(j=>j.variant==='terra_high_one_example')),example=exampleRound?.judges.find(j=>j.variant==='terra_high_one_example'),baseline=exampleRound?.judges.find(j=>j.variant==='enhanced_terra_high');if(example?.n_calls>=exampleRound?.n_pairs&&valid(example?.metrics?.kendall_tau_b))body+='The one-example Terra variant has point τ-b '+num(example.metrics.kendall_tau_b)+(valid(baseline?.metrics?.kendall_tau_b)?' versus '+num(baseline.metrics.kendall_tau_b)+' for cached zero-shot Terra':'')+'; paired uncertainty is pending. ';}
 body+='These comparisons do not replace the original frozen selection or provide a new held-out test. <a href="#extension">See provider, example, and panel results</a>.';return '<div class="notice" id="extension-overview" style="grid-column:1/-1;margin:0">'+body+'</div>';
}
let activeSystemTab='recipe';
const catalogEntry=key=>(D.judge_catalog||[]).find(entry=>entry.key===key);
const systemMembers=entry=>entry.kind==='derived'?[...new Set((entry.member_calls||[]).map(m=>m.catalog_key))].map(catalogEntry).filter(Boolean):[entry];
function exampleStatus(entry){const e=entry?.examples||{};return e.status==='inherited'?'Inherited from member calls':e.status==='none'?'Zero-shot · 0 human examples':e.status==='human_graded'?integer(e.n_examples)+' human-graded example'+(e.n_examples===1?'':'s'):'Example usage not recorded';}
function effortLabel(value){return value||'Not set in config; provider default';}
function promptBlock(title,text,filename){if(typeof text!=='string')return '<h4>'+esc(title)+'</h4>'+empty('Exact saved prompt not available.');return '<h4>'+esc(title)+'</h4><p class="small"><a download="'+esc(filename)+'" href="data:text/plain;charset=utf-8,'+encodeURIComponent(text)+'">Download text</a></p><pre>'+esc(text)+'</pre>';}
function aggregationDescription(entry){
 if(entry.kind==='derived'&&entry.aggregation!=='strict'&&new Set((entry.member_calls||[]).map(m=>m.normalized_weight)).size>1)return 'For each rubric question, give a member 1 for A, 0.5 for a tie, and 0 for B. Multiply these values by the shares in the table and add them. In a family-balanced recipe, every family has the same total share, divided among its members. The shares are fixed before outcomes are read. If any required member cannot answer a question, the panel abstains on that question.';
 if(entry.kind==='derived')return entry.aggregation==='strict'?'Strict majority of member A / tie / B directions. A four-member panel needs at least three matching directions; a 2–2 split abstains. Missing members remain in the majority denominator.':'For each rubric question, give every member 1 for A, 0.5 for a tie, and 0 for B, then average those values. For example, equal numbers of A and B votes average to 0.5 even though the judges disagree. The resulting rubric values feed the same leaderboard calculation as a single judge. If any required member cannot answer a question, the panel abstains on that question.';
 if(entry.kind==='legacy'&&(entry.member_calls||[]).length>1)return 'Legacy strict majority over A / tie / B directions, with abstentions included in the denominator. No strict majority produces an abstention; preference strength uses a conservative median and is ignored by the model-ranking estimator.';
 return 'One call per pair. The code restores the common A/B order and converts each answer to a win, half-win for a tie, or loss. Available rubric questions then contribute equally to the pair comparison used to rank generator systems. The primary ranking ignores whether a preference was slight or strong.';
}
function setupSystems(){
 const entries=D.judge_catalog||[],groups=[...new Set(entries.map(e=>e.group))];$('system-select').innerHTML=groups.map(group=>'<optgroup label="'+esc(group)+'">'+entries.filter(e=>e.group===group).map(e=>'<option value="'+esc(e.key)+'">'+esc(judgeName(e))+'</option>').join('')+'</optgroup>').join('')||'<option>No saved judge configuration</option>';
 const chosen=D.selection?.variant||D.selection;if(entries.some(e=>e.key===chosen))$('system-select').value=chosen;
 $('system-select').addEventListener('change',renderSystem);$('system-tabs').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{activeSystemTab=button.dataset.systemTab;renderSystem();}));renderSystem();
}
function renderSystem(){
 const entry=catalogEntry($('system-select').value);if(!entry){$('system-content').innerHTML=empty('Judge snapshots are not available yet.');return}const members=systemMembers(entry);
 $('system-tabs').querySelectorAll('button').forEach(b=>{const active=b.dataset.systemTab===activeSystemTab;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));});$('system-version').textContent='Exact version and request settings are available below';
 let content='<h3>'+esc(judgeName(entry))+'</h3><p class="small">'+esc(entry.group)+' · '+esc(entry.kind==='derived'?'Offline aggregation; no new model prompt':exampleStatus(entry))+'</p>'+(entry.availability?.status==='unavailable'?'<div class="notice"><strong>Unavailable, not a quality result.</strong> '+esc(entry.availability.reason||'No completed eligible request.')+'</div>':'')+(entry.availability?.terminal_failed_calls>0?'<div class="notice">'+integer(entry.availability.terminal_failed_calls)+' scheduled requests remain missing after retries. This is a request-level execution outcome, distinct from endpoint unavailability. No replacement rating is invented; inspect the analysis coverage.</div>':'');
 if(activeSystemTab==='recipe'){
  const calls=entry.member_calls||[];content+=table(['Member judge','Model','Reasoning requested','Plan order','Model family','Vote share'],calls.map(m=>[esc(judgeName(m.variant||m.name||'member')),esc(modelName(m.model)),esc(effortLabel(m.reasoning_effort)),valid(m.draw)?(m.draw===0?'Reproducibly shuffled order':'Plans swapped from first call'):esc(m.order||'As scheduled'),esc(m.family||'—'),pct(m.normalized_weight)]))+'<p class="caption">'+esc(aggregationDescription(entry))+'</p>';
  if(entry.kind==='legacy')content+='<p class="small">Legacy presentation orders above come from one saved pair. In the five-repeat judge, seeded random orders vary by pair; those displayed orders are not a fixed AB/BA recipe.</p>';
  if(entry.kind==='derived')content+='<details><summary>Exact configuration, identifiers, and weights</summary>'+brief({aggregation:entry.aggregation,definition:entry.derived_definition,members:entry.member_calls,definition_sha256:entry.definition_sha256,definition_file_sha256:entry.definition_file_sha256})+'</details>';
  for(const m of members){content+='<details><summary>'+esc(judgeName(m.variant))+' · frozen configuration and request provenance</summary>'+brief({judge_id:m.judge_id,version:m.version,requested:m.variant_configuration,returned_model_example:m.returned_model,returned_provider_example:m.response_provider,transport:m.transport,requested_response_format:m.requested_response_format,effective_response_format:m.effective_response_format,cost_provenance:m.cost_provenance,reported_cost_usage:m.cost_usage,plan_projection:m.plan_projection,context:m.pretest_context,examples:m.examples,exact_schema_source:m.schema_source,prompt_snapshot_matches:m.prompt_snapshot_matches})+(m.configuration?.config_text?'<pre>'+esc(m.configuration.config_text)+'</pre>':brief(m.configuration||{}))+'</details>';}
 }else if(activeSystemTab==='prompts'){
  content+='<p class="small">These are exact frozen system prompts and user templates, using the saved request snapshot when present. <code>$payload</code> is replaced by the evaluated student pretest, rubric, and masked A/B plan content. Source author identities and evaluation human verdicts are not part of that task payload. Derived systems reuse each member prompt shown below.</p>';
  for(const m of members)content+='<details open><summary>'+esc(judgeName(m.variant))+' · '+esc(responseName(m.response_style))+'</summary>'+promptBlock('System prompt',m.system_prompt,m.variant+'-system.txt')+promptBlock('User prompt template',m.user_template,m.variant+'-user.txt')+'</details>';
 }else if(activeSystemTab==='schema'){
  content+='<p class="small">These are the schemas recorded for representative saved requests and used to validate responses locally. Requested provider mode and actual raw-response conformance are separate facts. A versioned adapter may remove fences or losslessly reshape a wrapper; the canonical response must still pass the strict semantic parser, without repairing verdicts, criterion IDs, or strengths. Each request has the exact criterion IDs of that plan pair; those IDs vary across pairs. Provider schema enforcement depends on the effective response format, shown when the transport record is available.</p>';
  for(const m of members){const meaning=m.response_style==='evidence_winner'?'Evidence for each plan, a short explanation, explicit A/B/tie/insufficient winner, and preference strength. Code converts the winner to a signed score.':m.response_style==='absolute_ratings'?'Independent 1–5 ratings for A and B, evidence, and explanation. Code differences the ratings; the primary ranking uses only direction.':'Signed preference: +1/+2 favor displayed A, −1/−2 favor displayed B, 0 is an explicit tie, null is insufficient evidence.';content+='<details open><summary>'+esc(judgeName(m.variant))+'</summary><p>'+esc(meaning)+'</p>'+(m.transport?'<p class="small">Requested response format: '+esc(m.requested_response_format||m.transport.requested?.response_format||'recorded below')+'. Effective response format: '+esc(m.effective_response_format||m.transport.effective?.response_format||'recorded below')+'.</p>'+brief({requested:m.transport.requested,effective:m.transport.effective,compatibility_fallbacks:m.transport.compatibility_fallbacks}):'<p class="small">Transport-level format metadata is not available for this historical or pending call; provider enforcement is not inferred.</p>')+(m.parser_contract?'<h4>Recorded formatting/parser contract</h4>'+brief({parser_contract:m.parser_contract,parser_version:m.parser_version,parser_contract_sha256:m.parser_contract_sha256,raw_response_matched_schema:m.provider_schema_compliant,transformations:m.transformations,variant_counts:m.serialization_summary}):'')+(m.schema_source?.startsWith('calls/')?'<p class="small"><a href="'+esc((m.key.startsWith('followup:')?D.followup.directory+'/':m.key.startsWith('extension:')?D.extension.directory+'/':'./')+m.schema_source)+'">Inspect representative saved call, including original raw output</a></p>':'')+(m.output_schema?brief(m.output_schema):empty('No completed saved request schema is available yet.'))+'</details>';}
 }else{
  for(const m of members){const e=m.examples||{};content+='<h4>'+esc(judgeName(m.variant))+' · '+esc(exampleStatus(m))+'</h4><p class="small">'+(e.status==='none'?'No human-graded example was supplied. Human verdicts were used outside the prompt for evaluation and validation-based system selection.':esc(e.selection_note||e.note||'Human example details are recorded below.'))+'</p>'+table(['Input component','Status'],[['Example source split',esc(e.source_split||'Not applicable')],['Pretest context',m.pretest_context?'Restored text from the pinned application supplement':'Original exported text'],['Plan projection',m.plan_projection==='human_display'?'Human display: hidden answer_original fields removed':'Stored plan content'],['Author identities','Masked from judge task payload'],['Evaluation human verdicts','Excluded from evaluated task payload']])+((e.status!=='none')?brief(e):'');}
 }
 $('system-content').innerHTML=content;
}
function renderExtensionDeltas(){
 const data=D.extension?.paired_deltas;if(!data){$('extension-deltas').innerHTML='';return}const comparisons=Object.entries(data.comparisons||{}),primary=comparisons.find(([,c])=>c.role==='primary_example');let body='';
 if(primary){const [name,c]=primary,rows=['kendall_tau_b','spearman','rank_mae'].map(key=>{const m=c.metrics?.[key]||{};return[esc({kendall_tau_b:'Kendall τ-b',spearman:'Spearman ρ',rank_mae:'Rank MAE'}[key]),num(m.baseline),num(m.candidate),num(m.delta),interval(m),integer(m.n_valid)+' / '+integer(data.n_bootstrap),m.higher_is_better?'Positive delta improves':'Negative delta improves'];});body+='<div class="card" style="margin-bottom:18px"><h3>One human example: paired change from the cached Terra control</h3><p class="small">Candidate: '+esc(judgeName(name))+'. Baseline: '+esc(judgeName(c.baseline_variant||data.baseline_variant))+'. Delta is candidate minus baseline, using identical student-resampling draws for both fits.</p>'+table(['Metric','Zero-shot','One example','Point delta','95% paired range','Valid draws','Direction'],rows)+'<p class="caption">The 95% interval is the 2.5th and 97.5th percentiles of repeated student resamples, rather than the point estimate plus or minus a symmetric error. Rankings change in discrete steps, so the interval can be asymmetric and need not be centered on the observed change. The estimate and both endpoints use three decimal places. This uncertainty resamples the '+integer(data.n_subjects)+' validation student contexts, conditional on this one chosen example and these fixed generated answers. It does not resample example choice, new model generations, or provider/backend changes. This is exploratory validation after test exposure, not a general causal few-shot benefit. Disconnected draws: '+integer(c.n_disconnected_draws)+'; nonconverged draws: '+integer(c.n_nonconverged_draws)+'.</p>';
  const exampleRound=(D.extension.rounds||[]).find(r=>r.judges?.some(j=>j.variant===name)),candidate=exampleRound?.judges.find(j=>j.variant===name),baseline=exampleRound?.judges.find(j=>j.variant===(c.baseline_variant||data.baseline_variant));if(valid(candidate?.metrics?.probability_mae)&&valid(baseline?.metrics?.probability_mae))body+='<p class="caption" id="example-probability-note"><strong>A different metric:</strong> fitted win-probability MAE changes from '+num(baseline.metrics.probability_mae,5)+' to '+num(candidate.metrics.probability_mae,5)+' (lower is better). This measures differences between AI- and human-fitted model win probabilities; it is not verdict accuracy or agreement on rank order. Probability error and rank order can move in different directions.</p>';body+='</div>';}
 const other=comparisons.filter(([,c])=>c.role!=='primary_example');if(other.length)body+='<details><summary>Exploratory paired provider/panel deltas versus cached Terra</summary><p class="small">All deltas use the same baseline and resampled student contexts. The human reference retains the full scheduled cohort while missing AI outputs remain missing. Differences therefore describe the executed judging pipeline and can reflect which comparisons have usable AI answers. The matched-scored check below separately changes the reference to shared answers. Multiple candidate comparisons and repeated validation use remain exploratory; a positive point delta alone does not establish a superior judge.</p>'+table(['Candidate','Δ Kendall τ-b','95% paired range','Valid draws'],other.map(([name,c])=>[esc(judgeName(name)),num(c.metrics?.kendall_tau_b?.delta),interval(c.metrics?.kendall_tau_b),integer(c.metrics?.kendall_tau_b?.n_valid)+' / '+integer(data.n_bootstrap)]))+'</details>';
 $('extension-deltas').innerHTML=body+'<p class="small"><a href="'+esc(D.extension.directory+'/paired_deltas.json')+'">Download paired-delta analysis</a></p>';
}
function renderExampleAudit(){
 const audit=D.extension?.example_audit;if(!audit){$('extension-example-audit').innerHTML='';return}const s=audit.summary||{},sample=audit.sample||{},copy=audit.copying_check||{};
 const counts=[['Exact scores unchanged',s.same],['Only preference strength changes',s.strength_only],['Preference becomes a tie',s.preference_to_tie],['Tie becomes a preference',s.tie_to_preference],['Preference reverses A ↔ B',s.opposite_preference]];
 let body='<details><summary>Illustrative explanation audit · '+integer(sample.n_pairs)+' validation pairs</summary><p>'+esc(audit.interpretation)+'</p><p class="small">The first '+integer(sample.n_pairs)+' scheduled pairs were fixed before examining disagreement, with '+integer(sample.n_criteria)+' pair/criterion answers. '+integer(s.unchanged_directions)+' directions remain unchanged; ties change from '+integer(s.baseline_ties)+' to '+integer(s.example_ties)+'. These sample counts do not establish more tie use or better explanations in general. Changing strength alone has no effect on the primary direction-based ranking.</p>'+table(['Change within this sample','Pair/criterion answers'],counts.map(([label,n])=>[esc(label),integer(n)]));
 for(const finding of audit.findings||[])body+='<details><summary>Pair '+integer(finding.pair_number)+' · '+esc(finding.criterion)+' · '+esc(finding.title)+'</summary><p>'+esc(finding.observation)+'</p>'+finding.snippets.map(snippet=>'<p class="small">'+esc(snippet.source==='one_example'?'With one human example':'Cached zero-shot control')+' · '+esc(snippet.field)+'</p><blockquote>'+esc(snippet.text)+'</blockquote>').join('')+'</details>';
 body+='<p><strong>Historical-example copying check.</strong> '+esc(copy.result)+' '+esc(copy.limitations)+'</p><p class="small"><a href="'+esc(D.extension.directory+'/example_audit.json')+'">Full audit, methods, and local provenance</a></p></details>';$('extension-example-audit').innerHTML=body;
}
function extensionCost(j){
 const cost=j.cost_breakdown||D.extension?.summary?.costs_by_variant?.[j.variant];const unknown=cost?.complete===false||cost?.n_unknown>0,value=unknown?null:(j.cost_breakdown?(cost.economic_usd??cost.known_economic_usd??j.estimated_usd):(j.estimated_usd??cost?.economic_usd));
 return (valid(value)?'$'+num(value,2):'—')+((cost?.is_byok||cost?.byok_upstream_inference_usd>0)?'<span class="sub">Includes BYOK upstream inference</span>':'');
}
function terminalCount(j,prefix='extension:'){return j.terminal_failed_calls??catalogEntry(prefix+j.variant)?.availability?.terminal_failed_calls??0;}
function extensionExecution(j,prefix='extension:'){
 const available=catalogEntry(prefix+j.variant)?.availability,terminal=terminalCount(j,prefix);
 if(available?.status==='unavailable'||j.execution_status==='unavailable'||j.unavailable_member_variants?.length)return 'Endpoint unavailable; no quality estimate';
 if(terminal>0)return (j.n_calls>0?'Finished with ':'No successful requests; ')+integer(terminal)+' missing after retries';
 return j.execution_status==='completed'||(j.n_calls>0&&j.coverage?.missing_calls===0)?'Completed':j.execution_status==='incomplete'?'In progress; see coverage':'See coverage';
}
function contractCounts(value){return Array.isArray(value)?value.join(', '):value&&typeof value==='object'?Object.entries(value).map(([contract,n])=>contract+': '+integer(n)).join('; '):String(value||'—');}
function interruptionSummary(){
 const summary=D.extension?.summary||{},interruption=summary.execution_interruptions||summary.costs?.interrupted_in_flight_requests||{},events=(interruption.events||[]).filter(event=>event.signal_sent===true);if(!events.length)return '';
 const unknown=interruption.n_unknown_events??events.filter(event=>event.in_flight?.charges_usd==null).length;
 let body='<div class="notice" id="extension-interruption-note"><strong>Interrupted-batch billing.</strong> '+integer(events.length)+' recorded execution interruption'+(events.length===1?'':'s')+'. '+(unknown>0?'Billing remains unresolved for '+integer(unknown)+' interruption'+(unknown===1?'':'s')+'. Recorded or estimated amounts shown in this report exclude any unreconciled charges and are not a complete invoice total. ':'The interruption ledger records reconciled billing; consult its saved evidence. ')+'<details><summary>Actual interruptions and billing limits</summary><p>Only records with a sent signal are counted. Rejected proposals that sent no signal are excluded. A missing result does not establish that an API request was dispatched.</p>'+table(['Signal sent at','Affected models','Confirmed interrupted API requests','Simultaneous local operations','Billing'],events.map(event=>{const flight=event.in_flight||{};return[esc(event.signal_sent_at||'recorded without time'),esc((event.model_ids||event.variant_ids||[]).map(modelName).join(', ')),valid(flight.confirmed_interrupted_api_request_count)?integer(flight.confirmed_interrupted_api_request_count):'Unknown',valid(flight.local_api_calls_upper_bound)?integer(flight.local_api_calls_lower_bound??0)+'–'+integer(flight.local_api_calls_upper_bound):'Unknown',esc(flight.billing_reconciliation_status||'Unknown')];}))+'<p>The concurrency bounds limit simultaneous local operations. They do not count confirmed dispatches, queued jobs, or all upstream work that may continue after a local process stops.</p>'+(interruption.file?'<p class="small"><a href="'+esc(D.extension.directory+'/'+interruption.file)+'">Execution interruption ledger</a> · SHA-256 '+esc(interruption.sha256||'not recorded')+'</p>':'')+'</details></div>';return body;
}
function extensionMatched(round){
 const judges=(round.judges||[]).filter(j=>j.matched?.metrics);if(!judges.length)return '';
 return '<details><summary>Coverage check: compare only answers available from humans and this judge</summary><p>The primary result keeps the human answers for the full schedule. This diagnostic refits the human reference using only the pair/criterion answers also scored by the AI. For example, if the AI leaves two difficult accuracy questions unanswered, this check also removes those questions from the human ranking. Agreement can then change even if every AI answer stays the same. The full-schedule result remains the primary comparison.</p>'+table(['Judge','Full-reference τ-b','Matched-scored τ-b','Matched human pairs','Matched AI pairs'],judges.map(j=>[esc(judgeName(j)),num(j.metrics?.kendall_tau_b),num(j.matched.metrics.kendall_tau_b),integer(j.matched.human?.n_comparisons),integer(j.matched.judge?.n_comparisons)]))+'</details>';
}
function terminalSummary(){
 const summary=D.extension?.summary||{},terminal=summary.terminal_failures||{},count=summary.counts?.terminal_failed_calls??terminal.n_jobs??0;if(!count)return '';
 const rows=Object.entries(terminal.by_variant||{}).map(([variant,n])=>[esc(judgeName(variant)),integer(n)]),details=terminal.details||[];
 return '<div class="notice" id="extension-terminal-note"><strong>'+integer(count)+' scheduled requests remain missing after retries.</strong> The execution ledger records at least three verified failed API attempts per declared request. These are missing outputs, not ties, judge abstentions, or invented ratings. This differs from the Muse endpoint being blocked by account policy. The original comparison schedule and human answers for the full schedule remain intact; successful calls can still support a model ranking where the graph is connected.<details><summary>Missing requests and verified execution evidence</summary>'+table(['Variant','Requests missing after retries'],rows)+(details.length?table(['Request','Variant','Reason','Verified failed attempts'],details.map((row,i)=>['Request '+integer(i+1),esc(judgeName(row.variant)),esc(row.reason),integer(row.n_failed_api_attempts??row.verified_attempts?.length)])):'')+(terminal.file?'<p class="small"><a href="'+esc(D.extension.directory+'/'+terminal.file)+'">Local terminal-failure ledger</a> · SHA-256 '+esc(terminal.sha256||'not recorded')+'</p>':'')+'</details></div>';
}

function extensionLeaderboard(round){
 const judges=round.judges||[];if(!judges.length)return;
 $('extension-results').insertAdjacentHTML('beforeend','<div class="card" style="margin-top:18px"><h3>Extension model leaderboard · validation only</h3><div class="toolbar"><label class="control">Extension judge<select id="extension-judge-select">'+judges.map((j,i)=>'<option value="'+i+'">'+esc(judgeName(j))+'</option>').join('')+'</select></label></div><div id="extension-ranking-warning"></div><div id="extension-rank-chart" class="chart"></div><div id="extension-rank-table" class="scroll"></div><p class="caption">The ranked entities are the 12 lesson-plan generator systems. The human reference uses this extension’s fixed 120-pair validation schedule and exact rubric criteria. Bootstrap rank intervals, when supplied, resample student contexts; they do not establish a unique winner or account for new model generations.</p></div>');
 const first=judges.findIndex(j=>j.variant.startsWith('or_')&&j.fit?.connected);if(first>=0)$('extension-judge-select').value=first;
 function draw(){const j=judges[+$('extension-judge-select').value],availability=catalogEntry('extension:'+j.variant)?.availability;if(availability?.status==='unavailable'||j.execution_status==='unavailable'||j.unavailable_member_variants?.length||!j.fit?.connected){$('extension-ranking-warning').innerHTML=empty(availability?.reason||(j.unavailable_member_variants?.length?'Required endpoint unavailable: '+j.unavailable_member_variants.map(judgeName).join(', '):j.fit?.warning)||'A connected global leaderboard is not available for this judge.');$('extension-rank-chart').innerHTML='';$('extension-rank-table').innerHTML='';return}const humanRows=fitRows(round.human),judgeRows=fitRows(j.fit),byModel=new Map(judgeRows.map(row=>[row.model,row])),boot=j.bootstrap;
 $('extension-ranking-warning').innerHTML='<p class="small">Kendall τ-b '+num(j.metrics?.kendall_tau_b)+' · Spearman ρ '+num(j.metrics?.spearman)+' · model-order agreement '+pct(j.metrics?.pair_order_agreement)+' across '+integer(j.metrics?.n_model_pairs)+' generator-system pairs. Answer coverage '+pct(j.coverage?.criterion_coverage)+'. Missing scheduled calls: '+integer(j.coverage?.missing_calls??0)+'.</p>'+(terminalCount(j)>0?'<div class="notice">'+integer(terminalCount(j))+' requests remain missing after retries. The human reference retains the fixed '+integer(round.n_pairs)+'-pair schedule and exact criteria; the AI fit uses its remaining scored outputs. Missing outputs are not scored as ties.</div>':'')+(j.bootstrap_cohort_warning?'<div class="notice">'+esc(j.bootstrap_cohort_warning)+'</div>':'');
 $('extension-rank-chart').innerHTML=rankChart(humanRows,judgeRows,'Validation human reference',judgeName(j));
 $('extension-rank-table').innerHTML=table(['Lesson-plan generator system','Human rank','Judge rank','Rank difference'],humanRows.map(h=>{const a=byModel.get(h.model),hi=boot?.rank_intervals?.human?.[h.model],ai=boot?.rank_intervals?.judge?.[h.model];return[esc(modelName(h.model)),num(h.rank,1)+(hi?'<span class="sub">'+interval(hi,1)+'</span>':''),num(a?.rank,1)+(ai?'<span class="sub">'+interval(ai,1)+'</span>':''),valid(a?.rank)&&valid(h.rank)?num(a.rank-h.rank,1):'—'];}));}
 $('extension-judge-select').addEventListener('change',draw);draw();
}

function extensionResults(){
 const extension=D.extension,link=document.querySelector('nav a[href="#extension"]');$('extension').hidden=!extension;if(link)link.hidden=!extension;if(!extension)return;renderExtensionDeltas();renderExampleAudit();const meta=extension.metadata||{},rounds=extension.rounds||[];
 $('extension-protocol').textContent=meta.protocol_note||'Exploratory comparisons on validation subjects after the original test was exposed. These outcomes are not a fresh held-out test and do not replace the original frozen selection.';
 $('extension-plan').innerHTML='<div class="notice"><strong>Common comparison schedule.</strong> Provider variants and the human-example control use the same 120 validation plan pairs. The nine-provider baseline has no human-graded examples. The Terra-high example control uses one selected development example against cached zero-shot outputs on the same pairs. The example is an illustration with real human disagreement, not error-free gold. Separate generations can differ through sampling and backend variability.</div><p class="small">The panel recipes below reuse cached provider calls. Equal-model panels give every included model the same weight; family-balanced panels give every included family the same total weight. Availability-only alternatives are separate frozen recipes, not silent removal of missing members. Deriving panels adds no API calls, and their member costs must not be summed again with single-model costs. “High” effort is provider-specific and does not equate computation, reasoning tokens, latency, or cost across vendors. We chose a 32,768-token output cap as an experiment budget for the original OpenRouter comparison; it was not a provider requirement. Two GLM comparisons exhausted that cap after repeated attempts and remain missing; these omissions need not be random. The cached native controls did not request this cap. The later Qwen request omitted an explicit cap but was blocked by endpoint policy before generation.</p>';
 const availability=extension.summary?.availability||{},unavailable=Object.entries(availability).filter(([,value])=>value.status==='unavailable');if(unavailable.length)$('extension-plan').insertAdjacentHTML('beforeend','<div class="notice"><strong>Endpoint availability is not judge quality.</strong> '+unavailable.map(([name,value])=>esc(judgeName(name))+': '+esc(value.reason||'unavailable')).join(' ')+' No empirical quality score is inferred for an unavailable model or a panel that requires it.</div>');
 const recipes=(D.judge_catalog||[]).filter(e=>e.group==='Validation extension'&&e.kind==='derived');if(recipes.length)$('extension-plan').insertAdjacentHTML('beforeend','<div class="card" style="margin-bottom:18px"><h3>Panel membership and vote weights</h3>'+table(['Recipe','Members','Families','Member shares','Availability'],recipes.map(e=>[esc(judgeName(e)),integer(e.member_calls.length),integer(new Set(e.member_calls.map(m=>m.family).filter(Boolean)).size),esc(e.member_calls.map(m=>judgeName(m.variant)+': '+pct(m.normalized_weight)).join('; ')),e.availability?.status==='unavailable'?'Unavailable: required member missing':'See completed-call coverage']))+'</div>');
 const serialization=extension.summary?.serialization_by_variant||{};if(Object.keys(serialization).length)$('extension-plan').insertAdjacentHTML('beforeend','<details><summary>Provider response formatting and lossless adaptation counts</summary><p class="small">Direct conformance means the raw response matched the requested structure. Adapted responses required only a recorded serialization change and then passed the same strict semantic validation. Adaptation is not reported as direct schema compliance; no verdict, criterion ID, or strength is repaired.</p>'+table(['Judge','Completed','Direct schema match','Adapted','Recovered saved errors','Parser contracts'],Object.entries(serialization).map(([name,value])=>[esc(judgeName(name)),integer(value.completed_calls),integer(value.provider_schema_compliant_calls),integer(value.adapted_calls),integer(value.recovered_from_errors),esc(contractCounts(value.parser_contracts))]))+'</details>');
 const summary=extension.summary||{},statusLabel={complete:'Finished',running:'Running',complete_with_terminal_missing_outcomes:'Finished with requests missing after retries',complete_with_provider_unavailable:'Finished with an unavailable provider'};if(Object.keys(summary).length)$('extension-plan').insertAdjacentHTML('beforeend','<p class="small">Extension status: '+esc(statusLabel[summary.status]||summary.status||'recorded')+'. Successful calls: '+integer(summary.counts?.completed_calls)+' / '+integer(summary.counts?.scheduled_calls)+'. '+(valid(summary.counts?.missing_outcomes)?'Missing scheduled outputs, all causes: '+integer(summary.counts.missing_outcomes)+'. ':'')+(valid(summary.counts?.unavailable_calls)?'Requests without an available endpoint: '+integer(summary.counts.unavailable_calls)+'. ':'')+(valid(summary.counts?.terminal_failed_calls)?'Requests missing after retries: '+integer(summary.counts.terminal_failed_calls)+'. ':'')+(valid(summary.estimated_usd)?'Recorded/estimated subtotal: $'+num(summary.estimated_usd,2)+'. ':'')+esc(summary.cost_note||'Usage for failed attempts may be unavailable.')+'</p>'+terminalSummary());
 if(summary.costs_total){const cost=summary.costs_total,known=cost.known_economic_usd??cost.economic_usd;$('extension-plan').insertAdjacentHTML('beforeend','<p class="small"><strong>Recorded/estimated inference subtotal:</strong> '+(valid(known)?'$'+num(known,2):'not available')+'. '+(cost.complete===false||!valid(cost.economic_usd)?'Unknown charges are excluded. ':'')+'Router charge: '+(valid(cost.router_charge_usd)?'$'+num(cost.router_charge_usd,2):'not fully known')+'. BYOK upstream inference: '+(valid(cost.upstream_inference_usd)?'$'+num(cost.upstream_inference_usd,2):'not fully known')+'. A zero router charge does not mean free inference. These totals count underlying calls once; derived panel rows are cost attributions, not additional spend.</p>');}
 $('extension-plan').insertAdjacentHTML('beforeend',interruptionSummary());
 $('extension-round-select').innerHTML=rounds.map((r,i)=>'<option value="'+i+'">'+esc(extensionRoundLabel(r.name))+'</option>').join('')||'<option>Analysis pending</option>';
 function draw(){const r=rounds[+$('extension-round-select').value];if(!r){$('extension-results').innerHTML=empty('Extension analyses are pending. Judge configurations are inspectable in the Judge systems tab; no result is inferred.');return}const fixed=r.human?.n_comparisons;let body='<div class="card"><h3>'+esc(extensionRoundLabel(r.name))+'</h3><p class="small">Validation only · '+integer(r.n_pairs)+' scheduled plan pairs · '+integer(r.n_subjects)+' student contexts · '+integer(fixed)+' human reference comparisons.</p>'+(r.complete===false?'<div class="notice">This analysis retains missing scheduled outputs: '+integer(r.n_completed_calls)+' / '+integer(r.n_expected_calls)+' calls succeeded. Missing outcomes can change scored support; execution status distinguishes pending calls from requests that exhausted retries.</div>':'');body+='<div class="scroll">'+table(['Judge system','Execution status','Kendall τ-b','Spearman ρ','Model order agreement','Answer coverage','Successful calls','Missing calls','Attributed inference cost'],(r.judges||[]).map(j=>[esc(judgeName(j)),esc(extensionExecution(j)),num(j.metrics?.kendall_tau_b),num(j.metrics?.spearman),pct(j.metrics?.pair_order_agreement),coverageText(j),integer(j.n_calls),integer(j.coverage?.missing_calls??0),extensionCost(j)]))+'</div>';body+='<p class="caption">All metrics rank the same 12 generator systems against all human answers for the scheduled questions, including when some AI outputs are missing. Missing calls remain missing; they are not converted to ties or removed from the scheduled denominator. Answer coverage counts usable rubric answers divided by requested rubric comparisons (normally eight per plan pair). It falls when calls are missing or a judge cannot answer; an explicit tie still counts as an answer. Model-order agreement has 66 unordered model pairs as its denominator. This extension is exploratory after test exposure; differences across provider variants are not isolated causal effects of model architecture. Costs attribute successful saved calls; panel rows reuse member calls and are not additive. Failed or interrupted requests may have unknown charges, which these row amounts exclude. See the separate execution and accounting details below.</p>'+(r.source_file?'<p class="small"><a href="'+esc(extension.directory+'/'+r.source_file)+'">Download this analysis</a></p>':'')+'</div>';$('extension-results').innerHTML=body+extensionMatched(r);extensionLeaderboard(r);}
 $('extension-round-select').addEventListener('change',draw);draw();
}


function followupResults(){
 const follow=D.followup,section=$('followup'),link=document.querySelector('nav a[href="#followup"]');section.hidden=!follow;if(link)link.hidden=!follow;if(!follow)return;
 const entries=(D.judge_catalog||[]).filter(e=>e.key.startsWith('followup:')),panels=entries.filter(e=>e.kind==='derived'),rounds=follow.rounds||[];
 let body='<p class="experiment-intro">The broad provider comparison motivates smaller panels with one model per included family. The first trio combines Terra, DeepSeek Flash, and GLM Flash. A second trio replaces GLM Flash with Kimi. A quartet adds Qwen to the first trio. The member choices use the earlier validation results and were fixed before Qwen’s new outcomes were read. We again use the same 120 validation pairs, so this is a follow-up on already used validation data, not a new held-out test.</p><div class="card"><h3>6a. Which models are combined?</h3>'+table(['Judge panel','Members and voting shares'],panels.map(e=>[esc(judgeName(e)),e.member_calls.map(m=>esc(judgeName('followup:'+m.variant))+' ('+pct(m.normalized_weight)+')').join('<br>')]))+'<p class="caption">Each panel averages the saved member votes using these prespecified shares. Required missing member answers remain missing; they are not silently dropped or turned into ties. Qwen is the only model requested for new comparisons; the other members reuse prior calls. Attributed panel costs therefore overlap and cannot be added to estimate total spend.</p></div><div class="toolbar"><label class="control">Follow-up analysis<select id="followup-round-select">'+rounds.map((r,i)=>'<option value="'+i+'">'+esc(extensionRoundLabel(r.name))+'</option>').join('')+'</select></label></div><div id="followup-analysis"></div>';
 const unavailable=Object.entries(follow.summary?.availability||{}).filter(([,v])=>v.status==='unavailable');if(unavailable.length)body+='<div class="notice"><strong>Some requested models could not be evaluated.</strong> '+unavailable.map(([name,value])=>esc(judgeName('followup:'+name))+': '+esc(value.reason||'No eligible endpoint.')).join(' ')+' Scheduled comparisons for an unavailable endpoint remain missing; they are not 120 completed or dispatched requests. A panel requiring that member has no quality estimate.</div>';
 body+='<details id="execution-speed-note"><summary>Execution speed and our response-token budget</summary><p>The earlier provider batches used 24 workers overall and 8 concurrent calls per exact model in each process. The final DeepSeek Flash resume used 24 workers and 24 calls for that model. The runner now defaults to 128 workers and 128 calls per model, and both settings can be raised to 512. These are per-process admission limits, not a guarantee of achieved provider throughput.</p><p>The completed Flash records had a median duration of 650.1 seconds after admission, including retries and excluding time waiting in the local queue. Higher concurrency reduces queue waiting; it does not make an individual hosted response generate faster. Long responses and retries therefore remained important to elapsed time. Qwen attempted only one preflight, which was blocked before generation: no 128- or 512-concurrency Qwen bulk run occurred.</p><p>We chose the earlier 32,768-token output cap to limit the experiment budget; the provider did not require it. Two GLM comparisons exhausted that cap and remain missing. The later Qwen request omitted an explicit cap but never reached generation.</p><p class="small">See the <a href="../../docs/extension-runner.md">runner scheduling and duration documentation</a>, the <a href="'+esc(D.extension?.directory+'/execution_interruptions.json')+'">recorded execution interruptions</a>, and the <a href="'+esc(follow.directory+'/summary.json')+'">follow-up execution summary</a>. Saved records and documented limits distinguish configured capacity, admitted work, and actual completed responses.</p></details>';
 const summary=follow.summary||{},known=summary.costs_total?.known_economic_usd??summary.costs_total?.economic_usd;body+='<details><summary>Follow-up execution, costs, and exact provenance</summary><p class="small">This follow-up has its own ledger. Imported calls are reused evidence, and their attributed cost is distinct from newly incurred spend. '+(valid(known)?'Recorded/estimated subtotal: $'+num(known,2)+'. ':'')+(summary.costs_total?.complete===false?'Unknown charges are excluded. ':'')+'</p>'+brief(summary)+'</details>';
 $('followup-results').innerHTML=body;
 function draw(){const r=rounds[+$('followup-round-select').value];if(!r){$('followup-analysis').innerHTML=empty('The focused follow-up analyses are pending; the member recipes are available above.');return}const judges=r.judges||[],names=judges.map(j=>judgeName('followup:'+j.variant));let content='<div class="card"><h3>6b. Rankings on the common validation pairs</h3><p>Each judge below is evaluated on '+integer(r.n_pairs)+' scheduled pairs from '+integer(r.n_subjects)+' validation students. Generator rankings are compared with the human ranking for those same scheduled questions.</p>'+table(['Judge system','Execution','Kendall τ-b','Spearman ρ','Model-order agreement','Answer coverage','Calls','Missing calls','Attributed cost'],judges.map((j,i)=>[esc(names[i]),esc(extensionExecution(j,'followup:')),num(j.metrics?.kendall_tau_b),num(j.metrics?.spearman),pct(j.metrics?.pair_order_agreement),coverageText(j),integer(j.n_calls),integer(j.coverage?.missing_calls??0),j.execution_status==='unavailable'?'—':valid(j.estimated_usd)?'$'+num(j.estimated_usd,2):'—']))+'<p class="caption">Answer coverage is answered rubric comparisons divided by requested comparisons. The ranking target remains 12 generator systems and 66 possible generator matchups. Member selection was informed by earlier validation outcomes, so apparent improvements remain exploratory. Panel rows reuse calls and are not additional spend.</p></div>';
 const flash=judges.find(j=>j.variant==='or_deepseek41_flash_high'),panelResults=judges.filter(j=>catalogEntry('followup:'+j.variant)?.kind==='derived'&&valid(j.metrics?.kendall_tau_b));if(valid(flash?.metrics?.kendall_tau_b)&&panelResults.length&&panelResults.every(j=>j.metrics.kendall_tau_b<=flash.metrics.kendall_tau_b))content+='<p class="experiment-intro">The evaluable small panels do not improve the point ranking alignment over DeepSeek 4.1 Flash alone on these validation pairs (single-model Kendall τ-b '+num(flash.metrics.kendall_tau_b)+'). Combining models is therefore not automatically an improvement; the paired comparisons below show uncertainty relative to Terra, the original control.</p>';
 const paired=follow.paired_deltas,contrasts=Object.entries(paired?.comparisons||{});if(contrasts.length)content+='<div class="card table-section"><h3>6c. Changes relative to the Terra control</h3>'+table(['Judge','Change in Kendall τ-b','95% paired interval','Valid resamples'],contrasts.map(([name,c])=>{const m=c.metrics?.kendall_tau_b;return[esc(judgeName('followup:'+name)),num(m?.delta),interval(m),integer(m?.n_valid)+' / '+integer(paired.n_bootstrap)]}))+'<p class="caption">Changes are candidate minus control on the same resampled validation students. Intervals use the empirical 2.5th and 97.5th percentiles and can be asymmetric. They condition on the saved judge outputs and do not account for the earlier selection of these members. An interval including zero does not establish an improvement.</p></div>';
 content+='<div class="card table-section"><h3>Generator ranks under a selected follow-up judge</h3><div class="toolbar"><label class="control">Follow-up judge<select id="followup-judge-select">'+judges.map((j,i)=>'<option value="'+i+'">'+esc(names[i])+'</option>').join('')+'</select></label></div><div id="followup-ranking-note"></div><div id="followup-rank-chart" class="chart"></div><div id="followup-rank-table" class="scroll"></div></div>';
 $('followup-analysis').innerHTML=content;
 function rank(){const j=judges[+$('followup-judge-select').value];if(!j?.fit?.connected||j.execution_status==='unavailable'||j.unavailable_member_variants?.length){$('followup-ranking-note').innerHTML=empty(j?.fit?.warning||'No connected ranking is available.');$('followup-rank-chart').innerHTML='';$('followup-rank-table').innerHTML='';return}const hs=fitRows(r.human),as=fitRows(j.fit),by=new Map(as.map(row=>[row.model,row]));$('followup-ranking-note').innerHTML='<p class="caption">Human ranks use all '+integer(r.n_pairs)+' scheduled validation pairs. Missing scheduled calls: '+integer(j.coverage?.missing_calls??0)+'. Rank intervals resample students and do not measure new generation variability.</p>'+(j.bootstrap_cohort_warning?'<div class="notice">'+esc(j.bootstrap_cohort_warning)+'</div>':'');$('followup-rank-chart').innerHTML=rankChart(hs,as,'Validation humans',judgeName('followup:'+j.variant));$('followup-rank-table').innerHTML=table(['Lesson-plan generator system','Human rank','AI rank','Difference'],hs.map(h=>{const a=by.get(h.model);return[esc(modelName(h.model)),num(h.rank,1),num(a?.rank,1),valid(a?.rank)?num(a.rank-h.rank,1):'—'];}));}
 $('followup-judge-select').addEventListener('change',rank);rank();}
 $('followup-round-select').addEventListener('change',draw);draw();
}

function variantInfo(name){return (D.variants||[]).find(v=>v.variant===name)||{variant:name};}
function responseName(style){return ({evidence_winner:'Evidence, then A / B / tie',score_reason:'Signed preference and explanation',absolute_ratings:'Independent 1–5 ratings'}[style]||style||'—');}
function selectionCard(){
 const raw=D.selection;if(!raw)return;const s=typeof raw==='string'?{variant:raw}:raw,lock=D.manifest.selection_locked,info=variantInfo(s.variant),members=s.members||[[s.variant,0]],rule=s.rule||D.selection_rule||{};
 const rows=members.map(member=>{const name=Array.isArray(member)?member[0]:member.variant||member.name,draw=Array.isArray(member)?member[1]:member.draw,meta=variantInfo(name),config=D.configurations?.[name]||{};return[esc(judgeName(name)),esc(modelName(meta.model||config.members?.[0]?.model)),esc(meta.reasoning_effort||config.members?.[0]?.reasoning_effort||'—'),draw===0?'Reproducibly shuffled A/B order':draw===1?'The two plans swapped':valid(draw)?'Repeat '+integer(draw+1):'—',esc(responseName(meta.response_style||config.response_style)),esc(config.plan_projection==='human_display'?'Matches visible plan content':'Original exported plan')];});
 const testRound=D.rounds.find(r=>r.name==='test'),denseRound=D.rounds.find(r=>r.name==='dense'),test=testRound?.judges?.find(j=>j.variant===s.variant),dense=denseRound?.judges?.find(j=>j.variant===s.variant),testMetrics=D.final.test_metrics||test?.metrics,denseMetrics=D.final.dense_metrics||dense?.metrics;
 const resultRows=[['Validation selection',s.validation_metrics],['Held-out reviewed pairs',testMetrics],['Held-out dense AI comparisons',denseMetrics]].filter(([,m])=>m).map(([label,m])=>[esc(label),num(m.kendall_tau_b),num(m.spearman),pct(m.pair_order_agreement),num(m.rank_mae,2)]);
 let body='<div class="card" style="margin-bottom:18px"><div class="section-head"><h3>Selected judge: '+esc(judgeName(s.variant))+'</h3><span class="badge '+(lock?'':'pending')+'">'+(lock?'Selection locked before test':'Selection record; lock pending')+'</span></div><p class="small">'+(members.length===1?esc(modelName(info.model))+' judges each pair once, using '+esc(info.reasoning_effort||'the recorded')+' reasoning'+(info.pretest_context?' and restored pretest text':'')+'. A reproducible shuffle decides which plan appears as A or B. The prompt asks for evidence about each plan before choosing A, B, a tie, or insufficient information. Hidden archival answer fields are excluded so the visible lesson-plan content matches what humans reviewed.':'Member answers are combined using the recorded recipe. The first call uses a reproducibly shuffled plan order; a reversed call swaps the two plans.')+'</p><details><summary>Selected judge settings</summary><div class="scroll">'+table(['Judge','Model','Reasoning','Plan order','Requested answer','Plan content'],rows)+'</div></details>';
 if(resultRows.length)body+='<h4>Recorded outcomes for the selected judge</h4><div class="scroll">'+table(['Evaluation','Kendall τ-b','Spearman ρ','Model-order agreement','Rank MAE'],resultRows)+'</div><p class="caption" id="opening-metric-guide">Kendall τ-b ranges from −1 (reversed ordering) to 1 (identical ordering); near 0 means little ordering association. Spearman ρ correlates rank positions: 1 means identical ranks, −1 reversed ranks, and near 0 little association. Model-order agreement is the share of the 66 unordered generator-system pairs placed in the same order. Rank MAE is the average number of rank positions off; lower is better.</p>';
 const humanTop=fitRows(testRound?.human).find(row=>row.rank===1),judgeTop=fitRows(test?.fit).find(row=>row.rank===1);if(humanTop&&judgeTop){const range=test.bootstrap?.rank_intervals?.human?.[humanTop.model];body+='<p class="caption">Reviewed-test point leaders: human '+esc(modelName(humanTop.model))+'; AI '+esc(modelName(judgeTop.model))+'. '+(range?'The human point leader has a 95% bootstrap rank range of '+interval(range,1)+'. ':'')+'Broad ranking agreement does not identify a certain winner.</p>';}
 if(testMetrics&&denseMetrics){const reviewedEqual=test?.subject_balanced?.metrics?.kendall_tau_b,denseEqual=dense?.subject_balanced?.metrics?.kendall_tau_b;let densityNote='Expanding the AI schedule from '+integer(testRound?.n_pairs)+' reviewed pairs to '+integer(denseRound?.n_pairs)+' available pairs changes point Kendall alignment from '+num(testMetrics.kendall_tau_b)+' to '+num(denseMetrics.kendall_tau_b)+'. ';if(valid(reviewedEqual)&&valid(denseEqual))densityNote+='A student with many available plans otherwise contributes more pair comparisons. Giving each student the same total influence yields '+num(reviewedEqual)+' on the reviewed schedule and '+num(denseEqual)+' on the dense schedule. '+(Math.abs(reviewedEqual-denseEqual)<1e-9?'The small ranking gain disappears under this equal-student check. ':'');if(valid(testMetrics.probability_mae)&&valid(denseMetrics.probability_mae))densityNote+='Fitted win-probability MAE changes from '+num(testMetrics.probability_mae)+' to '+num(denseMetrics.probability_mae)+' (lower is better). ';densityNote+='These are sampling and weighting results on the existing plans, not an independent improvement trial.';body+='<p class="caption" id="density-outcome-note">'+esc(densityNote)+'</p>';}
 if(!testMetrics)body+='<p class="caption">Held-out results are pending. Validation selection performance is not a forecast of test performance.</p>';
 body+='<p class="caption">Selection criterion: '+esc(rule.primary||'as recorded')+'. '+(valid(rule.minimum_criterion_coverage)?'Minimum criterion coverage '+pct(rule.minimum_criterion_coverage)+'. ':'')+'Tie-breakers: '+esc((rule.tie_breakers||[]).join('; ')||'see decision record')+'.</p>';
 if(valid(D.final.new_completed_calls))body+='<p class="small">Completed study calls: '+integer(D.final.new_completed_calls)+(valid(D.final.estimated_usd)?' · Estimated total token cost: $'+num(D.final.estimated_usd,2):'')+'. Estimates use returned token usage from completed calls; usage for failed or timed-out attempts may be unavailable. These are not reconciled provider invoices.</p>';
 body+='<details><summary>Frozen selection and configuration hashes</summary>'+brief(s)+'</details></div>';$('selection-note').insertAdjacentHTML('beforeend',body);
}
function selectedRound(){return D.rounds[+$('round-select').value]||null}
function selectedJudge(){return selectedRound()?.judges?.[+$('judge-select').value]||null}
function setupSelectors(){
 $('round-select').innerHTML=D.rounds.map((r,i)=>'<option value="'+i+'">'+esc(roundLabel(r.name))+'</option>').join('')||'<option>No completed AI round</option>';
 $('round-select').value=Math.max(0,D.rounds.findLastIndex(r=>r.name==='test')>=0?D.rounds.findLastIndex(r=>r.name==='test'):D.rounds.length-1);
 function changeRound(){const r=selectedRound();$('judge-select').innerHTML=(r?.judges||[]).map((j,i)=>'<option value="'+i+'">'+esc(judgeName(j))+'</option>').join('')||'<option>No completed judge</option>';const selected=r?.judges?.findIndex(j=>j.variant===(D.selection?.variant||D.selection));if(selected>=0)$('judge-select').value=selected;$('compare-select').innerHTML='<option value="human">Human reference</option>'+(r?.judges||[]).map((j,i)=>'<option value="'+i+'">'+esc(judgeName(j))+'</option>').join('');activeCriterion='overall';changeJudge();}
 function changeJudge(){const j=selectedJudge();const keys=['overall',...Object.keys(j?.criteria||{})];if(!keys.includes(activeCriterion))activeCriterion='overall';$('criterion-tabs').innerHTML=keys.map(k=>'<button type="button" class="'+(activeCriterion===k?'active':'')+'" data-criterion="'+esc(k)+'">'+esc(criterionLabel(k))+'</button>').join('');$('criterion-tabs').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{activeCriterion=b.dataset.criterion;changeJudge()}));ranking();renderAIDensity();}
 $('round-select').addEventListener('change',changeRound);$('judge-select').addEventListener('change',changeJudge);$('compare-select').addEventListener('change',ranking);changeRound();
}
function ranking(){
 const r=selectedRound(),j=selectedJudge(),crit=j?.criteria?.[activeCriterion];let h=crit?.human||r?.human||human?.leaderboard,a=crit?.judge||crit?.fit||j?.fit,metrics=crit?.metrics||j?.metrics||{},boot=activeCriterion==='overall'?j?.bootstrap:null;
 let referenceName='Human reference · '+(r?.split||human?.split||'unavailable');const comparingAI=$('compare-select').value!=='human';if(comparingAI){const other=r?.judges?.[+$('compare-select').value];h=activeCriterion==='overall'?other?.fit:other?.criteria?.[activeCriterion]?.judge;referenceName=other?judgeName(other):'Comparison judge';metrics={};boot=null;}
 const intervals=boot?.metric_intervals||{};
 $('ranking-metrics').innerHTML=stat(num(metrics.kendall_tau_b),'Kendall τ-b','95% range '+interval(intervals.kendall_tau_b))+stat(num(metrics.spearman),'Spearman ρ','95% range '+interval(intervals.spearman))+stat(pct(metrics.pair_order_agreement),'Model-pair order agreement',valid(metrics.n_model_pairs)?integer(metrics.n_model_pairs)+' unordered generator-system pairs':'Generator-system ranks, not lesson-plan verdicts')+stat(num(metrics.rank_mae,2),'Mean absolute rank error','Lower is better');
 $('leaderboard-title').textContent=referenceName+' vs '+(j?judgeName(j):'AI results pending')+' · '+criterionLabel(activeCriterion);
 $('reference-legend').textContent=referenceName;$('judge-legend').textContent=judgeName(j)||'AI judge';
 $('ranking-coverage').textContent=(r?'Round: '+integer(r.n_pairs)+' pairs · '+integer(r.n_subjects)+' subjects':'Human validation reference')+(j?' · '+integer(j.n_calls)+' calls':'');
 const warning=metrics.warning||a?.warning||h?.warning;
 const bi=boot?'Valid paired bootstrap draws: '+integer(boot.valid_bootstrap_fits?.paired)+' / '+integer(boot.n_bootstrap)+'. ':'';
 $('ranking-warning').innerHTML=(comparingAI?'<p class="small">Comparing two AI rank positions. Human-alignment metrics and uncertainty are not recalculated for this browser comparison.</p>':'')+(r?.complete===false?'<div class="notice">Partial round: '+integer(r.n_completed_calls)+' of '+integer(r.n_expected_calls)+' calls completed. Missing judgments can change the comparison schedule.</div>':'')+(!comparingAI&&j?.bootstrap_cohort_warning?'<div class="notice">'+esc(j.bootstrap_cohort_warning)+'</div>':'')+(warning?'<div class="notice">'+esc(warning)+'</div>':bi?'<p class="small">'+esc(bi+boot.interval_warning)+'</p>':'');
 const hs=fitRows(h),as=fitRows(a),ai=new Map(as.map(x=>[x.model,x]));
 $('leaderboard-table').innerHTML=hs.length?table(['Lesson-plan generator system','Human / reference rank','AI rank','Rank difference','Reference BT','AI BT','Reference pairs','AI pairs'],hs.map(x=>{const y=ai.get(x.model),delta=valid(y?.rank)&&valid(x.rank)?y.rank-x.rank:null,hr=boot?.rank_intervals?.human?.[x.model]||x.rank_interval,ar=boot?.rank_intervals?.judge?.[x.model]||y?.rank_interval;return ['<strong>'+esc(x.label||modelName(x.model))+'</strong>',num(x.rank,1)+(hr?'<span class="sub">'+interval(hr,1)+'</span>':''),num(y?.rank,1)+(ar?'<span class="sub">'+interval(ar,1)+'</span>':''),valid(delta)?'<span class="'+(delta<0?'positive':delta>0?'negative':'')+'">'+(delta>0?'+':'')+num(delta,1)+'</span>':'—',num(x.score),num(y?.score),integer(x.n_comparisons),integer(y?.n_comparisons)];})):empty('No completed ranking fit is available.');
 $('rank-chart').innerHTML=rankChart(hs,as,referenceName,judgeName(j)||'AI judge');
}
function rankChart(h,a,referenceName,judgeName){if(!h.length||!a.length)return '';const n=h.length,w=1050,height=n*25+65,left=290,right=995,y=i=>45+i*25,x=rank=>left+(rank-1)/(Math.max(n-1,1))*(right-left);const am=new Map(a.map(z=>[z.model,z]));let s='<svg role="img" aria-label="Reference and AI rank positions by lesson-plan generator system" viewBox="0 0 '+w+' '+height+'">';for(let i=1;i<=n;i++)s+='<line class="gridline" x1="'+x(i)+'" y1="20" x2="'+x(i)+'" y2="'+(height-20)+'"/><text x="'+x(i)+'" y="13" text-anchor="middle">'+i+'</text>';h.forEach((row,i)=>{const other=am.get(row.model);s+='<text x="275" y="'+(y(i)+4)+'" text-anchor="end">'+esc(modelName(row.model))+'</text>';if(valid(row.rank))s+='<circle cx="'+x(row.rank)+'" cy="'+y(i)+'" r="4.5" fill="#087f83"><title>'+esc(referenceName)+': rank '+row.rank+'</title></circle>';if(valid(row.rank)&&valid(other?.rank))s+='<line x1="'+x(row.rank)+'" y1="'+y(i)+'" x2="'+x(other.rank)+'" y2="'+y(i)+'" stroke="#aebbc4" stroke-width="2"/>';if(valid(other?.rank))s+='<circle cx="'+x(other.rank)+'" cy="'+y(i)+'" r="4" fill="#5571c4"><title>'+esc(judgeName)+': rank '+other.rank+'</title></circle>';});return s+'</svg>';}
function roundResults(){
 const screen=D.rounds.find(r=>r.name==='round1'),refine=D.rounds.find(r=>r.name==='round2'),legacy=D.rounds.find(r=>r.name==='legacy_development');
 let body='<article class="experiment" id="screen-experiment"><h3>Experiment 1 · Screen models, reasoning effort, and response design</h3><p class="experiment-intro">We first compare Luna, Terra, Sol, and Astra at low and high reasoning effort on the same 60 validation pairs. Every model sees the original exported inputs. The main prompt asks for evidence about each plan and then an explicit winner. Two Luna controls test a signed preference score and separate ratings for the plans. This screen asks which judge designs are promising; it does not estimate their held-out performance.</p><div class="card"><h4>1a. Model and reasoning effort</h4><div id="model-effort-table" class="scroll"></div><p class="caption">All eight rows use the same 60 plan pairs, 60 students, 480 requested rubric comparisons, and the same evidence-and-winner prompt. Each score compares the resulting 12-generator ranking with the human ranking for those pairs. “High” changes the requested reasoning setting; it does not guarantee better alignment or equal computation across models.</p><p>Why can more reasoning perform worse? A longer reasoning budget can change rubric interpretation, how minor defects are weighted, and which issues receive attention. This screen does not isolate those mechanisms: each setting has one saved generation per task, and the cohort is small. A lower point score therefore shows what happened here, rather than proving that additional reasoning is harmful.</p><div id="reasoning-deltas"></div></div><div class="card table-section"><h4>1b. Response design, holding the model and reasoning fixed</h4><div id="prompt-ablation-table" class="scroll"></div><p class="caption">Compare the two low-reasoning Luna rows with each other, and the two high-reasoning Luna rows with each other. The explicit-winner format avoids asking the model to encode the winner as a signed number. The signed-score control also differs in evidence and quality instructions, so this is a prompt-design comparison rather than a pure JSON-format effect.</p></div><div class="card table-section"><h4>1c. Combine the four models</h4><p>Each panel uses Luna, Terra, Sol, and Astra on the same pairs, with all four members at the named reasoning level. Averaging combines their fractional votes. Strict majority requires enough matching A/tie/B answers and otherwise leaves that rubric comparison unanswered.</p><div id="screen-panel-table" class="scroll"></div><p class="caption">Panels reuse the corresponding single-model calls above. Their cost is the attributed cost of those calls, not an additional panel purchase. A mean vote of 0.5 can result from opposite preferences; it need not mean the members agreed that the plans were equal.</p></div></article>';
 body+='<article class="experiment" id="refinement-experiment"><h3>Experiment 2 · Refine the inputs and choose a judge</h3><p class="experiment-intro">The explanation audit found both response-direction problems and incomplete pretest inputs. We restore verified text that the export omitted and remove archival answer fields that human reviewers did not see. Round 2 compares these refined Luna and Terra judges, two ways to combine both plan orders, and the original Luna low-reasoning control on 120 validation pairs. The larger sample helps choose a judge, but its scores cannot be compared directly with Round 1 to measure the effect of an input change.</p><div class="card"><h4>2a. Candidate selection on the common 120-pair schedule</h4><div id="refinement-table" class="scroll"></div><p class="caption">The five candidates use the same 120 reviewed pairs from 82 validation students. The human reference uses those scheduled questions even when a judge abstains. Selection maximizes Kendall τ-b among candidates with at least 90% answer coverage, with the recorded tie-breakers. The winning configuration was frozen before any test verdict entered the evaluation or selection calculations.</p></div><div class="card table-section"><h4>2b. Isolate the input bundle on the original 60 pairs</h4><p>The original and restored-input versions of Luna and Terra are rerun on exactly the same 60 screening pairs. The bundle restores pretest text, removes hidden archival answer fields, and revises template wording; it does not tell us which of those components caused any change.</p><div id="input-ablation-table" class="scroll"></div><p id="input-ablation-note" class="caption"></p></div><div id="order-ablation"></div></article>';
 if(legacy)body+='<details><summary>Development baseline: the four earlier judge systems</summary><p>These 100 previously inspected comparisons belong to development, not to the validation or held-out test schedules. They motivate the revisions, and are not a fair additional test of the selected judge.</p>'+judgeResultTable(legacy)+'</details>';
 $('round-results').innerHTML=body;
 $('refinement-table').innerHTML=refine?judgeResultTable(refine):empty('The refinement analysis is pending.');
 $('screen-panel-table').innerHTML=screen?judgeResultTable(screen,j=>(j.members||[]).length>1):empty('The panel analysis is pending.');
}
function judgeResultTable(round,filter=()=>true){
 return table(['Judge system','Kendall τ-b','Spearman ρ','Model-order agreement','Rank MAE','Answer coverage','Calls','Estimated cost'],(round.judges||[]).filter(filter).map(j=>{const m=j.metrics||{};return[esc(judgeName(j)),num(m.kendall_tau_b)+(j.bootstrap?.metric_intervals?.kendall_tau_b?'<span class="sub">'+interval(j.bootstrap.metric_intervals.kendall_tau_b)+'</span>':''),num(m.spearman),pct(m.pair_order_agreement),num(m.rank_mae),coverageText(j),integer(j.n_calls),valid(j.estimated_usd)?'$'+num(j.estimated_usd,2):'—'];}));
}
function setupHumanSelector(){
 const splits=['validation','test'].filter(key=>D.human[key]);$('human-split-select').innerHTML=splits.map(key=>'<option value="'+key+'">'+(key==='test'?'Held-out test subjects':'Validation subjects')+'</option>').join('')||'<option>Human analysis pending</option>';if(splits.length)$('human-split-select').value=D.human.test?'test':'validation';
 $('human-split-select').addEventListener('change',()=>{human=D.human[$('human-split-select').value];humanResults();if(!selectedRound())ranking();});$('human-density-ref').addEventListener('change',humanDensity);$('human-density-axis').addEventListener('change',humanDensity);humanResults();
}

function humanComparator(){
 const d=D.human_comparator;if(!d)return;const values=d.summary?.kendall_tau_b||{},support=d.valid_partition_support?.ai||{};
 $('human-comparator').innerHTML='<div class="card table-section"><h3>A more comparable test: the same questions and human reference</h3><p>We divide the test reviewers into two disjoint groups, keep only questions answered by both groups and Terra, and compare both Terra and one human group against the other group. Then we swap which human group is the reference and average the two orientations. Every side uses the same students, plan pairs, and exact rubric questions within a split.</p>'+table(['Comparison against the same human reference','Mean Kendall τ-b','2.5th–97.5th percentile across reviewer splits'],[['Terra judge',num(values.ai_vs_reference?.mean),interval(values.ai_vs_reference)],['Other human reviewer group',num(values.human_vs_reference?.mean),interval(values.human_vs_reference)],['Difference: Terra minus human group',num(values.delta_ai_minus_human?.mean),interval(values.delta_ai_minus_human)]])+'<p class="caption">This post-study diagnostic has '+integer(d.n_valid_partitions)+' valid connected partitions out of '+integer(d.n_partitions)+'. Each side uses an average of '+num(support.pairs?.mean,1)+' plan pairs, '+num(support.subjects?.mean,1)+' students, and '+num(support.criterion_units?.mean,1)+' exact rubric comparisons. Human groups can still contribute different numbers of repeated reviews; Terra has one saved answer per question. Both reference orientations are averaged within a partition, not counted as independent observations.</p><p>Terra aligns better on average in this matched comparison, but some reviewer splits favor the human group. The displayed range measures sensitivity to repeated partitions of these reviewers. It is not a confidence interval from independent new studies, does not resample new AI answers, and does not establish that AI is generally more reliable than humans.</p></div>';
}
function humanResults(){
 if(!human){$('human-stats').innerHTML=empty('Human analysis is pending.');return}const direct=human.direct_human_agreement||{},split=human.reviewer_split_half||{},s=split.summary||{};
 $('human-split-note').textContent=(human.split==='test'?'Held-out human benchmark; evaluated after selection.':'Validation benchmark used during judge development.')+' '+integer(human.coverage?.subjects)+' subjects, '+integer(human.coverage?.pairs)+' pairs. Human density below uses this same cohort.';$('human-reference-title').textContent=(human.split==='test'?'Test':'Validation')+' human reference and rank uncertainty';
 $('human-stats').innerHTML=stat(pct(direct.direction_agreement),'Human rubric direction agreement',integer(direct.repeated_criterion_units)+' repeated pair/criterion units')+stat(pct(s.pair_order_agreement?.mean),'Human model-order agreement','Mean across reviewer-disjoint splits')+stat(num(s.kendall_tau_b?.mean),'Human ranking Kendall τ-b','Mean across connected split fits')+stat(integer(s.kendall_tau_b?.n)+' / '+integer(split.repeats),'Connected reviewer splits','Disconnected fits excluded from rank metrics');
 $('human-consistency').innerHTML=table(['Measurement','Mean','2.5–97.5 percentile'],[['Kendall τ-b',num(s.kendall_tau_b?.mean),interval(s.kendall_tau_b)],['Spearman ρ',num(s.spearman?.mean),interval(s.spearman)],['Model order agreement',pct(s.pair_order_agreement?.mean),percentInterval(s.pair_order_agreement)],['Rank MAE',num(s.rank_mae?.mean,2),interval(s.rank_mae)],['Same top model',pct(s.top1_overlap?.mean),percentInterval(s.top1_overlap)]])+'<p class="caption">'+esc(split.interpretation||'')+'</p><p class="caption">'+integer(human.coverage?.review_sessions)+' reviews from '+integer(human.coverage?.reviewers)+' reviewers over '+integer(human.coverage?.pairs)+' pairs. Direct agreement averages '+integer(direct.reviewer_pair_comparisons)+' cross-reviewer comparisons with equal weight per exact pair/criterion unit. Each valid split ranks the fixed 12 generator systems and compares their 66 unordered model pairs. Full-reference and split-half top-model results use different amounts of data. Inspect rank intervals before treating a point winner as stable.</p>';
 const cross=D.human_cross_split?.metrics;if(cross)$('human-consistency').insertAdjacentHTML('beforeend','<p class="small"><strong>Full human leaderboard across cohorts.</strong> Validation versus held-out test: Kendall τ-b '+num(cross.kendall_tau_b)+', Spearman ρ '+num(cross.spearman)+', model-order agreement '+pct(cross.pair_order_agreement)+', rank MAE '+num(cross.rank_mae,2)+'. This post-lock descriptive comparison uses disjoint student contexts, with reviewers allowed to recur; it is separate from the reviewer-disjoint results above. Broad rank stability does not establish a certain test winner.</p>');
 const comparison=D.revision?.human_comparison_context?.[human.split];if(comparison){const pairs=comparison.half_pair_counts_connected_draws,students=comparison.half_subject_counts_connected_draws;$('human-consistency').insertAdjacentHTML('beforeend','<p class="caption"><strong>Different amounts of evidence:</strong> the connected reviewer-half comparisons use an average of '+num(pairs?.mean,1)+' shared plan pairs and '+num(students?.mean,1)+' students. The AI comparison uses the full scheduled cohort ('+(human.split==='test'?'155 pairs and 83 students':'120 pairs and 82 students')+'). A higher AI-versus-human-aggregate correlation does not establish higher reliability than individual humans; the reference and sample sizes differ.</p>');}
 const boot=human.human_subject_bootstrap,rows=boot?.leaderboard||fitRows(human.leaderboard);
 const top=rows.find(r=>r.rank===1);$('human-leaderboard').innerHTML=(top&&valid(top.top1_frequency)?'<div class="notice '+(top.top1_frequency>=.8?'good':'')+'">Point leader '+esc(modelName(top.model))+' appears first in '+pct(top.top1_frequency)+' of valid bootstrap fits; its 95% rank range is '+interval(top.rank_interval,1)+'. This is resampling uncertainty, not a probability of future performance.</div>':'')+table(['Lesson-plan generator system','Rank','95% rank range','First-place frequency'],rows.map(r=>[esc(modelName(r.model)),num(r.rank,1),interval(r.rank_interval,1),pct(r.top1_frequency)]))+'<p class="caption">'+esc(boot?.interpretation||'')+'</p>';
 const previous=$('human-density-axis').value,keys=[...new Set((human.density_independent_reviewers?.experiments||[]).map(e=>e.axis))];$('human-density-axis').innerHTML=keys.map(k=>'<option value="'+esc(k)+'">'+esc(k.replaceAll('_',' '))+'</option>').join('');if(keys.includes(previous))$('human-density-axis').value=previous;else if(keys.includes('pairs'))$('human-density-axis').value='pairs';humanDensity();
}
function sensitivityTables(){
 $('statistical-guide').innerHTML='<div class="card" id="statistical-ablations"><h3>Three different checks, with different interpretations</h3><p><strong>Student weighting:</strong> a student with four plans supplies six pairs, while a student with two plans supplies one. Equal pair weights therefore let the first student contribute six times as much. The equal-student check gives each student the same total influence, shared among their pairs, then rescales weights to preserve the original total pair weight.</p><p><strong>Human reference:</strong> normally we compare the AI ranking with all human answers for the scheduled questions. The matched-answer check uses only questions the AI also answered. For example, if a judge skips difficult accuracy questions, removing those human answers can change the human ranking and raise measured agreement without improving a single AI answer. The whole-split check instead includes every available reviewed criterion variant in the split. For example, comparing the unchanged 60-pair screening AI ranking with humans from all 134 reviewed validation pairs changes the reference and can change agreement without changing the judge.</p><p><strong>Quality rubric:</strong> the default combines eight families of quality questions. Removing answer accuracy or strategies/formats changes what “quality” means for both human and AI rankings. These are diagnostic checks using the same verdicts; they are not new judge configurations or additional confirmation tests.</p></div>';
 const chosen=D.selection?.variant||D.selection,selected=[],extra=[];
 for(const r of D.rounds){const rows=[];for(const j of r.judges||[]){const variants=[['Primary: equal weight per plan pair',j.metrics],['Equal total weight per student',j.subject_balanced?.metrics],['Human reference: only questions also answered by AI',j.matched?.metrics],['Human reference: all reviewed criterion variants',j.metrics_full_human]];for(const [key,a]of Object.entries(j.rubric_ablations||{}))variants.push([({exclude_q07:'Exclude answer accuracy',exclude_q08:'Exclude strategies / formats',exclude_q07_q08:'Exclude both accuracy and strategies / formats'}[key]||key.replaceAll('_',' ')),a.metrics]);if(r.name==='dense'&&j.reviewed_only?.metrics)variants.push(['AI uses only human-reviewed pairs',j.reviewed_only.metrics]);const target=j.variant===chosen&&['test','dense'].includes(r.name)?selected:rows;for(const [condition,m]of variants)if(m)target.push([esc(roundLabel(r.name)),esc(judgeName(j)),esc(condition),num(m.kendall_tau_b),num(m.spearman),pct(m.pair_order_agreement),num(m.rank_mae)]);}if(rows.length)extra.push('<details><summary>'+esc(roundLabel(r.name))+'</summary>'+table(['Cohort','Judge','What changes','Kendall τ-b','Spearman ρ','Model-order agreement','Rank MAE'],rows)+'<p class="caption">These checks reuse '+integer(r.n_pairs)+' scheduled plan pairs. Each row changes only the stated ranking definition or human comparison set; the judge outputs are unchanged.</p></details>');}
 $('selected-sensitivities').innerHTML='<div class="card table-section"><h3>Selected Terra judge: held-out sensitivity results</h3>'+ (selected.length?table(['Evaluation schedule','Judge','What changes','Kendall τ-b','Spearman ρ','Model-order agreement','Rank MAE'],selected):empty('Selected held-out checks are pending.'))+'<p class="caption">The reviewed test contains 155 plan pairs; the dense AI schedule contains 293. Both use the same 83 test students. Primary rows provide a baseline within each schedule. Changes in human reference alter which human answers define the comparison; rubric exclusions apply to both rankings. These rows do not change the frozen judge or supply new held-out evidence.</p></div>';
 $('extra-sensitivities').innerHTML=extra.join('');
}
function ablationRow(name,analysis,condition=''){
 const meta=variantInfo(name),j=analysis?.judges?.find(x=>x.variant===name),m=j?.metrics||{};
 return [esc(judgeName(name)),esc(condition||responseName(meta.response_style)),num(m.kendall_tau_b),num(m.spearman),pct(m.pair_order_agreement),coverageText(j),valid(j?.estimated_usd)?'$'+num(j.estimated_usd,2):'—'];
}
function judgeAblations(){
 const r=D.rounds.find(x=>x.name==='round1'),headers=['Judge system','Response / input','Kendall τ-b','Spearman ρ','Model-order agreement','Answer coverage','Estimated cost'];
 const bases=['luna_low','luna_high','terra_low','terra_high','sol_low','sol_high','astra_low','astra_high'];
 $('model-effort-table').innerHTML=(!r?empty('The screening analysis is pending.'):r.complete===false?'<div class="notice">Partial screening: '+integer(r.n_completed_calls)+' / '+integer(r.n_expected_calls)+' calls completed.</div>':'')+table(headers,bases.map(name=>ablationRow(name,r)));
 $('prompt-ablation-table').innerHTML=table(headers,['legacy_luna_low','luna_low','luna_high','absolute_luna_high'].map(name=>ablationRow(name,r)));
 orderAblation();
 const a=D.input_ablation,expected=D.manifest.round1_validation_pairs?.length,matched=a&&a.n_pairs===expected&&D.input_ablation_cohort?.verified;
 if(!matched){$('input-ablation-table').innerHTML=empty('Verified comparisons on the same input-ablation pairs are pending.');$('input-ablation-note').textContent='This comparison requires the same '+integer(expected)+' screening pairs for all four variants. '+(D.input_ablation_cohort?.reason||'');return;}
 $('input-ablation-table').innerHTML=(a.complete===false?'<div class="notice">Some input-ablation calls are missing.</div>':'')+table(headers,['luna_high','enhanced_luna_high','terra_high','enhanced_terra_high'].map(name=>ablationRow(name,a,name.startsWith('enhanced_')?'Restored text; hidden archive fields removed':'Original exported input')));
 $('input-ablation-note').textContent='All four rows use the same '+integer(a.n_pairs)+' pairs and '+integer(a.n_subjects)+' students, with independently generated judge responses. Changes here may reflect the input bundle and generation variability. The 120-pair selection scores are deliberately excluded from this table.';
}
function orderAblation(){
 const r=D.rounds.find(x=>x.name==='round2');if(!r)return;const names=['enhanced_luna_high','enhanced_luna_high_balanced','enhanced_luna_high_balanced_strict'],variants=names.map(name=>r.judges?.find(j=>j.variant===name)).filter(Boolean);if(variants.length<2)return;
 const labels=['One reproducibly shuffled presentation','Both plan orders; average votes','Both plan orders; require agreement'];const rows=variants.map(j=>[esc(labels[names.indexOf(j.variant)]),num(j.metrics?.kendall_tau_b),pct(j.metrics?.pair_order_agreement),coverageText(j),integer(j.n_calls)]);const single=variants.find(j=>j.variant===names[0]),soft=variants.find(j=>j.variant===names[1]);
 let note='All three rows use Luna with high reasoning and the same '+integer(r.n_pairs)+' validation pairs. Both-order strategies reuse the original call plus one call with the plans swapped. Averaging opposite votes gives a half-win; requiring agreement instead leaves that question unanswered.';
 if(valid(single?.metrics?.kendall_tau_b)&&valid(soft?.metrics?.kendall_tau_b)&&soft.metrics.kendall_tau_b<=single.metrics.kendall_tau_b)note+=' Averaging both orders did not improve the point ranking alignment in this cohort.';
 $('order-ablation').innerHTML='<div class="card table-section"><h4>2c. Does reversing the plans reduce presentation bias?</h4><p>The first plan can receive an advantage merely from appearing first. We judge the same pair in both orders, restore the common direction, and compare averaging with requiring matching answers.</p><div class="scroll">'+table(['Luna strategy','Kendall τ-b','Model-order agreement','Answer coverage','Calls'],rows)+'</div><p class="caption">'+esc(note)+'</p></div>';
}

function revisionDetails(){
 const revision=D.revision;if(!revision)return;const effort=revision.reasoning_effort;
 if(effort&&$('reasoning-deltas'))$('reasoning-deltas').innerHTML='<h4>How much does the reasoning setting change alignment?</h4>'+table(['Judge model','Low reasoning τ-b','High reasoning τ-b','Change: high minus low','95% paired interval'],Object.entries(effort.contrasts||{}).map(([name,c])=>{const m=c.metrics?.kendall_tau_b||{};return[esc(modelName('gpt-'+(name==='astra'?'6-':'5.6-')+name)),num(m.baseline),num(m.candidate),num(m.delta),interval(m)];}))+'<p class="caption">These paired differences reuse the same '+integer(effort.n_pairs)+' saved pair comparisons and '+integer(effort.n_bootstrap)+' shared student resamples per model. Positive changes favor high reasoning. All four intervals include zero: low had higher point alignment for Luna and Terra, while high had higher point alignment for Sol and Astra. The saved prompts, schema, and plan order match within each model’s comparison. These results do not demonstrate a general benefit or harm from higher reasoning effort.</p>';
 const audit=revision.example_interval_audit?.contrasts?.one_example_minus_zero_example?.metrics?.kendall_tau_b;if(audit?.lower_exactly_equals_point)$('extension-deltas').insertAdjacentHTML('beforeend','<details><summary>Why does the example interval start at the observed change?</summary><p>The observed change is '+num(audit.delta,6)+', and the audited lower percentile is exactly the same at full precision. It is not clipped to the estimate. Of '+integer(audit.n_valid)+' saved resamples, '+integer(audit.n_draws_below_point)+' fall below the point change, '+integer(audit.n_draws_equal_point_tolerance_1e12)+' equal it within numerical tolerance, and '+integer(audit.n_draws_above_point)+' lie above it. The 2.5th percentile falls on that repeated value.</p><p>With 12 untied ranks there are 66 generator matchups; reversing one matchup changes Kendall τ by 2/66, about 0.0303. Such steps create repeated values. Percentile intervals follow this empirical distribution and need not be centered on the observed statistic. This is conditional resampling uncertainty for the saved outputs, not a probability that a human example harms future judging.</p></details>');
}
function contextDetails(){
 const c=D.context?.coverage||{},p=D.context?.provenance||{};if(!Object.keys(c).length){$('context-stats').innerHTML=empty('No completed context-coverage audit is available.');return}
 const restored=valid(c.unique_qc_missing_before)&&valid(c.unique_qc_missing_after)?c.unique_qc_missing_before-c.unique_qc_missing_after:null;
 $('context-stats').innerHTML=stat(integer(restored),'QC quantity pairs restored','Missing before: '+integer(c.unique_qc_missing_before)+'; after: '+integer(c.unique_qc_missing_after))+stat(integer(c.unique_chart_stimuli_recovered),'Shared chart texts restored','Original application text supplement')+stat(integer(c.unique_figures_unrendered),'Figures still unrendered','Unavailable figures explicitly marked')+stat(integer(p.matched_items)+' / '+integer(c.unique_items),'Items with matching shared fields','CSV and application-bank comparison');
 $('context-changes').innerHTML=table(['Input component','Baseline','Enhanced variants'],[['Quantitative-comparison quantities',integer(c.unique_qc_missing_before)+' incomplete items',integer(c.unique_qc_missing_after)+' incomplete items'],['Shared chart text','Absent',integer(c.unique_chart_stimuli_recovered)+' original texts restored'],['SVG figure rendering','Absent','Still absent; limitation marked'],['Stored practice answer','Displayed answer retained','Displayed answer retained'],['Archival answer_original','Present when stored','Removed from judge input']])+'<p class="caption">Coverage audit: '+integer(c.development_subjects)+' development subjects and '+integer(c.development_pretest_rows)+' pretest rows. Quantities affect '+integer(c.row_qc_missing_before)+' response rows; chart text is added to '+integer(c.rows_shared_data_added)+' rows.</p><p class="small">The human-display projection removes hidden archival answer metadata. It does not restore original keys, repair plans, or rewrite human verdicts.</p>';
 $('context-provenance').innerHTML='<p class="small">The supplement comes from the neighboring original application bank, outside the immutable SSOT release. Shared item fields match exactly, but omitted historical fields cannot be validated against a CSV that never contained them. No historical deployment replay was performed.</p><p class="small">All '+integer(c.unique_figures_unrendered)+' source SVGs remain unrendered in these text requests. Available chart text is retained; no figure descriptions are invented.</p><p class="mono"><strong>Application supplement SHA-256</strong><br>'+esc(p.source_bank_sha256||'—')+'</p><p class="mono"><strong>SSOT comparison-bank SHA-256</strong><br>'+esc(p.ssot_bank_sha256||'—')+'</p>'+(c.human_verdicts_used_for_context_reconstruction===false?'<p class="small">No human verdicts were used to reconstruct input context.</p>':'')+'<details><summary>Context coverage and provenance record</summary>'+brief({coverage:c,provenance:p})+'</details>';
 $('context-summary-note').textContent='Rankings describe stored plans, including upstream answer-key postprocessing. Enhanced variants restore '+integer(restored)+' missing QC quantity pairs and '+integer(c.unique_chart_stimuli_recovered)+' shared chart texts using a separately pinned application supplement; '+integer(c.unique_figures_unrendered)+' figures remain unrendered. Baseline judge inputs are unchanged.';
}
function densityCaveats(){
 $('density').insertAdjacentHTML('beforeend','<p class="caption">The Bradley–Terry ridge penalty stays fixed at 1.0. Its relative shrinkage is stronger at lower density, so fitted-probability errors do not isolate sampling or calibration effects. Additional unreviewed edges can change the opponent schedule, student weights, and q08 wording mix; inspect the subject-balanced and no-q08 sensitivities alongside the primary result.</p>');
}
function lineChart(series,xLabel='Completed pair comparisons',yLabel='Kendall τ-b'){
 const all=series.flatMap(s=>s.points).filter(p=>valid(p.x)&&valid(p.y));if(!all.length)return empty('No connected fits are available for this density setting.');const w=555,h=280,left=48,right=532,top=28,bottom=233,xmax=Math.max(...all.map(p=>p.x),1),xmin=Math.min(0,...all.map(p=>p.x)),ymin=Math.min(0,...all.map(p=>p.low??p.y)),ymax=1;const X=x=>left+(x-xmin)/(xmax-xmin)*(right-left),Y=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);let out='<svg role="img" aria-label="'+esc(yLabel+' versus '+xLabel)+'" viewBox="0 0 '+w+' '+h+'">';for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;out+='<line class="gridline" x1="'+left+'" x2="'+right+'" y1="'+Y(y)+'" y2="'+Y(y)+'"/><text x="39" y="'+(Y(y)+4)+'" text-anchor="end">'+num(y,2)+'</text>';const x=xmin+(xmax-xmin)*i/4;out+='<text x="'+X(x)+'" y="252" text-anchor="middle">'+num(x,xmax<5?1:0)+'</text>';}out+='<text class="axis-title" x="'+left+'" y="15">'+esc(yLabel)+'</text><text class="axis-title" x="290" y="276" text-anchor="middle">'+esc(xLabel)+'</text>';series.forEach((s,i)=>{const color=['#087f83','#5571c4','#b77623','#a05483'][i%4],ps=s.points.filter(p=>valid(p.x)&&valid(p.y)).sort((a,b)=>a.x-b.x);out+='<polyline class="plot-line" stroke="'+color+'" points="'+ps.map(p=>X(p.x)+','+Y(p.y)).join(' ')+'"/>';ps.forEach(p=>{if(valid(p.low)&&valid(p.high))out+='<line x1="'+X(p.x)+'" x2="'+X(p.x)+'" y1="'+Y(p.low)+'" y2="'+Y(p.high)+'" stroke="'+color+'" opacity=".22" stroke-width="6"/>';out+='<circle cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="4" fill="'+color+'"><title>'+esc(s.label)+'; n='+num(p.x,1)+'; τ='+num(p.y)+'; '+(p.connected!=null?'connected '+pct(p.connected):'')+'</title></circle>';});});return out+'</svg><div class="legend">'+series.map((s,i)=>'<span><span class="dot" style="background:'+['#087f83','#5571c4','#b77623','#a05483'][i%4]+'"></span>'+esc(s.label)+'</span>').join('')+'</div>';
}
function humanDensity(){const type=$('human-density-ref').value,axis=$('human-density-axis').value,block=human?.[type],experiments=(block?.experiments||[]).filter(x=>x.axis===axis);const points=experiments.map(e=>{const k=e.summary?.kendall_tau_b||{},cov=e.coverage||{},field=axis==='subjects'?'subjects':axis.includes('review')?'mean_reviews_per_pair':'pairs';return {x:cov[field]?.mean??e.density,y:k.mean,low:k.p025,high:k.p975,connected:e.connected_fraction}});$('human-density-chart').innerHTML=lineChart([{label:type==='density_self_overlap'?'Self-overlap (optimistic)':'Independent reviewers',points}],axis==='subjects'?'Student contexts':axis.includes('review')?'Mean reviews per pair':'Sampled pairs');$('human-density-table').innerHTML=table(['Density','Mean pairs','Mean τ-b','Connected fits'],experiments.map(e=>[esc(e.density),num(e.coverage?.pairs?.mean,1),num(e.summary?.kendall_tau_b?.mean),pct(e.connected_fraction)]));$('human-density-note').textContent='Human cohort: '+(human?.split||'unavailable')+'. '+(block?.interpretation||'')+' Each point averages valid connected resamples of the fixed 12-model universe. Bars show repeated-subsampling percentile ranges, not independent confidence intervals.';}
function densityGroups(value,path='AI'){if(!value)return [];if(Array.isArray(value)){if(value.some(x=>x?.summary?.kendall_tau_b||x?.metrics?.kendall_tau_b!=null))return [{label:path,experiments:value}];return value.flatMap((v,i)=>densityGroups(v,v?.variant||v?.name||path+' '+(i+1)))}if(typeof value==='object'){if(value.experiments)return [{label:value.label||value.cohort?.replaceAll('_',' ')||path,experiments:value.experiments,note:value.interpretation||value.note},...densityGroups(value.reviewed_only,'Reviewed pairs only'),...densityGroups(value.augmentation,'Add unreviewed pairs')];return Object.entries(value).flatMap(([k,v])=>['fit','reference','reference_fit','human','judge','bootstrap'].includes(k)?[]:densityGroups(v,path+' / '+k))}return []}
function renderAIDensity(){const r=selectedRound(),j=selectedJudge();const raw=j?.density||r?.density?.[j?.variant]||r?.density||D.final.density;let groups=densityGroups(raw,j?judgeName(j):'AI');if(!groups.length){$('ai-density-chart').innerHTML=empty('Completed AI density analysis is not available yet.');$('ai-density-table').innerHTML='';$('ai-density-note').textContent='';return}groups=groups.slice(0,4);$('ai-density-note').textContent='Overall rubric composite for '+(j?judgeName(j):'the selected judge')+' on '+roundLabel(r?.name)+'. Choosing a quality tab changes only the main leaderboard; this density analysis continues to use the overall quality score. Each point averages connected subsample fits against the fixed human reference; ranges describe sampling variability. '+groups.map(g=>g.note||'').filter(Boolean).join(' ');const series=groups.map(g=>({label:g.label,points:g.experiments.map(e=>{const k=e.summary?.kendall_tau_b||e.kendall_tau_b||{},x=pick(e,'n_pairs','pairs','budget','density')??e.coverage?.n_pairs?.mean??e.coverage?.pairs?.mean??e.fraction,y=k.mean??e.metrics?.kendall_tau_b;return{x:typeof x==='object'?x.mean:x,y,low:k.p025??k.low,high:k.p975??k.high,connected:e.connected_fraction,usable:e.valid_fit_fraction}})}));$('ai-density-chart').innerHTML=lineChart(series);$('ai-density-table').innerHTML=table(['Schedule','Pairs','Mean τ-b','Connected fits','Usable fits'],series.flatMap(s=>s.points.map(p=>[esc(s.label),num(p.x,1),num(p.y),pct(p.connected),pct(p.usable)])));}
function audit(){const a=D.audit.summary||{},manual=a.manual_review?.stratified_random;$('audit-stats').innerHTML=stat(integer(a.calls),'Saved calls audited','Existing development sample only')+stat(integer(a.summary?.criteria),'Criterion responses checked','Raw JSON and sign normalization')+stat(integer(manual?.clear_direction_contradictions)+' / '+integer(manual?.reviewed),'Manual direction contradictions','Small stratified random sample')+stat(pct(a.swap?.direction_changed/a.swap?.criteria),'AB / BA direction changes','Includes order and generation variation');$('audit-findings').innerHTML=(D.audit.findings||[]).map(f=>'<div class="card audit-item"><h3>'+esc(f.title)+'</h3><p>'+esc(f.finding)+'</p><p><strong>Judge design implication.</strong> '+esc(f.action)+'</p>'+(f.examples||[]).map(e=>'<details><summary>'+esc(modelName(e.model))+' · raw score '+esc(e.score)+' · '+esc(e.order)+'</summary><p class="mono">'+esc(e.id)+'</p><blockquote>'+esc(e.reason)+'</blockquote><p><strong>Audit note.</strong> '+esc(e.manual_review?.review_note||'')+'</p></details>').join('')+'</div>').join('')||empty('Explanation audit artifacts are not available.');}
function protocol(){const m=D.manifest,p=m.protocol||{};$('protocol-content').innerHTML=table(['Split','Subjects','Human-reviewed pairs','Available pairs'],Object.entries(m.split_subject_counts||{}).map(([k,n])=>[esc(k),integer(n),integer(m.pair_counts?.[k]?.human_reviewed),integer(m.pair_counts?.[k]?.available)]))+'<ul class="protocol-list">'+Object.entries(p).map(([k,v])=>'<li><strong>'+esc(k.replaceAll('_',' '))+'.</strong> '+esc(Array.isArray(v)?v.join('; '):v)+'</li>').join('')+'</ul><p class="small"><strong>Holdout scope.</strong> Test verdicts are excluded from selection and evaluation calculations before the selection lock. This is a computational exclusion rule: the dataset loader can read complete CSV files, so it is not a filesystem access barrier. Students and lesson plans are held out; reviewers and exam questions can recur across splits.</p><p class="mono">Dataset SHA-256: '+esc(m.dataset_sha256||'pending')+'</p>';$('source-links').innerHTML=(D.human_comparator_source?'<a href="'+esc(D.human_comparator_source)+'">Matched human-reference comparison · exact support and reviewer partitions</a>':'')+(D.revision_source?'<a href="'+esc(D.revision_source)+'">Statistical explanation audit · counts, effort contrasts, and interval reconstruction</a>':'')+D.sources.map(p=>'<a href="'+esc(p)+'">'+esc(p)+'</a>').join('')+[...(D.extension?.artifacts||[]),...(D.followup?.artifacts||[])].map(a=>'<a href="'+esc(a.path)+'">'+esc(a.label)+'</a>').join('');$('artifact-details').innerHTML='<details><summary>Final study summary</summary>'+brief(D.final)+'</details><details><summary>Judge configurations</summary>'+brief(D.variants)+'</details>'+D.rounds.map(r=>'<details><summary>'+esc(roundLabel(r.name))+' analysis</summary>'+brief(r)+'</details>').join('');}
$('print-report').addEventListener('click',()=>window.print());overview();readingGuide();studyDesign();setupSelectors();roundResults();sensitivityTables();judgeAblations();setupSystems();extensionResults();followupResults();revisionDetails();humanComparator();setupHumanSelector();audit();contextDetails();densityCaveats();protocol();
</script></body></html>'''
