"""The report must be portable and cannot execute source strings as markup."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from studentbench_leaderboard.study_report import render_report
from studentbench_leaderboard.data import digest


class StudyReportTests(unittest.TestCase):
    @staticmethod
    def _payload(root):
        output = render_report(root).read_text()
        return output, json.loads(output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0])

    def test_pending_report_is_self_contained_and_escapes_saved_text(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audit").mkdir()
            malicious = '</script><script>globalThis.bad = true</script><img src=x onerror="bad()">'
            (root / "audit/findings.json").write_text(json.dumps([{
                "title": malicious, "finding": malicious, "action": "check", "examples": []}]))
            # A pointer outside the containing results directory is not loaded.
            (root / "report_extension.json").write_text(json.dumps({"study_dir": "../../"}))
            output = render_report(root).read_text()
            self.assertNotIn(malicious, output)
            self.assertNotIn('<script src=', output)
            self.assertNotIn('<link ', output)
            payload = output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0]
            parsed = json.loads(payload)
            self.assertEqual(parsed["audit"]["findings"][0]["title"], malicious)
            self.assertEqual(parsed["rounds"], [])
            self.assertIsNone(parsed["extension"])
            self.assertIn("Study in progress", output)

    def test_available_rounds_and_uncertainty_survive_compaction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"splits": {"development": ["d1"], "test": ["t1", "t2"]},
                "pair_catalog": {"test": [{"human_reviewed": True}, {"human_reviewed": False}]}}))
            expected = {"name": "test", "complete": True, "judges": [{"variant": "one", "fit": {"leaderboard": [
                {"model": "author_a", "rank": 1, "score": 0.5}]}, "bootstrap": {"metric_intervals": {
                    "kendall_tau_b": {"estimate": .7, "low": .3, "high": .9, "n_valid": 190}},
                    "rank_intervals": {"judge": {"author_a": {"low": 1, "high": 3}}}},
                "density": {"experiments": [{"fraction": .5, "summary": {"kendall_tau_b": {"mean": .5}}, "runs": [1, 2]}]}}]}
            (root / "test_analysis.json").write_text(json.dumps(expected))
            (root / "human").mkdir()
            cross_split = {"metrics": {"kendall_tau_b": .8}, "interpretation": "Post-lock descriptive comparison"}
            (root / "human/validation_test_human_alignment.json").write_text(json.dumps(cross_split))
            (root / "exports").mkdir()
            (root / "exports/human_test_leaderboard.csv").write_text("model,rank\nauthor_a,1\n")
            # A preview is deliberately not an evaluation-round artifact.
            (root / "round1_preview_analysis.json").write_text(json.dumps({"name": "preview"}))
            output = render_report(root).read_text()
            payload = json.loads(output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0])
            self.assertEqual(len(payload["rounds"]), 1)
            self.assertEqual(payload["rounds"][0]["judges"][0]["bootstrap"], expected["judges"][0]["bootstrap"])
            self.assertNotIn("runs", payload["rounds"][0]["judges"][0]["density"]["experiments"][0])
            self.assertEqual(payload["manifest"]["split_subject_counts"]["test"], 2)
            self.assertEqual(payload["manifest"]["pair_counts"]["test"], {"available": 2, "human_reviewed": 1})
            self.assertEqual(payload["human_cross_split"], cross_split)
            self.assertIn("exports/human_test_leaderboard.csv", payload["sources"])
            self.assertNotIn("exports/selected_ai_dense_leaderboard.csv", payload["sources"])

    def test_context_and_frozen_selection_are_displayed_without_bank_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "context").mkdir()
            config = root / "configs/enhanced_luna_high"
            config.mkdir(parents=True)
            (root / "variants.json").write_text(json.dumps([{
                "variant": "enhanced_luna_high", "judge_id": "judge_5", "model": "openai/gpt-5.6-luna",
                "reasoning_effort": "high", "response_style": "evidence_winner"}]))
            (config / "judge_5.toml").write_text('judge_id="judge_5"\nstrategy="single"\nplan_projection="human_display"\n')
            selection = {"variant": "enhanced_luna_high_balanced", "members": [["enhanced_luna_high", 0], ["enhanced_luna_high", 1]], "aggregation": "soft"}
            (root / "selection.json").write_text(json.dumps(selection))
            coverage = {"unique_qc_missing_before": 13, "unique_qc_missing_after": 0,
                        "unique_chart_stimuli_recovered": 6, "unique_figures_unrendered": 8,
                        "provenance": {"source_bank_sha256": "pinned", "source_type": "application_supplement"}}
            (root / "context/coverage.json").write_text(json.dumps(coverage))
            (root / "context/quant_application_bank.json").write_text('{"raw_assessment_stem":"not embedded"}')
            output = render_report(root).read_text()
            payload = json.loads(output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0])
            self.assertEqual(payload["selection"], selection)
            self.assertEqual(payload["context"]["coverage"], coverage)
            self.assertEqual(payload["configurations"]["enhanced_luna_high"]["plan_projection"], "human_display")
            self.assertIn("context/coverage.json", payload["sources"])
            self.assertNotIn("context/quant_application_bank.json", payload["sources"])
            self.assertNotIn("raw_assessment_stem", output)
            self.assertIsNone(payload["input_ablation"])

    def test_input_ablation_requires_the_same_pairs_not_just_the_same_count(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"round1_validation_pairs": ["one", "two"]}))
            variants = ("luna_high", "terra_high", "enhanced_luna_high", "enhanced_terra_high")
            jobs = [{"variant": variant, "pair_id": pair} for variant in variants for pair in ("three", "four")]
            (root / "input_ablation_jobs.json").write_text(json.dumps(jobs))
            (root / "input_ablation_analysis.json").write_text(json.dumps({"n_pairs": 2, "jobs_sha256": digest(jobs)}))
            output = render_report(root).read_text()
            payload = json.loads(output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0])
            self.assertFalse(payload["input_ablation_cohort"]["verified"])
            self.assertTrue(payload["input_ablation_cohort"]["jobs_hash_matches"])
            jobs = [{"variant": variant, "pair_id": pair} for variant in variants for pair in ("one", "two")]
            (root / "input_ablation_jobs.json").write_text(json.dumps(jobs))
            (root / "input_ablation_analysis.json").write_text(json.dumps({"n_pairs": 2, "jobs_sha256": digest(jobs)}))
            output = render_report(root).read_text()
            payload = json.loads(output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0])
            self.assertTrue(payload["input_ablation_cohort"]["verified"])

    def test_catalog_uses_saved_prompts_and_preserves_panel_weights_without_request_payloads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calls").mkdir()
            names = ["luna_low", "terra_low", "sol_low", "astra_low"]
            variants = []
            saved_prompt = 'Exact snapshot </script><script>bad()</script>'
            for name in names:
                variants.append({"variant": name, "judge_id": "judge_5", "response_style": "evidence_winner"})
                config = root / "configs" / name
                config.mkdir(parents=True)
                (config / "system.txt").write_text("Later local text")
                (config / "user.txt").write_text("$payload")
                (config / "judge_5.toml").write_text('judge_id="judge_5"\nstrategy="single"\n[prompts]\nsystem="system.txt"\nuser="user.txt"\n[[members]]\nname="member"\nmodel="openai/' + name + '"\nreasoning_effort="low"\n')
                (root / "calls" / (name + ".json")).write_text(json.dumps({
                    "variant": name, "judge": {"version": "frozen", "system_prompt": saved_prompt, "user_template": "$payload"},
                    "response_schema": {"type": "object", "properties": {"q01": {"type": "string"}}},
                    "parser_contract": {"version": "lossless.v1"}, "parser_version": "lossless.v1",
                    "transformations": ["outer_json_fence"], "provider_schema_compliant": False,
                    "messages": [{"content": "PRIVATE_EVALUATION_PAYLOAD"}],
                    "transport": {"requested": {"headers": "PRIVATE_AUTH_HEADER", "body": {"model": "provider/model", "messages": "PRIVATE_EVALUATION_PAYLOAD", "reasoning": {"effort": "high"}}},
                                  "effective": {"response_format": "json_object", "body": {"max_tokens": 32768}}},
                    "requested_response_format": "json_schema", "effective_response_format": "json_object"}))
            (root / "variants.json").write_text(json.dumps(variants))
            # Historical descriptive recipes use the earlier non-executable layout.
            (root / "derived_judges.toml").write_text('[judge_8]\nstrategy="panel_soft"\n')
            shares = [.5, 1 / 6, 1 / 6, 1 / 6]
            (root / "round1_analysis.json").write_text(json.dumps({"name": "round1", "judges": [{
                "variant": "panel_low", "members": [[name, 0] for name in names], "aggregation": "soft",
                "weighted_members": [{"variant": name, "draw": 0, "weight": share, "normalized_weight": share,
                                      "family": "a" if index == 0 else "b"} for index, (name, share) in enumerate(zip(names, shares))],
                "derived_definition": {"aggregation": "weighted_mean"}}]}))
            output, payload = self._payload(root)
            catalog = {entry["key"]: entry for entry in payload["judge_catalog"]}
            self.assertEqual(catalog["luna_low"]["system_prompt"], saved_prompt)
            self.assertFalse(catalog["luna_low"]["prompt_snapshot_matches"])
            self.assertIn("q01", catalog["luna_low"]["output_schema"]["properties"])
            self.assertEqual([member["normalized_weight"] for member in catalog["panel_low"]["member_calls"]], shares)
            self.assertEqual(catalog["panel_low"]["member_calls"][0]["catalog_key"], "luna_low")
            self.assertEqual(catalog["luna_low"]["transport"]["effective"]["body_settings"]["max_tokens"], 32768)
            self.assertFalse(catalog["luna_low"]["provider_schema_compliant"])
            self.assertEqual(catalog["luna_low"]["parser_version"], "lossless.v1")
            self.assertNotIn("PRIVATE_EVALUATION_PAYLOAD", output)
            self.assertNotIn("PRIVATE_AUTH_HEADER", output)
            self.assertNotIn(saved_prompt, output)

    def test_extension_stays_separate_and_example_display_removes_identity_metadata(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            root, extension = parent / "original", parent / "extension"
            root.mkdir(); extension.mkdir()
            (root / "report_extension.json").write_text(json.dumps({"study_dir": "../extension"}))
            (extension / "variants.json").write_text(json.dumps([{"variant": "example_judge", "judge_id": "judge_5"}]))
            (extension / "report_metadata.json").write_text(json.dumps({"analysis_files": ["validation_analysis.json"],
                "protocol_note": "Validation after test exposure", "examples": {"example_judge": {
                    "status": "human_graded", "n_examples": 1, "source_split": "development", "artifact_file": "example.json"}}}))
            (extension / "validation_analysis.json").write_text(json.dumps({"name": "new_validation", "split": "validation", "judges": []}))
            (extension / "summary.json").write_text(json.dumps({"availability": {"example_judge": {
                "status": "unavailable", "reason": "Preflight endpoint unavailable"}}}))
            (extension / "derived_judges.toml").write_text('schema_version=1\n[[judges]]\njudge_id="judge_10"\nvariant="two_draws"\nlabel="Two draws"\naggregation="soft"\n[[judges.members]]\nvariant="example_judge"\ndraw=0\nweight=1\n[[judges.members]]\nvariant="example_judge"\ndraw=1\nweight=2\n')
            (extension / "example.json").write_text(json.dumps({"author_model": "PRIVATE_AUTHOR", "example_input": {
                "plan_ids": ["PRIVATE_PLAN_ID"], "plans": {"A": {"text": "Example plan"}}},
                "human_reviews": [{"reviewer_id": "PRIVATE_REVIEWER", "preference": 1}]}))
            paired = {"baseline_variant": "cached", "comparisons": {"example_judge": {
                "role": "primary_example", "metrics": {"kendall_tau_b": {"delta": .1, "low": -.2, "high": .3}}}}}
            (extension / "paired_deltas.json").write_text(json.dumps(paired))
            (extension / "source_snapshot.json").write_text('{"files": []}')
            (extension / "source_snapshot.zip").write_bytes(b"local archive fixture")
            (extension / "exports").mkdir()
            (extension / "exports/example_validation_judges.csv").write_text("variant,tau\nexample_judge,0.5\n")
            (parent / "outside.csv").write_text("private,outside\n")
            (extension / "exports/unsafe.csv").symlink_to(parent / "outside.csv")
            (extension / "example_audit.json").write_text(json.dumps({
                "sample": {"n_pairs": 6, "pair_ids": ["PRIVATE_AUDIT_PAIR"]},
                "summary": {"same": 30}, "copying_check": {"result": "No flags", "pairs": ["PRIVATE_AUDIT_PAIR"]},
                "findings": [{"pair_number": 1, "pair_id": "PRIVATE_AUDIT_PAIR", "title": "Illustration",
                              "snippets": [{"source": "baseline", "text": "Exact saved explanation", "pair_id": "PRIVATE_AUDIT_PAIR"}]}]}))
            output, payload = self._payload(root)
            self.assertEqual(payload["rounds"], [])
            self.assertEqual(payload["extension"]["rounds"][0]["name"], "new_validation")
            self.assertEqual(payload["extension"]["paired_deltas"], paired)
            artifacts = {item["path"]: item["label"] for item in payload["extension"]["artifacts"]}
            for filename in ("summary.json", "source_snapshot.json", "source_snapshot.zip", "paired_deltas.json",
                             "exports/example_validation_judges.csv"):
                self.assertTrue(artifacts["../extension/" + filename].startswith("Extension · "))
            self.assertNotIn("../extension/verification.json", artifacts)
            self.assertNotIn("../extension/exports/unsafe.csv", artifacts)
            self.assertEqual(payload["extension"]["example_audit"]["sample"], {"n_pairs": 6})
            self.assertEqual(payload["extension"]["example_audit"]["findings"][0]["snippets"],
                             [{"source": "baseline", "text": "Exact saved explanation"}])
            example = payload["judge_catalog"][0]["examples"]
            self.assertEqual(payload["judge_catalog"][0]["key"], "extension:example_judge")
            self.assertEqual(example["source_split"], "development")
            self.assertEqual(example["display_artifact"]["human_reviews"], [{"preference": 1}])
            panel = next(entry for entry in payload["judge_catalog"] if entry["key"] == "extension:two_draws")
            self.assertEqual(panel["evaluated_rounds"], [])
            self.assertEqual(panel["availability"]["status"], "unavailable")
            self.assertEqual([m["normalized_weight"] for m in panel["member_calls"]], [1 / 3, 2 / 3])
            for private in ("PRIVATE_AUTHOR", "PRIVATE_PLAN_ID", "PRIVATE_REVIEWER", "PRIVATE_AUDIT_PAIR"):
                self.assertNotIn(private, output)

    def test_terminal_missing_requests_preserve_endpoint_status_and_scheduled_reference(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            root, extension = parent / "original", parent / "extension"
            root.mkdir(); extension.mkdir()
            (root / "report_extension.json").write_text(json.dumps({"study_dir": "../extension"}))
            (extension / "report_metadata.json").write_text(json.dumps({"analysis_files": ["provider_validation_analysis.json"]}))
            (extension / "variants.json").write_text(json.dumps([{"variant": "provider", "judge_id": "judge_5"}]))
            summary = {"status": "complete_with_terminal_missing_outcomes",
                       "counts": {"scheduled_calls": 120, "completed_calls": 119,
                                  "terminal_failed_calls": 1, "missing_outcomes": 1},
                       "availability": {"provider": {"status": "available", "terminal_failed_calls": 1}},
                       "terminal_failures": {"n_jobs": 1, "by_variant": {"provider": 1},
                                             "details": [{"variant": "provider", "n_failed_api_attempts": 3,
                                                          "outcome": "missing", "rating_imputed": False}]},
                       "execution_interruptions": {"n_events": 1, "n_unknown_events": 1,
                           "events": [{"signal_sent": True, "in_flight": {"charges_usd": None,
                                       "local_api_calls_upper_bound": 24}}]}}
            (extension / "summary.json").write_text(json.dumps(summary))
            analysis = {"name": "provider_validation", "n_pairs": 120, "complete": False,
                        "human": {"n_comparisons": 120}, "judges": [{"variant": "provider",
                            "n_calls": 119, "terminal_failed_calls": 1,
                            "execution_status": "completed_with_terminal_failures",
                            "coverage": {"missing_calls": 1, "criterion_coverage": 119 / 120}}]}
            (extension / "provider_validation_analysis.json").write_text(json.dumps(analysis))
            _, payload = self._payload(root)
            self.assertEqual(payload["extension"]["summary"], summary)
            self.assertEqual(payload["extension"]["rounds"][0]["human"]["n_comparisons"], 120)
            self.assertEqual(payload["extension"]["rounds"][0]["judges"][0]["coverage"]["missing_calls"], 1)
            self.assertEqual(payload["judge_catalog"][0]["availability"]["status"], "available")
            self.assertEqual(payload["judge_catalog"][0]["availability"]["terminal_failed_calls"], 1)


    def test_followup_keeps_duplicate_variant_names_scoped_to_their_study(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            original, extension, followup = (parent / name for name in ("original", "extension", "followup"))
            original.mkdir()
            (original / "report_extension.json").write_text(json.dumps({
                "study_dir": "../extension", "followup_study_dir": "../followup",
                "revision_statistics_file": "../statistics.json"}))
            for folder, prompt, cost in ((extension, "Original provider instructions", 3),
                                         (followup, "Separate follow-up instructions", 7)):
                config = folder / "configs/shared"
                config.mkdir(parents=True)
                (config / "system.txt").write_text(prompt)
                (config / "judge_5.toml").write_text(
                    'judge_id="judge_5"\nstrategy="single"\n[prompts]\nsystem="system.txt"\n')
                (folder / "variants.json").write_text(json.dumps([{"variant": "shared", "judge_id": "judge_5"}]))
                (folder / "report_metadata.json").write_text(json.dumps({"analysis_files": ["validation_analysis.json"]}))
                (folder / "validation_analysis.json").write_text(json.dumps({
                    "name": folder.name, "judges": [{"variant": "shared", "estimated_usd": cost}]}))
                (folder / "summary.json").write_text(json.dumps({"counts": {"completed_calls": cost}}))
            statistics = {"reasoning_effort": {"n_bootstrap": 500, "contrasts": {
                "luna": {"metrics": {"kendall_tau_b": {"delta": -.1, "draws": [1, 2]}}}}}}
            (parent / "statistics.json").write_text(json.dumps(statistics))
            _, payload = self._payload(original)
            catalog = {entry["key"]: entry for entry in payload["judge_catalog"]}
            self.assertEqual(catalog["extension:shared"]["system_prompt"], "Original provider instructions")
            self.assertEqual(catalog["followup:shared"]["system_prompt"], "Separate follow-up instructions")
            self.assertEqual(catalog["followup:shared"]["group"], "Focused follow-up")
            self.assertEqual(payload["extension"]["rounds"][0]["judges"][0]["estimated_usd"], 3)
            self.assertEqual(payload["followup"]["rounds"][0]["judges"][0]["estimated_usd"], 7)
            self.assertEqual(payload["revision_source"], "../statistics.json")
            self.assertNotIn("draws", payload["revision"]["reasoning_effort"]["contrasts"]["luna"]["metrics"]["kendall_tau_b"])
            self.assertEqual(json.loads((parent / "statistics.json").read_text()), statistics)

    def test_followup_and_revision_pointers_cannot_escape_results_directory(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "results/original"
            original.mkdir(parents=True)
            outside = parent / "outside"
            outside.mkdir()
            (outside / "report_metadata.json").write_text('{"private": "DO_NOT_LOAD"}')
            (outside / "statistics.json").write_text('{"private": "DO_NOT_LOAD"}')
            (original / "report_extension.json").write_text(json.dumps({
                "followup_study_dir": "../../outside",
                "revision_statistics_file": "../../outside/statistics.json"}))
            output, payload = self._payload(original)
            self.assertIsNone(payload["followup"])
            self.assertIsNone(payload["revision"])
            self.assertNotIn("DO_NOT_LOAD", output)


if __name__ == "__main__":
    unittest.main()
