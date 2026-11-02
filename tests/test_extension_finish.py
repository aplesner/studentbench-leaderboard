"""Finalization must reconcile costs and hash the artifacts it actually writes."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/ranking-extension/finish.py"
SPEC = importlib.util.spec_from_file_location("ranking_extension_finish_tests", SOURCE)
finish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finish)


def byok(amount):
    return {"transport": {"adapter": "openrouter-openai-sdk"},
            "usage": {"cost": 0, "is_byok": True,
                      "cost_details": {"upstream_inference_cost": amount}}}


class FinishIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.study.mkdir()

    def write(self, name, value):
        path = self.study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_request_and_result_are_one_diagnostic_with_byok_upstream_cost(self):
        self.write("diagnostics/one_probe.request.json", {"purpose": "routing only"})
        self.write("diagnostics/one_probe.json", byok(.2))
        self.write("diagnostics/endpoints.json", {"metadata": "not an executed call"})
        records, reconciliation = finish.diagnostic_attempts(self.study)
        self.assertEqual(len(records), 1)
        self.assertEqual(reconciliation["attempts"], 1)
        self.assertEqual(reconciliation["requests_without_saved_results"], [])
        self.assertAlmostEqual(finish.costs(records)["known_economic_usd"], .2)
        self.assertEqual(finish.costs(records)["router_charge_usd"], 0)

    def test_unsaved_diagnostic_response_stays_unknown_and_is_not_silently_dropped(self):
        self.write("diagnostics/first_probe.request.json", {"purpose": "routing only"})
        self.write("diagnostics/first_probe.json", byok(.2))
        self.write("diagnostics/second_probe.request.json", {"purpose": "routing only"})
        records, reconciliation = finish.diagnostic_attempts(self.study)
        self.assertEqual(len(records), 2)
        self.assertEqual(reconciliation["recorded_requests"], 2)
        self.assertEqual(reconciliation["saved_results"], 1)
        self.assertEqual(reconciliation["requests_without_saved_results"],
                         ["diagnostics/second_probe.request.json"])
        self.assertEqual(finish.costs(records)["n_unknown"], 1)
        self.assertIsNone(finish.economic_total(records)["economic_usd"])
        self.assertAlmostEqual(finish.costs(records)["known_economic_usd"], .2)

    def test_rejected_interruption_proposals_do_not_create_unknown_costs(self):
        self.write("execution_interruptions.json", {"interruptions": [
            {"signal_sent": False, "status": "blocked-by-automatic-approval-review",
             "in_flight": {"charges_usd": None}}]})
        interruption = finish.interruption_costs(self.study)
        self.assertEqual(interruption["n_events"], 0)
        self.assertEqual(interruption["ignored_unsent_events"], 1)
        self.assertEqual(interruption["n_unknown_events"], 0)
        total = finish.economic_total([{"estimated_usd": .2}], interruption)
        self.assertTrue(total["complete"])
        self.assertEqual(total["economic_usd"], .2)

    def test_actual_interruption_preserves_unknown_count_bounds_and_incomplete_total(self):
        models = {"openrouter/anthropic/claude-opus-5": 8,
                  "openrouter/anthropic/claude-sonnet-5": 8,
                  "openrouter/deepseek/deepseek-v4.1-flash": 8}
        self.write("execution_interruptions.json", {"interruptions": [
            {"signal_sent": False, "status": "rejected", "in_flight": {"charges_usd": None}},
            {"interruption_id": "approved-stop", "signal_sent": True, "target_pid": 123,
             "status": "terminated-and-success-hashes-verified", "target_run_id": "batch3",
             "before": {"missing_scheduled_jobs": 357},
             "in_flight": {"confirmed_interrupted_api_request_count": None,
                           "local_api_calls_lower_bound": 0, "local_api_calls_upper_bound": 24,
                           "local_api_calls_upper_bound_by_model": models,
                           "charges_usd": None, "billing_reconciliation_status": "unresolved"}}]})
        interruption = finish.interruption_costs(self.study)
        self.assertEqual(interruption["n_events"], 1)
        self.assertEqual(interruption["n_unknown_events"], 1)
        self.assertIsNone(interruption["confirmed_interrupted_api_request_count"])
        self.assertEqual(interruption["local_api_calls_lower_bound"], 0)
        self.assertEqual(interruption["local_api_calls_upper_bound"], 24)
        self.assertEqual(interruption["local_api_calls_upper_bound_by_model"], models)
        self.assertNotIn("before", interruption["events"][0])
        total = finish.economic_total([byok(.2)], interruption)
        self.assertFalse(total["complete"])
        self.assertIsNone(total["economic_usd"])
        self.assertEqual(total["known_economic_usd"], .2)
        self.assertEqual(total["interruption_events_with_unknown_billing"], 1)

    def test_main_excludes_controls_deduplicates_recoveries_and_hashes_new_paired_output(self):
        baseline, provider, example = "enhanced_terra_high", "provider", "terra_high_one_example"
        jobs = {name: {"variant": name, "pair_id": "p", "draw": 0} for name in (baseline, provider, example)}
        self.write("provider_jobs.json", [jobs[baseline], jobs[provider]])
        self.write("example_jobs.json", [jobs[baseline], jobs[example]])
        self.write("availability.json", {"exclusions": {}})
        self.write("variants.json", [{"variant": name} for name in jobs])
        self.write("manifest.json", {"original_selection_sha256": "locked"})
        self.write("imported_calls.json", {"calls": {f"{baseline}--p--0.json": {"sha256": "verified"}}})
        self.write(f"calls/{baseline}--p--0.json", {**jobs[baseline], "estimated_usd": 1.0})
        self.write(f"calls/{example}--p--0.json", {**jobs[example], "estimated_usd": .1})
        self.write(f"calls/{provider}--p--0.json", {**jobs[provider], **byok(.2),
                                                  "recovered_from_error": {"file": "errors/recovered.json"}})
        self.write("errors/recovered.json", {**jobs[provider], **byok(.2)})
        self.write("errors/extra.json", {**jobs[provider], **byok(.05)})
        self.write("diagnostics/unfinished_probe.request.json", {"purpose": "routing only"})
        self.write("execution_interruptions.json", {"signal_sent": False, "status": "approval rejected"})
        self.write("paired_deltas.json", {"status": "older phase"})
        self.write("example_audit.json", {"n_pairs": 6})
        self.write("source_snapshot.json", {"archive": "source_snapshot.zip"})
        (self.study / "source_snapshot.zip").write_bytes(b"snapshot fixture bytes")
        self.write("derived_judges.toml", {"recipe": "hash only in this isolated fixture"})
        original = self.root / "original"
        original.mkdir()
        (original / "manifest.json").write_text(json.dumps({"selection_sha256": "locked"}))

        def analysis(study, name, schedule, **kwargs):
            panel = {"variant": "panel", "label": "Panel", "members": [[baseline, 0], [provider, 0]],
                     "estimated_usd": 0, "metrics": {}, "fit": {"leaderboard": []},
                     "coverage": {"n_calls": 2, "missing_calls": 0, "criterion_coverage": 1}}
            return {"name": name, "interpretation": "fixture", "human": {"leaderboard": []},
                    "human_full": {"leaderboard": []}, "judges": [panel] if name == "provider_validation" else []}

        def paired(study, **kwargs):
            return self.write("paired_deltas.json", {"status": "new complete paired output"})

        with patch.object(finish, "ORIGINAL", original), \
                patch.object(finish, "run_jobs", return_value={"reused": 3, "pending": 0}) as runner, \
                patch.object(finish, "analyze_round", side_effect=analysis), \
                patch.object(finish, "render_report"), \
                patch("studentbench_leaderboard.paired_deltas.run_paired_deltas", side_effect=paired), \
                redirect_stdout(io.StringIO()):
            finish.main(self.study, bootstrap=500)
        self.assertTrue(runner.call_args.kwargs["check_only"])
        summary = json.loads((self.study / "summary.json").read_text())
        self.assertEqual(summary["counts"]["scheduled_calls"], 3)
        self.assertEqual(summary["counts"]["imported_controls"], 1)
        self.assertEqual(summary["counts"]["recovered_error_attempts"], 1)
        self.assertEqual(summary["counts"]["routing_diagnostic_unsaved_results"], 1)
        self.assertAlmostEqual(summary["estimated_usd"], .2 + .1 + .05)
        self.assertAlmostEqual(summary["costs"]["imported_controls"]["known_economic_usd"], 1)
        self.assertEqual(summary["costs"]["failed_attempts"]["n_records"], 1)
        self.assertIsNone(summary["costs_total"]["economic_usd"])
        self.assertEqual(summary["costs"]["routing_diagnostics"]["n_unknown"], 1)
        self.assertEqual(summary["counts"]["actual_execution_interruptions"], 0)
        self.assertEqual(summary["costs"]["interrupted_in_flight_requests"]["n_unknown_events"], 0)
        provider_analysis = json.loads((self.study / "provider_validation_analysis.json").read_text())
        self.assertAlmostEqual(provider_analysis["judges"][0]["estimated_usd"], 1.2)
        for name in ("paired_deltas.json", "example_audit.json", "derived_judges.toml",
                     "execution_interruptions.json", "source_snapshot.json", "source_snapshot.zip",
                     "exports/provider_validation_judges.csv"):
            self.assertEqual(summary["artifact_sha256"][name],
                             hashlib.sha256((self.study / name).read_bytes()).hexdigest())
        self.assertTrue(summary["original_selection_unchanged"])

    def test_terminal_completion_gate_is_distinct_from_availability_and_preserves_missing_judge_results(self):
        failed = {"variant": "provider", "pair_id": "p", "draw": 0}
        blocked = {"variant": "muse", "pair_id": "p", "draw": 0}
        self.write("availability.json", {"exclusions": {"muse": {"reason": "Account policy"}}})
        key = ("provider", "p", 0)
        declaration = {**failed, "n_failed_api_attempts": 3, "outcome": "missing", "rating_imputed": False}
        with patch.object(finish, "load_terminal_failures", return_value={}):
            with self.assertRaisesRegex(ValueError, "Incomplete executable schedule"):
                finish.execution_completion(self.study, [failed, blocked])
        with patch.object(finish, "load_terminal_failures", return_value={key: declaration}):
            excluded, terminal, missing, unavailable = finish.execution_completion(self.study, [failed, blocked])
        self.assertEqual(set(excluded), {"muse"})
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(len(missing), 2)
        self.assertEqual(len(terminal), 1)
        judge = {"variant": "provider", "label": "Provider", "members": [["provider", 0]],
                 "estimated_usd": 0, "metrics": {}, "fit": {"leaderboard": []},
                 "coverage": {"n_calls": 0, "missing_calls": 1, "criterion_coverage": 0}}
        analysis = {"name": "provider_validation", "interpretation": "fixture", "human": {"leaderboard": []},
                    "human_full": {"leaderboard": []}, "judges": [judge]}
        actual = finish.enrich_and_export(self.study, analysis, [], terminal)["judges"][0]
        self.assertEqual(actual["terminal_failed_calls"], 1)
        self.assertEqual(actual["execution_status"], "terminal_failure")
        self.assertEqual(actual["coverage"]["missing_calls"], 1)
        self.assertEqual(actual["fit"]["leaderboard"], [])

    def test_unavailable_muse_precedes_terminal_failure_in_nine_member_panel_status(self):
        self.write("availability.json", {"exclusions": {"muse": {"reason": "Account policy"}}})
        successful = [f"member_{i}" for i in range(7)]
        members = successful + ["muse", "glm"]
        calls = [{"variant": name, "pair_id": "p", "draw": 0, "estimated_usd": .1} for name in successful]
        terminal = {("glm", "p", 0): {"n_failed_api_attempts": 3}}
        coverage = {"n_calls": 7, "missing_calls": 2, "criterion_coverage": 0, "scored_criteria": 0}
        judge = {"variant": "nine_panel", "label": "Nine required members",
                 "members": [[name, 0] for name in members], "estimated_usd": .7,
                 "metrics": {"kendall_tau_b": None}, "fit": {"leaderboard": []}, "coverage": dict(coverage)}
        analysis = {"name": "provider_validation", "interpretation": "fixture", "human": {"leaderboard": []},
                    "human_full": {"leaderboard": []}, "judges": [judge]}
        actual = finish.enrich_and_export(self.study, analysis, calls, terminal)["judges"][0]
        self.assertEqual(actual["terminal_failed_calls"], 1)
        self.assertEqual(actual["unavailable_member_variants"], ["muse"])
        self.assertEqual(actual["execution_status"], "unavailable")
        self.assertEqual(actual["coverage"], coverage)
        self.assertIsNone(actual["metrics"]["kendall_tau_b"])
        self.assertEqual(actual["fit"]["leaderboard"], [])
        with (self.study / "exports/provider_validation_judges.csv").open() as stream:
            import csv
            exported = next(csv.DictReader(stream))
        self.assertEqual(exported["execution_status"], "unavailable")
        self.assertEqual(exported["unavailable_member_variants"], "muse")

    def test_terminal_completion_status_requires_every_missing_call_to_be_declared(self):
        calls = [{"variant": "glm", "pair_id": "done", "draw": 0, "estimated_usd": .1}]
        for terminal_count, expected in ((1, "incomplete"), (2, "completed_with_terminal_failures")):
            with self.subTest(terminal_count=terminal_count):
                terminal = {("glm", f"failed_{i}", 0): {} for i in range(terminal_count)}
                coverage = {"n_calls": 1, "missing_calls": 2, "criterion_coverage": 1 / 3}
                judge = {"variant": "glm", "label": "GLM", "members": [["glm", 0]],
                         "estimated_usd": .1, "metrics": {}, "fit": {"leaderboard": []},
                         "coverage": dict(coverage)}
                analysis = {"name": "provider_validation", "interpretation": "fixture", "human": {"leaderboard": []},
                            "human_full": {"leaderboard": []}, "judges": [judge]}
                actual = finish.enrich_and_export(self.study, analysis, calls, terminal)["judges"][0]
                self.assertEqual(actual["execution_status"], expected)
                self.assertEqual(actual["unavailable_member_variants"], [])
                self.assertEqual(actual["coverage"], coverage)


if __name__ == "__main__":
    unittest.main()
