"""Exercise actual strategies through the public API using synthetic provider responses."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from studentbench_leaderboard import ComparisonContext, compare_plans
from studentbench_leaderboard.cli import parser, records_from, run
from studentbench_leaderboard.comparison import ConfiguredJudge, request_id
from studentbench_leaderboard.data import Dataset
from studentbench_leaderboard.judge import LiteLLMJudge, evaluate, make_task
from studentbench_leaderboard.llm_client import LLMClient
from studentbench_leaderboard.judge_config import DEFAULT_JUDGES_DIR, check_strategy_history, load_definition
from studentbench_leaderboard.report import judge_report
from test_pipeline import fixture


class ConfiguredJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.configs = self.root / "judges"
        shutil.copytree(DEFAULT_JUDGES_DIR, self.configs)
        fixture(self.root / "ssot")
        self.dataset = Dataset(self.root / "ssot", "gre_quant")
        self.task = make_task(self.dataset, "g1")
        self.context = ComparisonContext(self.task.instrument, self.task.subject, self.task.pretest,
                                         self.task.rubric, tuple(self.task.plan_ids), self.task.dataset_sha256)

    def definition(self, judge_id):
        return load_definition(judge_id, self.configs)

    def response(self, task, score):
        return json.dumps({q["id"]: {"score": score, "reason": "Synthetic evidence"} for q in task.rubric}), {
            "usage": {"total_tokens": 10}, "response_id": "synthetic-call", "response_model": "synthetic"}

    def compare(self, judge_id):
        return compare_plans(*self.task.plans, judge_id, context=self.context, judges_dir=self.configs)

    def test_public_interface_single_call_preserves_evidence_and_result_provenance(self):
        with patch.object(LiteLLMJudge, "complete", autospec=True,
                          side_effect=lambda adapter, task: self.response(task, 2)) as call:
            result = self.compare("judge_1")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["judge_id"], "judge_1")
        self.assertEqual(result["judge_version"], self.definition("judge_1").version)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["ratings"]["format_old"]["score"], 2)
        self.assertEqual(result["calls"][0]["usage"]["total_tokens"], 10)
        self.assertNotIn("SECRET", json.dumps(result["calls"][0]["messages"]))
        self.assertEqual(result["request_id"], request_id(self.task, ConfiguredJudge(self.definition("judge_1"))))
        self.assertEqual(result["prompt_sha256"], ConfiguredJudge(self.definition("judge_1")).preview(self.task)["prompt_sha256"])

    def test_order_check_normalizes_swapped_winner(self):
        def stable(adapter, task):
            return self.response(task, 2 if task.plan_ids == self.task.plan_ids else -2)

        with patch.object(LiteLLMJudge, "complete", autospec=True, side_effect=stable):
            result = self.compare("judge_2")
        self.assertEqual([c["order"] for c in result["calls"]], ["AB", "BA"])
        self.assertEqual([c["ratings"]["format_old"]["score"] for c in result["calls"]], [2, -2])
        self.assertEqual(result["diagnostics"]["format_old"]["normalized_scores"], [2, 2])
        self.assertFalse(result["diagnostics"]["format_old"]["direction_changed"])
        self.assertEqual(result["ratings"]["format_old"]["score"], 2)
        report = judge_report(self.dataset, [result])
        metric = next(r for r in report["judges"][0]["agreement"] if r["question_variant_id"] == "format_old")
        self.assertEqual(metric["direction_agreement_against_human_mean"], 1)

    def test_order_bias_is_flagged_and_not_reported_as_a_tie(self):
        with patch.object(LiteLLMJudge, "complete", autospec=True,
                          side_effect=lambda adapter, task: self.response(task, 1)):
            result = self.compare("judge_2")
        self.assertTrue(result["diagnostics"]["format_old"]["direction_changed"])
        self.assertIsNone(result["ratings"]["format_old"]["score"])

    def test_order_check_reports_strength_change_with_consistent_winner(self):
        with patch.object(LiteLLMJudge, "complete", autospec=True,
                          side_effect=lambda adapter, task: self.response(
                              task, 1 if task.plan_ids == self.task.plan_ids else -2)):
            result = self.compare("judge_2")
        self.assertTrue(result["diagnostics"]["format_old"]["score_changed"])
        self.assertFalse(result["diagnostics"]["format_old"]["direction_changed"])
        self.assertEqual(result["ratings"]["format_old"]["score"], 1)

    def test_randomized_votes_are_normalized_and_use_strict_majority(self):
        votes = iter([1, 2, 2, -2, 0])

        def response(adapter, task):
            score = next(votes)
            return self.response(task, score if task.plan_ids == self.task.plan_ids else -score)

        judge = ConfiguredJudge(self.definition("judge_3"))
        self.assertEqual(judge.schedule(self.task), judge.schedule(self.task))
        self.assertEqual(len(judge.schedule(self.task)), 5)
        with patch.object(LiteLLMJudge, "complete", autospec=True, side_effect=response) as calls:
            result = self.compare("judge_3")
        self.assertEqual(calls.call_count, 5)
        self.assertEqual(result["diagnostics"]["format_old"]["votes"], {"A": 3, "B": 1, "tie": 1, "abstain": 0})
        self.assertEqual(result["ratings"]["format_old"]["score"], 2)

    def test_abstentions_do_not_create_a_majority_and_ties_are_explicit_votes(self):
        for scores, expected in (([2, 2, None, None, None], None), ([0, 0, 0, 1, -1], 0)):
            votes = iter(scores)

            def response(adapter, task):
                score = next(votes)
                return self.response(task, -score if score is not None and task.plan_ids != self.task.plan_ids else score)

            with self.subTest(scores=scores), patch.object(LiteLLMJudge, "complete", autospec=True, side_effect=response):
                result = self.compare("judge_3")
                self.assertEqual(result["ratings"]["format_old"]["score"], expected)

    def test_panel_has_one_vote_per_model_and_records_member_models(self):
        def response(adapter, task):
            return self.response(task, -1 if adapter.model.startswith("gemini/") else 1)

        with patch.object(LiteLLMJudge, "complete", autospec=True, side_effect=response) as calls:
            result = self.compare("judge_4")
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(len({c["member"]["model"] for c in result["calls"]}), 3)
        self.assertEqual(result["ratings"]["format_old"]["score"], 1)

    def test_prompt_model_and_effort_changes_resolve_new_subversions(self):
        original = self.definition("judge_1")
        changes = [replace(original, system_prompt=original.system_prompt + "\nMore detail."),
                   replace(original, user_template="New framing\n$payload"),
                   replace(original, members=(replace(original.members[0], model="another/model"),)),
                   replace(original, members=(replace(original.members[0], reasoning_effort="high"),))]
        for changed in changes:
            with self.subTest(version=changed.version):
                self.assertNotEqual(original.version, changed.version)
                self.assertEqual(original.structure(), changed.structure())
        prompt = self.configs / "prompts/pairwise_system.txt"
        prompt.write_text(prompt.read_text() + "\nAdditional instruction.\n")
        self.assertNotEqual(original.version, self.definition("judge_1").version)

    def test_reports_keep_subversions_and_legacy_judges_separate(self):
        with patch.object(LiteLLMJudge, "complete", autospec=True,
                          side_effect=lambda adapter, task: self.response(task, 1)):
            first = self.compare("judge_1")
            config = self.configs / "judge_1.toml"
            config.write_text(config.read_text().replace('reasoning_effort = "low"', 'reasoning_effort = "high"'))
            second = self.compare("judge_1")
            legacy = {"request_id": "legacy", **evaluate(self.task, LiteLLMJudge("synthetic"))}
        reports = judge_report(self.dataset, [first, second, legacy])["judges"]
        self.assertEqual(len(reports), 3)
        self.assertEqual(len({r["judge_version"] for r in reports}), 3)

    def test_historical_and_routed_versions_coexist_and_key_rotation_is_stable(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'first-key'}, clear=True), \
                patch.object(LiteLLMJudge, 'complete', autospec=True,
                             side_effect=lambda adapter, task: self.response(task, 1)):
            historical = self.compare('judge_1')
            def routed(client):
                return compare_plans(*self.task.plans, 'judge_1', context=self.context,
                                     judges_dir=self.configs, client=client)
            native = routed(LLMClient('openai'))
            os.environ['OPENAI_API_KEY'] = 'rotated-key'
            rotated = routed(LLMClient('openai'))
            router = routed(LLMClient('openrouter'))
        self.assertEqual(historical['judge_version'], self.definition('judge_1').version)
        self.assertEqual(native['judge_version'], rotated['judge_version'])
        self.assertNotEqual(native['judge_version'], router['judge_version'])
        reports = judge_report(self.dataset, [historical, native, router])['judges']
        self.assertEqual(len({r['judge_version'] for r in reports}), 3)

    def test_strategy_changes_require_new_id_in_existing_results(self):
        original = self.definition("judge_3")
        changed = replace(original, repetitions=7)
        with self.assertRaisesRegex(ValueError, "new judge ID"):
            check_strategy_history(changed.configuration(), [{"judge": original.configuration()}])

    def test_bad_config_and_single_plan_fail_before_any_call(self):
        with patch.object(LiteLLMJudge, "complete") as call:
            with self.assertRaisesRegex(ValueError, "exactly two"):
                ConfiguredJudge(self.definition("judge_1")).complete(make_task(self.dataset, "g1", mode="single"))
            for judge_id in ("missing", "../judge_1"):
                with self.subTest(judge_id=judge_id), self.assertRaises(ValueError):
                    self.definition(judge_id)
            config = self.configs / "judge_1.toml"
            config.write_text(config.read_text().replace('strategy = "single"', 'strategy = "single"\nrepetitions = 5'))
            with self.assertRaisesRegex(ValueError, "only"):
                self.definition("judge_1")
            call.assert_not_called()

    def test_incomplete_or_malformed_call_does_not_produce_a_majority_result(self):
        with patch.object(LiteLLMJudge, "complete", return_value=('{}', {})) as call:
            with self.assertRaisesRegex(ValueError, "rubric IDs"):
                self.compare("judge_4")
            self.assertEqual(call.call_count, 1)

    def test_cli_preview_call_count_and_resume(self):
        args = ["--ssot", str(self.dataset.root), "judge", "--grading-id", "g1",
                "--judge", "judge_2", "--judges-dir", str(self.configs)]
        with chdir(self.root), redirect_stdout(io.StringIO()) as stdout, patch.object(LiteLLMJudge, "complete") as calls:
            run(parser().parse_args(args + ["--dry-run"]))
            preview = json.loads(stdout.getvalue())
            calls.assert_not_called()
            self.assertEqual(preview["calls"], 2)
            self.assertEqual([c["order"] for c in preview["tasks"][0]["calls"]], ["AB", "BA"])
            self.assertFalse(Path("results/judgments.jsonl").exists())
        with chdir(self.root), redirect_stdout(io.StringIO()), patch.object(
            LiteLLMJudge, "complete", autospec=True,
            side_effect=lambda adapter, task: self.response(task, 0),
        ) as calls:
            run(parser().parse_args(args))
            run(parser().parse_args(args))
            self.assertEqual(calls.call_count, 2)
            saved = records_from(Path("results/judgments.jsonl"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(len(saved[0]["calls"]), 2)
            self.assertEqual(saved[0]["request_id"], preview["tasks"][0]["request_id"])
            prompt = self.configs / "prompts/pairwise_system.txt"
            prompt.write_text(prompt.read_text() + "\nAdditional instruction.\n")
            run(parser().parse_args(args))
            self.assertEqual(calls.call_count, 4)
            saved = records_from(Path("results/judgments.jsonl"))
            self.assertEqual(len(saved), 2)
            self.assertNotEqual(saved[0]["judge_version"], saved[1]["judge_version"])

    def test_cli_rejects_strategy_change_before_spending_on_calls(self):
        with patch.object(LiteLLMJudge, "complete", autospec=True,
                          side_effect=lambda adapter, task: self.response(task, 0)):
            saved = self.compare("judge_1")
        config = self.configs / "judge_1.toml"
        config.write_text(config.read_text().replace('strategy = "single"', 'strategy = "order_check"'))
        output = self.root / "results.jsonl"
        output.write_text(json.dumps(saved) + "\n")
        args = ["--ssot", str(self.dataset.root), "judge", "--grading-id", "g1",
                "--judge", "judge_1", "--judges-dir", str(self.configs), "--output", str(output)]
        with patch.object(LiteLLMJudge, "complete") as calls, self.assertRaisesRegex(ValueError, "new judge ID"):
            run(parser().parse_args(args))
        calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
