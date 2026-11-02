"""Soft ranking outputs must coexist with honest categorical diagnostics."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studentbench_leaderboard import ComparisonContext, compare_plans
from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.judge import LiteLLMJudge, make_task
from studentbench_leaderboard.judge_config import load_definition
from studentbench_leaderboard.ranking import aggregate_pairs, fit_bradley_terry, judgment_rows
from test_pipeline import fixture


class SoftJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root / "ssot")
        self.dataset = Dataset(self.root / "ssot", "gre_quant")
        self.task = make_task(self.dataset, "g1")
        self.context = ComparisonContext(self.task.instrument, self.task.subject, self.task.pretest,
                                         self.task.rubric, tuple(self.task.plan_ids), self.task.dataset_sha256)

    def response(self, task, winner):
        return json.dumps({criterion["id"]: {
            "evidence_a": "Plan A evidence", "evidence_b": "Plan B evidence", "reason": "Specific tradeoff",
            "winner": winner, "strength": "slight" if winner in ("A", "B") else "none",
        } for criterion in task.rubric}), {}

    def compare(self, judge, response):
        with patch.object(LiteLLMJudge, "complete", autospec=True, side_effect=response):
            return compare_plans(*self.task.plans, judge, context=self.context)

    def test_opposite_order_votes_are_fractional_evidence_without_categorical_tie(self):
        result = self.compare("judge_7", lambda adapter, task: self.response(task, "A"))
        self.assertEqual([call["order"] for call in result["calls"]], ["AB", "BA"])
        self.assertEqual(result["diagnostics"]["format_old"]["normalized_scores"], [1, -1])
        self.assertIsNone(result["ratings"]["format_old"]["score"])
        self.assertEqual(result["preference_probabilities"]["format_old"], .5)
        self.assertEqual(result["diagnostics"]["format_old"]["normalized_win_shares"], [1, 0])
        rows = aggregate_pairs(judgment_rows(self.dataset, [result]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], .5)
        fit = fit_bradley_terry(rows)
        self.assertTrue(fit["connected"])
        self.assertEqual(fit["n_comparisons"], 1)
        self.assertEqual([row["rank"] for row in fit["leaderboard"]], [1.5, 1.5])

    def test_swapped_calls_and_adapter_restore_actual_plan_identity(self):
        result = self.compare("judge_7", lambda adapter, task: self.response(
            task, "A" if task.plan_ids == self.task.plan_ids else "B"))
        self.assertEqual(result["preference_probabilities"]["format_old"], 1)
        self.assertEqual(result["ratings"]["format_old"]["score"], 1)
        rows = judgment_rows(self.dataset, [result])
        self.assertEqual(rows[0]["plan_a"], "a")
        self.assertEqual(rows[0]["plan_b"], "z")
        self.assertEqual(rows[0]["value"], 0)
        fit = fit_bradley_terry(aggregate_pairs(rows))
        self.assertEqual(fit["leaderboard"][0]["model"], "AUTHOR_z_SECRET")

    def test_abstention_nulls_soft_share_even_when_categorical_majority_exists(self):
        result = self.compare("judge_8", lambda adapter, task: self.response(
            task, "insufficient" if adapter.model == "openai/gpt-6-astra" else "A"))
        self.assertEqual(len(result["calls"]), 4)
        self.assertEqual(result["ratings"]["format_old"]["score"], 1)
        self.assertIsNone(result["preference_probabilities"]["format_old"])
        self.assertEqual(result["diagnostics"]["format_old"]["preference_status"], "member_abstention")
        self.assertEqual(aggregate_pairs(judgment_rows(self.dataset, [result])), [])

    def test_panel_soft_preserves_all_four_equal_votes(self):
        winners = {"openai/gpt-5.6-luna": "A", "openai/gpt-5.6-terra": "A",
                   "openai/gpt-5.6-sol": "tie", "openai/gpt-6-astra": "B"}
        result = self.compare("judge_8", lambda adapter, task: self.response(task, winners[adapter.model]))
        self.assertEqual(result["preference_probabilities"]["format_old"], .625)
        self.assertIsNone(result["ratings"]["format_old"]["score"])

    def test_legacy_scores_remain_supported_and_soft_fields_are_validated(self):
        result = self.compare("judge_7", lambda adapter, task: self.response(task, "tie"))
        legacy = deepcopy(result)
        del legacy["preference_probabilities"]
        self.assertEqual(judgment_rows(self.dataset, [legacy])[0]["value_source"], "categorical_score")
        for bad in (float("nan"), 1.1, True):
            changed = deepcopy(result)
            changed["preference_probabilities"]["format_old"] = bad
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "finite numbers"):
                judgment_rows(self.dataset, [changed])
        changed = deepcopy(result)
        changed["preference_probabilities"] = {}
        with self.assertRaisesRegex(ValueError, "exactly"):
            judgment_rows(self.dataset, [changed])

    def test_adapter_rejects_fingerprint_unknown_plan_and_cross_subject_identity(self):
        result = self.compare("judge_7", lambda adapter, task: self.response(task, "tie"))
        for field, value in (("dataset_sha256", "wrong"), ("plan_ids", ["z", "unknown"]),
                             ("subject", "wrong"), ("model_a", "wrong")):
            changed = {**result, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                judgment_rows(self.dataset, [changed])

    def test_adapter_rejects_duplicate_requests_and_mixed_judges(self):
        result = self.compare("judge_7", lambda adapter, task: self.response(task, "tie"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            judgment_rows(self.dataset, [result, result])
        other = {**result, "request_id": "another", "judge_version": "judge_8_other"}
        with self.assertRaisesRegex(ValueError, "one judge version"):
            judgment_rows(self.dataset, [result, other])

    def test_existing_judge_versions_and_configuration_hashes_do_not_change(self):
        expected = {
            "judge_1": ("97a23a27f145_23a0cd4fe2a2", "f9556e9af6b3e78f2ec4376a8c0ce1baac67809d99015c55b4000b64a2c18829"),
            "judge_2": ("97a23a27f145_23a0cd4fe2a2", "9a282cd2c9e487234dc05c50d7040c1666aca238462dcea5dc275e907fdf0218"),
            "judge_3": ("97a23a27f145_23a0cd4fe2a2", "f7c768f093975b377748ac5a246e70d29918dc0e13cb9688fe5e0982e60d1d64"),
            "judge_4": ("97a23a27f145_3a49e010a3f1", "41f03a6844a94666f5403d45b964976ef7cb51584bcbdb2f3c7ef26b2daeaa65"),
            "judge_5": ("0df3ac622a18_048544b8a34e", "37016d4530b7b4f2b97e366d8a807b333cb4de6e180bf9a4925340f9c90bb02c"),
            "judge_6": ("bb671964b88b_048544b8a34e", "4a4093db482dad3aa3e7839a9e8a5ccb89e9352105f8a821adf8ce9a80107048"),
        }
        for name, (suffix, configuration_hash) in expected.items():
            with self.subTest(judge=name):
                definition = load_definition(name)
                self.assertEqual(definition.version, name + "_" + suffix)
                self.assertEqual(digest(definition.configuration()), configuration_hash)


if __name__ == "__main__":
    unittest.main()
