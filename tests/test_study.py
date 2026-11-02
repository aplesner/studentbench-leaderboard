"""Offline study-runner integrity checks with synthetic source data and no APIs."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from studentbench_leaderboard.data import Dataset, STUDY
from studentbench_leaderboard.study import PRICES, call_path, estimated_cost, run_jobs
from test_pipeline import fixture


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.study.mkdir()
        fixture(self.root / "ssot")
        # A historical upstream answer is source metadata, not a human verdict.
        plan_path = self.root / "ssot" / STUDY / "gre_quant" / "reviewed_lesson_plans.json"
        source = json.loads(plan_path.read_text())
        for plan in source["plans"]:
            problem = plan["plan_as_stored"]["concepts"][0]["practice_problems"][0]
            problem.update(answer="3", answer_original="upstream_original_answer")
        plan_path.write_text(json.dumps(source))
        self.dataset = Dataset(self.root / "ssot", "gre_quant")
        self.entry = {"pair_id": "synthetic_pair", "subject": "s1", "grading_id": "g1",
                      "plan_ids": ["a", "z"], "model_a": "AUTHOR_a_SECRET",
                      "model_b": "AUTHOR_z_SECRET", "human_reviewed": True}
        self.manifest = {"instrument": "gre_quant", "seed": 7,
                         "dataset_sha256": self.dataset.fingerprint,
                         "pair_catalog": {"validation": [self.entry]}}
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        source_configs = Path(__file__).resolve().parents[1] / "experiments" / "ranking-study"
        shutil.copytree(source_configs / "configs" / "luna_low", self.study / "configs" / "luna_low")
        shutil.copytree(source_configs / "prompts", self.study / "prompts")
        variants = [v for v in json.loads((source_configs / "variants.json").read_text())
                    if v["variant"] == "luna_low"]
        (self.study / "variants.json").write_text(json.dumps(variants))

    @staticmethod
    def response(task):
        result = {q["id"]: {"evidence_a": "A has concrete examples", "evidence_b": "B has examples",
                             "reason": "Presented A has a slight advantage", "winner": "A",
                             "strength": "slight"} for q in task.rubric}
        return json.dumps(result), {"usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                                               "prompt_tokens_details": {"cached_tokens": 250}},
                                    "response_model": "synthetic", "response_id": "mock-response"}

    def run_mocked(self, jobs):
        with patch("studentbench_leaderboard.study.Dataset", return_value=self.dataset), \
                patch("studentbench_leaderboard.study.LiteLLMJudge.complete", side_effect=self.response) as api, \
                redirect_stdout(io.StringIO()):
            counts = run_jobs(self.study, jobs, workers=1)
        return counts, api

    def test_opposite_order_draws_have_distinct_identity_and_correct_normalized_sign(self):
        jobs = [{"variant": "luna_low", "pair_id": "synthetic_pair", "draw": draw} for draw in (0, 1)]
        counts, api = self.run_mocked(jobs)
        self.assertEqual(counts["completed"], 2)
        self.assertEqual(api.call_count, 2)
        records = [json.loads(call_path(self.study, job).read_text()) for job in jobs]
        self.assertEqual({r["order"] for r in records}, {"AB", "BA"})
        self.assertNotEqual(records[0]["request_id"], records[1]["request_id"])
        for record in records:
            self.assertEqual(record["plan_ids"], ["a", "z"])
            expected = 1 if record["order"] == "AB" else -1
            for rating in record["normalized_ratings"].values():
                self.assertEqual(rating["score"], expected)
            self.assertGreater(record["estimated_usd"], 0)

    def test_prompt_hides_human_answers_and_authors_and_preserves_displayed_source_answer(self):
        job = {"variant": "luna_low", "pair_id": "synthetic_pair", "draw": 0}
        _, api = self.run_mocked([job])
        task = api.call_args.args[0]
        messages = task.messages()
        text = json.dumps(messages)
        for secret in ("HUMAN_ID_SECRET", "HUMAN_NOTES_SECRET", "HUMAN_REASON_SECRET", "LINKED_ID_SECRET",
                       "AUTHOR_a_SECRET", "AUTHOR_z_SECRET", "lesson_plan_a_rating", "preference_score"):
            self.assertNotIn(secret, text)
        self.assertNotIn("rating_labels", messages[1]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["pretest"][0]["correct_answer"], "5")
        for plan in payload["plans"].values():
            problem = plan["concepts"][0]["practice_problems"][0]
            self.assertEqual(problem["answer"], "3")
        # Source preservation: the runner never rewrites SSOT or replaces the
        # reviewed displayed answer with an upstream historical answer.
        for plan in self.dataset.plans.values():
            problem = plan["plan_as_stored"]["concepts"][0]["practice_problems"][0]
            self.assertEqual(problem["answer_original"], "upstream_original_answer")
            self.assertEqual(problem["answer"], "3")

    def test_completed_job_reuses_result_without_another_api_call(self):
        job = {"variant": "luna_low", "pair_id": "synthetic_pair", "draw": 0}
        self.run_mocked([job])
        counts, api = self.run_mocked([job])
        self.assertEqual(counts["reused"], 1)
        self.assertEqual(counts["completed"], 0)
        api.assert_not_called()

    def test_changed_cached_request_is_rejected(self):
        job = {"variant": "luna_low", "pair_id": "synthetic_pair", "draw": 0}
        self.run_mocked([job])
        path = call_path(self.study, job)
        record = json.loads(path.read_text())
        record["request_id"] = "different-request"
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.run_mocked([job])

    def test_dataset_fingerprint_mismatch_is_rejected_before_api(self):
        self.manifest["dataset_sha256"] = "wrong-dataset"
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_mocked([{"variant": "luna_low", "pair_id": "synthetic_pair"}])

    def test_variant_metadata_cannot_override_the_hashed_judge_configuration(self):
        variants_path = self.study / "variants.json"
        original = json.loads(variants_path.read_text())
        for key, changed in (("model", "openai/gpt-5.6-sol"), ("reasoning_effort", "high"),
                             ("response_style", "absolute_ratings")):
            with self.subTest(key=key):
                variants = [{**original[0], key: changed}]
                variants_path.write_text(json.dumps(variants))
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    self.run_mocked([{"variant": "luna_low", "pair_id": "synthetic_pair"}])
        variants_path.write_text(json.dumps(original))

    def test_frozen_prompt_mutation_is_rejected_before_new_draw(self):
        job = {"variant": "luna_low", "pair_id": "synthetic_pair", "draw": 0}
        self.run_mocked([job])
        prompt = self.study / "configs" / "luna_low" / "prompts" / "system.txt"
        prompt.write_text(prompt.read_text() + "\nChanged instructions.\n")
        with self.assertRaisesRegex(ValueError, "frozen judge configuration changed"):
            self.run_mocked([{**job, "draw": 1}])

    def test_cached_normalization_and_judge_metadata_are_revalidated(self):
        job = {"variant": "luna_low", "pair_id": "synthetic_pair", "draw": 0}
        self.run_mocked([job])
        path = call_path(self.study, job)
        original = json.loads(path.read_text())
        record = json.loads(json.dumps(original))
        criterion = next(iter(record["normalized_ratings"]))
        record["normalized_ratings"][criterion]["score"] *= -1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "current response parser"):
            self.run_mocked([job])
        record = json.loads(json.dumps(original))
        record["judge"]["version"] = "wrong-version"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "judge version mismatch"):
            self.run_mocked([job])

    def test_estimated_cost_accounts_for_cached_input_and_completion_once(self):
        for model, (input_price, cache_price, output_price) in PRICES.items():
            usage = {"prompt_tokens": 1_000_000, "completion_tokens": 10_000,
                     "prompt_tokens_details": {"cached_tokens": 250_000},
                     "completion_tokens_details": {"reasoning_tokens": 9_000}}
            expected = .75 * input_price + .25 * cache_price + .01 * output_price
            self.assertAlmostEqual(estimated_cost("openai/" + model, usage), expected)
            self.assertEqual(estimated_cost(model, {}), 0)


if __name__ == "__main__":
    unittest.main()
