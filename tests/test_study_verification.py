"""The final verifier must fail closed without invoking a provider or runner."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studentbench_leaderboard.comparison import ConfiguredJudge
from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.judge import LiteLLMJudge, make_task
from studentbench_leaderboard.judge_config import load_definition
from test_pipeline import fixture


spec = importlib.util.spec_from_file_location("study_verification", Path(__file__).resolve().parents[1]
                                            / "experiments/ranking-study/verify.py")
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


class FinalVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_completion_and_selection_checks_fail_before_data_loading(self):
        selection = {"variant": "judge", "rule": {"minimum_coverage": .9}}
        self.write("analysis.json", {"status": "running"})
        self.write("manifest.json", {})
        with patch.object(verification, "Dataset") as dataset, self.assertRaisesRegex(ValueError, "not complete"):
            verification.verify(self.root, self.root / "ssot")
        dataset.assert_not_called()
        self.write("analysis.json", {"status": "complete", "selected_judge": selection})
        self.write("selection.json", selection)
        self.write("selection_rule.json", selection["rule"])
        self.write("manifest.json", {"selection_locked": True, "selection": selection, "selection_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "lock hash"):
            verification.check_completion(self.root)
        self.write("manifest.json", {"selection_locked": True, "selection": selection, "selection_sha256": digest(selection)})
        self.assertEqual(verification.check_completion(self.root)[2], selection)

    def test_job_ledger_deduplicates_stage_reuse_but_rejects_duplicate_stage_jobs(self):
        a = {"variant": "v", "pair_id": "a", "draw": 0}
        b = {"variant": "v", "pair_id": "b", "draw": 0}
        c = {"variant": "v", "pair_id": "c", "draw": 0}
        for name, rows in zip(verification.JOB_FILES, ([a], [a, b], [c])):
            self.write(name, rows)
        jobs, counts = verification.expected_jobs(self.root)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(sum(counts.values()), 4)
        self.write("round1_jobs.json", [a, a])
        with self.assertRaisesRegex(ValueError, "Duplicate jobs"):
            verification.expected_jobs(self.root)

    def cached_fixture(self):
        fixture(self.root / "ssot")
        dataset = Dataset(self.root / "ssot", "gre_quant")
        task = make_task(dataset, "g1", ["a", "z"])
        judge = ConfiguredJudge(load_definition("judge_1"))
        pair_id = digest([task.subject, *task.plan_ids])[:20]
        entry = {"pair_id": pair_id, "subject": task.subject, "plan_ids": task.plan_ids,
                 "model_a": "AUTHOR_a_SECRET", "model_b": "AUTHOR_z_SECRET", "grading_id": "g1",
                 "human_reviewed": True, "split": "validation"}
        job = {"variant": "luna_low", "pair_id": pair_id, "draw": 0}
        manifest = {"seed": 1, "dataset_sha256": dataset.fingerprint}
        swapped = bool(int(digest([manifest["seed"], pair_id]), 16) % 2)
        displayed = judge.call_task(task, swapped)
        parsed = {criterion["id"]: {"score": 1, "reason": "Synthetic evidence"} for criterion in task.rubric}
        normalized = {key: {**value, "score": -1 if swapped else 1} for key, value in parsed.items()}
        messages, schema = displayed.messages(), displayed.schema()
        record = {**entry, **job, "dataset_sha256": dataset.fingerprint,
                  "criteria": task.metadata()["criteria"], "judge": judge.configuration(),
                  "order": "BA" if swapped else "AB", "presented_plan_ids": displayed.plan_ids,
                  "messages": messages, "response_schema": schema,
                  "request_id": digest({"messages": messages, "schema": schema, "judge": judge.configuration(), "draw": 0}),
                  "raw_response": json.dumps(parsed), "ratings": parsed, "normalized_ratings": normalized}
        return record, job, entry, task, judge, manifest

    def test_cache_reconstruction_is_read_only_and_reparses_scores(self):
        args = self.cached_fixture()
        before = deepcopy(args[0])
        with patch.object(LiteLLMJudge, "complete", side_effect=AssertionError("No API")) as provider:
            verification.check_cached_call(*args)
        provider.assert_not_called()
        self.assertEqual(args[0], before)
        corrupt = deepcopy(args[0])
        corrupt["normalized_ratings"]["format_old"]["score"] *= -1
        with self.assertRaisesRegex(ValueError, "normalized preferences"):
            verification.check_cached_call(corrupt, *args[1:])

    def test_cache_reconstruction_rejects_altered_prompt_and_catalog_identity(self):
        args = self.cached_fixture()
        for key, value, expected in (("messages", [], "prompt"), ("subject", "wrong", "catalog"),
                                     ("request_id", "wrong", "request hash")):
            corrupt = {**args[0], key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, expected):
                verification.check_cached_call(corrupt, *args[1:])

    def test_error_attempts_distinguish_resolved_and_pending_jobs(self):
        (self.root / "errors").mkdir()
        (self.root / "calls").mkdir()
        job = {"variant": "v", "pair_id": "p", "draw": 0}
        other = {"variant": "v", "pair_id": "pending", "draw": 0}
        self.write("errors/resolved.json", {**job, "attempt": 1, "type": "Timeout", "status_code": 408})
        self.write("errors/pending.json", {**other, "attempt": 1, "type": "ServerError", "status_code": 500})
        self.write("calls/v--p--0.json", {})
        result = verification.summarize_errors(self.root, {verification.job_key(job): job})
        self.assertEqual(result["saved_error_attempts"], 2)
        self.assertEqual(result["resolved_attempts"], 1)
        self.assertEqual(result["unresolved_attempts"], 1)
        self.assertEqual(result["type_status_counts"]["Timeout / 408"], 1)


if __name__ == "__main__":
    unittest.main()
