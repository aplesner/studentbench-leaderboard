"""Offline behavioral tests with entirely synthetic lesson plans and human reviews."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from studentbench_leaderboard.cli import output_path, parser, records_from, run
from studentbench_leaderboard.data import ACCURACY_SCALE, DEFAULT_SCALE, Dataset, STUDY
from studentbench_leaderboard.judge import PRETEST_FIELDS, evaluate, make_task, parse_result
from studentbench_leaderboard.report import human_scores, judge_report, leaderboard, spearman


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    directory = root / STUDY / "gre_quant"
    directory.mkdir(parents=True)
    common = {"instrument_id": "gre_quant", "rubric_version": "synthetic.v1", "comparison_type": "ai_vs_ai"}
    write_csv(directory / "human_grading_sessions.csv", [
        {**common, "grading_id": "g1", "comparison_id": "c1", "review_subject_id": "s1",
         "lesson_plan_a_id": "z", "lesson_plan_b_id": "a", "review_display_variant": "complete_plan",
         "human_reviewer_anon_id": "HUMAN_ID_SECRET", "general_notes": "HUMAN_NOTES_SECRET"},
        {**common, "grading_id": "g2", "comparison_id": "c1", "review_subject_id": "s1",
         "lesson_plan_a_id": "a", "lesson_plan_b_id": "z", "review_display_variant": "complete_plan",
         "human_reviewer_anon_id": "HUMAN_ID_2_SECRET", "general_notes": ""},
    ])
    questions = [{**common, "question_variant_id": key, "question_position": str(i),
                  "question_text": f"Assess {key}", "rubric_question_id": key}
                 for i, key in enumerate(("format_old", "strategy_new", "accuracy"), 1)]
    write_csv(directory / "rubric_questions.csv", questions)
    scale_rows = []
    for scale, labels in ((DEFAULT_SCALE, ["Very Poor", "Poor", "Average", "Good", "Excellent"]),
                          (ACCURACY_SCALE, [f"solution label {n}" for n in range(1, 6)])):
        scale_rows.extend({"scale_id": scale, "score": str(i), "label": label}
                          for i, label in enumerate(labels, 1))
    write_csv(directory / "rating_scales.csv", scale_rows)
    grades = []
    for grading, key, position, ra, rb, la, lb, preference in [
        ("g1", "format_old", 1, 5, 5, "Excellent", "Excellent", 2),
        ("g1", "accuracy", 3, 3, 2, "solution label 3", "solution label 2", 0),
        ("g2", "strategy_new", 2, 1, 4, "Very Poor", "Good", -1),
        ("g2", "accuracy", 3, 1, 5, "Very Poor", "Excellent", -2),
    ]:
        grades.append({**common, "grading_id": grading, "comparison_id": "c1",
                       "question_variant_id": key, "question_position": str(position),
                       "lesson_plan_a_rating": str(ra), "lesson_plan_b_rating": str(rb),
                       "lesson_plan_a_rating_label": la, "lesson_plan_b_rating_label": lb,
                       "preference_score": str(preference), "reviewer_reasoning": "HUMAN_REASON_SECRET"})
    write_csv(directory / "rubric_grades.csv", grades)
    write_csv(directory / "review_subject_pretest_responses.csv", [
        {**{key: "" for key in PRETEST_FIELDS}, "review_subject_id": "s1", "question_id": "Q1",
         "question_text": "2 + 3 = ?", "correct_answer": "5", "student_answer": "4",
         "is_correct": "0", "linked_ssot_student_id": "LINKED_ID_SECRET"},
    ])
    plans = [{"lesson_plan_id": key, "review_subject_ids": ["s1"],
              "ai_model_preset_id": f"AUTHOR_{key}_SECRET", "plan_as_stored": {
                  "summary": f"Synthetic plan {key}", "concepts": [{"name": "Addition",
                  "practice_problems": [{"stem": "1 + 2 = ?", "solution": "3"}]}]}}
             for key in ("a", "z")]
    (directory / "reviewed_lesson_plans.json").write_text(json.dumps({"plans": plans}))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "ssot"
        fixture(self.root)
        self.dataset = Dataset(self.root, "gre_quant")

    def result(self, task, values):
        return json.dumps({q["id"]: {"score": value, "reason": "Synthetic evidence"}
                           for q, value in zip(task.rubric, values)})

    def record(self, task, values, request="request-1"):
        raw = self.result(task, values)

        class FakeJudge:
            def configuration(self):
                return {"adapter": "test", "model": "synthetic"}

            def complete(self, task):
                return raw, {}

        return {"request_id": request, **evaluate(task, FakeJudge())}

    def test_prompts_hide_human_answers_and_author_metadata_but_keep_problem_text(self):
        task = make_task(self.dataset, "g1")
        text = json.dumps(task.messages())
        self.assertNotIn("SECRET", text)
        self.assertIn("1 + 2 = ?", text)
        self.assertIn("student_answer", text)
        self.assertNotIn("reviewer_reasoning", text)
        self.assertEqual(task.rubric[1]["scale_id"], ACCURACY_SCALE)

    def test_reversed_slots_align_to_direct_human_preference(self):
        task = make_task(self.dataset, "g1", swap=True)
        report = judge_report(self.dataset, [self.record(task, [-2, 0])])
        rows = report["judges"][0]["agreement"]
        self.assertTrue(all(r["mae_against_human_mean"] == 0 for r in rows))
        self.assertTrue(all(r["direction_agreement_against_human_mean"] == 1 for r in rows))
        rankings = report["judges"][0]["leaderboard"]
        winner = next(r for r in rankings if r["question_variant_id"] == "format_old")
        self.assertEqual(winner["model"], "AUTHOR_z_SECRET")
        self.assertEqual(winner["score"], 1)
        # Both human absolute scores were 5, but the direct preference was +2.
        self.assertEqual(rows[1]["n_matched_units"], 1)

    def test_scale_and_wording_variants_do_not_merge(self):
        values = human_scores(self.dataset)
        accuracy_keys = [k for k in values if k[0] == "pairwise" and k[3] == "accuracy"]
        self.assertEqual({k[4] for k in accuracy_keys}, {DEFAULT_SCALE, ACCURACY_SCALE})
        self.assertEqual({k[3] for k in values}, {"format_old", "strategy_new", "accuracy"})

    def test_abstention_is_excluded_and_tie_counts_as_half_win(self):
        task = make_task(self.dataset, "g1")
        report = judge_report(self.dataset, [self.record(task, [None, 0])])
        group = report["judges"][0]
        abstained = next(r for r in group["agreement"] if r["question_variant_id"] == "format_old")
        self.assertEqual(abstained["n_abstentions"], 1)
        self.assertEqual(abstained["n_matched_units"], 0)
        self.assertIsNone(abstained["mae_against_human_mean"])
        self.assertTrue(all(r["score"] == .5 for r in group["leaderboard"]))

    def test_single_plan_and_repeated_human_ratings(self):
        task = make_task(self.dataset, "g1", mode="single", slot="b")
        self.assertEqual(task.plan_ids, ["a"])
        self.assertEqual(len(task.plans), 1)
        report = judge_report(self.dataset, [self.record(task, [5, 2])])
        self.assertTrue(all(r["mae_against_human_mean"] == 0 for r in report["judges"][0]["agreement"]))
        values = human_scores(self.dataset)
        key = next(k for k in values if k[0] == "single" and k[3] == "format_old")
        values[key] = [1, 5]
        row = next(r for r in leaderboard(self.dataset, {key: values[key]}, "plan"))
        self.assertEqual(row["score"], 3)
        self.assertEqual(row["n_units"], 1)

    def test_strict_result_validation(self):
        task = make_task(self.dataset, "g1", mode="single")
        for value in (True, 2.0, "3", 0, 6, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_result(self.result(task, [value, 3]), task)
        for raw in ('{}', '[]', '{"format_old": {}, "format_old": {}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_result(raw, task)

    def test_invalid_context_and_output_are_rejected(self):
        self.dataset.plans["a"]["review_subject_ids"] = ["different-student"]
        with self.assertRaisesRegex(ValueError, "pretest"):
            make_task(self.dataset, "g1")
        with self.assertRaisesRegex(ValueError, "distinct"):
            make_task(self.dataset, "g1", ["z", "z"])
        with self.assertRaisesRegex(ValueError, "SSOT"):
            output_path(self.root / "accidental-output.json", self.dataset)

    def test_mismatched_snapshot_and_duplicate_results_are_rejected(self):
        task = make_task(self.dataset, "g1")
        record = self.record(task, [1, 0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            judge_report(self.dataset, [record, record])
        record["dataset_sha256"] = "old-snapshot"
        with self.assertRaisesRegex(ValueError, "different SSOT"):
            judge_report(self.dataset, [record])

    def test_cli_preview_resume_and_report_without_network(self):
        preview = Path(self.temp.name) / "preview.json"
        output = Path(self.temp.name) / "judgments.jsonl"
        args = ["--ssot", str(self.root), "judge", "--grading-id", "g1"]
        with patch("builtins.print"), patch("studentbench_leaderboard.judge.LiteLLMJudge.complete") as call:
            run(parser().parse_args(args + ["--dry-run", "--output", str(preview)]))
            call.assert_not_called()
        self.assertEqual(json.loads(preview.read_text())["calls"], 1)
        task = make_task(self.dataset, "g1")
        with patch("builtins.print"), patch("studentbench_leaderboard.judge.LiteLLMJudge.complete",
                                          return_value=(self.result(task, [2, 0]), {})) as call:
            real_args = parser().parse_args(args + ["--model", "synthetic/test", "--output", str(output)])
            run(real_args)
            run(real_args)
            self.assertEqual(call.call_count, 1)
        records = records_from(output)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(judge_report(self.dataset, records)["judges"]), 1)

    def test_spearman_uses_average_ranks_and_reports_undefined_values(self):
        self.assertEqual(spearman([1, 1, 3], [5, 5, 9]), 1)
        self.assertIsNone(spearman([1, 1], [1, 2]))
        self.assertIsNone(spearman([], []))

    def test_default_results_resume_and_dry_run_preserves_saved_judgments(self):
        task = make_task(self.dataset, "g1")
        args = ["--ssot", str(self.root), "judge", "--grading-id", "g1", "--model", "synthetic/test"]
        with chdir(self.temp.name), patch("builtins.print"), patch(
            "studentbench_leaderboard.judge.LiteLLMJudge.complete",
            return_value=(self.result(task, [2, 0]), {}),
        ) as call:
            output = Path("results/judgments.jsonl")
            run(parser().parse_args(args + ["--dry-run"]))
            call.assert_not_called()
            self.assertFalse(output.exists())
            run(parser().parse_args(args))
            saved = output.read_bytes()
            run(parser().parse_args(args))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(len(records_from(output)), 1)
            run(parser().parse_args(args + ["--dry-run"]))
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
