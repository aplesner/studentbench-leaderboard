"""Fixed-example selection must remain development-only and preserve source text."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/ranking-extension/prepare_example.py"
spec = importlib.util.spec_from_file_location("prepare_calibration_example", MODULE_PATH)
example_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_module)


class DevelopmentGradesOnly(dict):
    def __getitem__(self, key):
        if key.startswith("heldout"):
            raise AssertionError("Held-out human grades were inspected")
        return super().__getitem__(key)


def selection_fixture(mixed_order=False):
    sessions = {}
    grades = DevelopmentGradesOnly()
    for index in range(2):
        key = f"dev{index}"
        plans = ("b", "a") if index and mixed_order else ("a", "b")
        sessions[key] = {"review_subject_id": "development_student", "lesson_plan_a_id": plans[0],
                         "lesson_plan_b_id": plans[1], "human_reviewer_anon_id": f"reviewer{index}"}
        grades[key] = [{"question_variant_id": f"q{question}",
                        "preference_score": str(1 if question % 2 else -1),
                        "reviewer_reasoning": "Source observation"} for question in range(1, 9)]
    # A larger held-out panel must not become an eligible demonstration.
    for index in range(5):
        sessions[f"heldout{index}"] = {"review_subject_id": "test_student", "lesson_plan_a_id": "x",
                                       "lesson_plan_b_id": "z", "human_reviewer_anon_id": f"other{index}"}
    dataset = SimpleNamespace(sessions=sessions, grades_by_session=grades, rating_scale=lambda _: "scale")
    return dataset, {"splits": {"development": ["development_student"], "test": ["test_student"]}}


class CalibrationExampleTests(unittest.TestCase):
    def test_selection_never_reads_heldout_grades_and_keeps_one_complete_pair(self):
        dataset, manifest = selection_fixture()
        chosen, summary = example_module.select_example(dataset, manifest)
        self.assertEqual(chosen["subject"], "development_student")
        self.assertEqual(chosen["grading_ids"], ["dev0", "dev1"])
        self.assertEqual(chosen["unanimous_direction_criteria"], 8)
        self.assertEqual(summary["development_pairs_considered"], 1)

    def test_different_human_presentation_is_excluded_instead_of_rewriting_reasons(self):
        dataset, manifest = selection_fixture(mixed_order=True)
        with self.assertRaisesRegex(ValueError, "No development example"):
            example_module.select_example(dataset, manifest)

    def test_one_sided_example_is_ineligible(self):
        dataset, manifest = selection_fixture()
        for rows in dataset.grades_by_session.values():
            for row in rows:
                row["preference_score"] = "1"
        with self.assertRaisesRegex(ValueError, "No development example"):
            example_module.select_example(dataset, manifest)

    def test_redaction_preserves_math_dollars_and_human_wording_except_identifiers(self):
        from collections import Counter
        value = {"reason": "human_tutor_001 says Plan A's $2 problem is clearer.",
                 "plan": {"text": "model-x-high wrote this $x$ equation"}}
        counts = Counter()
        result = example_module._redact_identifiers(value, {"human_tutor_001": "Reviewer 1",
                                                           "model-x-high": "[author identity omitted]"}, counts)
        self.assertEqual(result["reason"], "Reviewer 1 says Plan A's $2 problem is clearer.")
        self.assertIn("$x$", result["plan"]["text"])
        self.assertEqual(sum(counts.values()), 2)

    def test_fixed_artifact_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.json"
            first = example_module._write_once(path, {"example": 1})
            self.assertEqual(example_module._write_once(path, {"example": 1}), first)
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                example_module._write_once(path, {"example": 2})

    def test_local_example_keeps_actual_source_preferences_ratings_and_reasons(self):
        root = MODULE_PATH.parents[2]
        artifact = root / "results/ranking-extension-20260911/example.json"
        provenance_path = artifact.with_name("example_provenance.json")
        ssot = root / "studentbench_data_ssot"
        if not artifact.exists() or not provenance_path.exists() or not ssot.exists():
            self.skipTest("Local ignored source and fixed example are unavailable")
        from studentbench_leaderboard.data import Dataset, score
        dataset = Dataset(ssot, "gre_quant")
        example = json.loads(artifact.read_text())
        provenance = json.loads(provenance_path.read_text())
        self.assertEqual(provenance["source_split"], "development")
        self.assertEqual(provenance["n_full_comparisons"], 1)
        self.assertEqual(provenance["identifier_redactions"], {})
        self.assertEqual(example_module.sha256(artifact), provenance["example_sha256"])
        for review, grading_id in zip(example["human_reviews"], provenance["source_grading_ids"]):
            session = dataset.sessions[grading_id]
            self.assertEqual([session["lesson_plan_a_id"], session["lesson_plan_b_id"]],
                             provenance["source_plan_ids_in_example_order"])
            source = {g["question_variant_id"]: g for g in dataset.grades_by_session[grading_id]}
            for grade in review["criteria"]:
                original = source[grade["criterion_id"]]
                self.assertEqual(grade["reason"], original["reviewer_reasoning"])
                self.assertEqual(grade["preference"]["score"], score(original["preference_score"], -2, 2))
                self.assertEqual(grade["preference"]["label"], original["preference_label"])
                for slot in ("a", "b"):
                    self.assertEqual(grade[f"plan_{slot}_rating"]["score"], score(original[f"lesson_plan_{slot}_rating"], 1, 5))
                    self.assertEqual(grade[f"plan_{slot}_rating"]["label"], original[f"lesson_plan_{slot}_rating_label"])


if __name__ == "__main__":
    unittest.main()
