"""Synthetic exact-join and metadata isolation tests for optional input context."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from studentbench_leaderboard.pretest_context import PretestContextBank


class PretestContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "application_bank.json"
        self.items = [
            {"id": "M1", "stem": "x is an integer", "quantity_a": "x + 1", "quantity_b": "x",
             "correct_answer": "A", "format": "QC", "skill": "Arithmetic", "subskill": "integer", "difficulty": "2",
             "options": {"A": "A larger", "B": "B larger"}, "human_verdict": "HIDDEN_SOURCE_ANSWER"},
            {"id": "N1", "stem": "", "quantity_a": "8^10", "quantity_b": "4^15",
             "correct_answer": "C", "format": "QC", "skill": "Arithmetic", "subskill": "powers", "difficulty": "2",
             "options": {"C": "equal"}},
            {"id": "M2", "stem": "Which group has more?", "correct_answer": "A", "format": "MC",
             "skill": "Data", "subskill": "chart", "difficulty": "3", "options": {"A": "one", "B": "two"},
             "stimulus": "Group one: 40%; Group two: 30%", "asset": "<svg><text>exact source</text></svg>"},
        ]
        self.exported = []
        for item, question_id in zip(self.items, ["P1", "Q1", "P2"]):
            stem = item["stem"] or f"Quantity A:  {item['quantity_a']}\nQuantity B:  {item['quantity_b']}"
            self.exported.append({"question_id": question_id, "internal_id": item["id"], "question": stem,
                                  "correct_answer": item["correct_answer"], "format": item["format"],
                                  "format_code": item["format"], "skill": item["skill"], "subskill": item["subskill"],
                                  "difficulty": item["difficulty"], "options_json": json.dumps(item["options"])})
        bank = self.root / "question_bank_quant.csv"
        with bank.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.exported[0]))
            writer.writeheader(); writer.writerows(self.exported)
        (self.root / "release_manifest.json").write_text(json.dumps({"files": [
            {"path": bank.name, "sha256": hashlib.sha256(bank.read_bytes()).hexdigest()}]}))
        self.write_source()
        self.pretest = [{"question_id": r["internal_id"], "question_text": r["question"],
                         "correct_answer": r["correct_answer"], "format": r["format_code"], "skill": r["skill"],
                         "subskill": r["subskill"], "difficulty": r["difficulty"], "options_json": r["options_json"],
                         "student_answer": "B", "is_correct": "0", "reviewer_reasoning": "HIDDEN_HUMAN_REASON",
                         "ai_model_preset_id": "HIDDEN_AUTHOR"} for r in self.exported]

    def write_source(self):
        self.source.write_text(json.dumps({"domain": "gre_quant", "pretest": self.items, "posttest": []}))
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def load(self):
        return PretestContextBank.load(self.root, source_bank=self.source, expected_source_sha256=self.source_hash)

    def test_exact_join_recovers_quantities_and_shared_data_without_mutation_or_secrets(self):
        before = copy.deepcopy(self.pretest)
        result = self.load().enrich_pretest(self.pretest)
        self.assertEqual(result[0]["question_context"]["quantity_a"], "x + 1")
        self.assertEqual(result[1]["question_context"]["quantity_b"], "4^15")
        self.assertEqual(result[2]["question_context"]["shared_data"], self.items[2]["stimulus"])
        self.assertIn("not rendered", result[2]["question_context"]["figure_status"])
        self.assertNotIn("<svg", json.dumps(result))
        self.assertNotIn("HIDDEN", json.dumps(result))
        self.assertEqual(result[0]["student_answer"], "B")
        self.assertEqual(result[0]["is_correct"], "0")
        self.assertEqual(self.pretest, before)

    def test_external_bank_requires_pinned_matching_hash(self):
        for hash_value in (None, "incorrect"):
            with self.assertRaisesRegex(ValueError, "pinned SHA256"):
                PretestContextBank.load(self.root, source_bank=self.source, expected_source_sha256=hash_value)

    def test_changed_source_shared_fields_fail_even_with_updated_source_hash(self):
        for field, changed in (("stem", "a different question"), ("correct_answer", "D"),
                               ("options", {"A": "different choice"})):
            original = copy.deepcopy(self.items[0])
            with self.subTest(field=field):
                self.items[0][field] = changed
                self.write_source()
                with self.assertRaisesRegex(ValueError, "differ[s]? from SSOT"):
                    self.load()
                self.items[0] = original

    def test_pretest_aliases_work_but_changed_student_question_context_is_rejected(self):
        bank = self.load()
        aliases = copy.deepcopy(self.pretest)
        aliases[0]["question_id"] = "P1"
        self.assertEqual(bank.enrich_pretest(aliases)[0]["question_context"]["quantity_a"], "x + 1")
        aliases[0]["correct_answer"] = "D"
        with self.assertRaisesRegex(ValueError, "does not match SSOT"):
            bank.enrich_pretest(aliases)

    def test_ssot_only_flags_missing_quantities_and_never_invents_chart_data(self):
        result = PretestContextBank.load(self.root).enrich_pretest(self.pretest)
        self.assertNotIn("quantity_a", result[0]["question_context"])
        self.assertIn("absent", result[0]["question_context"]["missing_evidence"][0])
        self.assertEqual(result[1]["question_context"]["quantity_a"], "8^10")
        self.assertNotIn("shared_data", result[2]["question_context"])

    def test_changed_ssot_question_bank_is_rejected(self):
        path = self.root / "question_bank_quant.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "release manifest"):
            self.load()


if __name__ == "__main__":
    unittest.main()
