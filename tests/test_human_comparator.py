"""Shared references, common scored support and partition-level counting."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "experiments/report-revision/human_comparator.py"
spec = importlib.util.spec_from_file_location("human_comparator", path)
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


class HumanComparatorTests(unittest.TestCase):
    def test_copy_of_comparator_matches_its_shared_reference_and_orientations_count_once(self):
        rows = [{"subject": "s", "pair_id": a + b, "model_a": a, "model_b": b,
                 "criterion": "q01|scale", "criterion_family": "q01", "value": 1.0}
                for a, b in (("a", "b"), ("a", "c"), ("b", "c"))]
        left = [{**row, "reviewer": "left"} for row in rows]
        right = [{**row, "reviewer": "right", "value": 0.0 if row["pair_id"] == "ab" else 1.0}
                 for row in rows]
        ai = [{key: value for key, value in row.items() if key != "reviewer"} for row in right]
        # A missing AI criterion must be excluded from all three sides rather
        # than counted as a tie or retained only in the human reference.
        extra = {**rows[0], "criterion": "q02|scale", "criterion_family": "q02"}
        left.append({**extra, "reviewer": "left"})
        right.append({**extra, "reviewer": "right", "value": 0.0})
        ai.append({**extra, "value": None})
        result = comparator.compare_partition(left, right, ai, ["a", "b", "c"])
        self.assertTrue(result["connected"])
        self.assertEqual([result["support"][side]["criterion_units"] for side in ("left", "right", "ai")], [3, 3, 3])
        first, second = result["orientations"]
        self.assertEqual(first["reference_group"], "left")
        self.assertEqual(first["ai_vs_reference"], first["human_vs_reference"])
        self.assertAlmostEqual(second["ai_vs_reference"]["kendall_tau_b"], 1.0)
        self.assertAlmostEqual(result["metrics"]["kendall_tau_b"]["ai_vs_reference"],
                               (first["ai_vs_reference"]["kendall_tau_b"] + second["ai_vs_reference"]["kendall_tau_b"]) / 2)
        summary = comparator.summarize_partitions([result])
        self.assertEqual(summary["kendall_tau_b"]["delta_ai_minus_human"]["n"], 1)

    def test_disconnected_support_is_excluded_and_shared_reviewers_rejected(self):
        row = {"subject": "s", "pair_id": "ab", "model_a": "a", "model_b": "b",
               "criterion": "q01|scale", "criterion_family": "q01", "value": 1.0}
        left, right = [{**row, "reviewer": "left"}], [{**row, "reviewer": "right"}]
        result = comparator.compare_partition(left, right, [row], ["a", "b", "c"])
        self.assertFalse(result["connected"])
        self.assertEqual(comparator.summarize_partitions([result])["kendall_tau_b"]["delta_ai_minus_human"]["n"], 0)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            comparator.compare_partition(left, left, [row], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
