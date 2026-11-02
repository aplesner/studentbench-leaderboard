"""Paired differences must share clusters, reference, direction, and support."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.paired_deltas import BASELINE, EXAMPLE, paired_bootstrap_deltas, run_paired_deltas
from studentbench_leaderboard.ranking import fit_bradley_terry
from studentbench_leaderboard.study_analysis import analyze_round
from test_pipeline import fixture


def pairs(subjects=4, reverse=False):
    return [{"subject": f"s{s}", "pair_id": f"p{s}_{a}{b}", "model_a": a,
             "model_b": b, "value": 1 - value if reverse else value, "weight": 1.0}
            for s in range(subjects) for a, b, value in (("a", "b", .8), ("b", "c", .7), ("a", "c", .9))]


class PairedDeltaTests(unittest.TestCase):
    def test_identical_judges_have_exactly_zero_paired_differences(self):
        human, baseline = pairs(), pairs(reverse=True)
        before = deepcopy(baseline)
        result = paired_bootstrap_deltas(human, baseline, {"same": baseline}, models=["a", "b", "c"],
                                         n_bootstrap=25, seed=2)
        comparison = result["comparisons"]["same"]
        self.assertEqual(comparison["n_valid_fits"], 25)
        for entry in comparison["metrics"].values():
            self.assertEqual((entry["delta"], entry["low"], entry["high"]), (0, 0, 0))
            self.assertEqual(entry["proportion_delta_gt_zero"], 0)
            self.assertEqual(entry["proportion_delta_equal_zero"], 1)
        self.assertEqual(baseline, before)
        self.assertEqual(result, paired_bootstrap_deltas(human, baseline, {"same": baseline},
                                                        models=["a", "b", "c"], n_bootstrap=25, seed=2))

    def test_raw_rank_error_difference_has_opposite_improvement_direction(self):
        result = paired_bootstrap_deltas(pairs(), pairs(reverse=True), {"better": pairs()},
                                         models=["a", "b", "c"], n_bootstrap=10)
        metrics = result["comparisons"]["better"]["metrics"]
        self.assertEqual(metrics["kendall_tau_b"]["delta"], 2)
        self.assertEqual(metrics["spearman"]["delta"], 2)
        self.assertEqual(metrics["pair_order_agreement"]["delta"], 1)
        self.assertLess(metrics["rank_mae"]["delta"], 0)
        self.assertEqual(metrics["rank_mae"]["proportion_delta_gt_zero"], 0)
        self.assertEqual(metrics["rank_mae"]["proportion_improved"], 1)
        self.assertFalse(metrics["rank_mae"]["higher_is_better"])

    def test_every_fit_uses_identical_subject_multiplicities_and_retains_all_pairs(self):
        observations = []

        def record(rows, **kwargs):
            observations.append(deepcopy(rows))
            return fit_bradley_terry(rows, **kwargs)

        with patch("studentbench_leaderboard.paired_deltas.fit_bradley_terry", side_effect=record):
            paired_bootstrap_deltas(pairs(), pairs(), {"one": pairs(), "two": pairs(reverse=True)},
                                    models=["a", "b", "c"], n_bootstrap=15, seed=3)
        self.assertEqual(len(observations), 4 * 16)  # Four point fits, then four per shared draw.
        repeated_cluster_seen = False
        for start in range(4, len(observations), 4):
            fingerprints = [[(r["subject"], r["pair_id"], r["weight"]) for r in rows]
                            for rows in observations[start:start + 4]]
            self.assertTrue(all(rows == fingerprints[0] for rows in fingerprints))
            counts = {}
            for subject, _, weight in fingerprints[0]:
                counts.setdefault(subject, []).append(weight)
                repeated_cluster_seen |= weight > 1
            self.assertTrue(all(len(weights) == 3 and len(set(weights)) == 1 for weights in counts.values()))
            self.assertEqual(sum(weights[0] for weights in counts.values()), 4)
        self.assertTrue(repeated_cluster_seen)

    def test_sparse_connectivity_counts_are_visible_and_missing_candidate_does_not_shrink_cohort(self):
        human = [{"subject": "s1", "pair_id": "p1", "model_a": "a", "model_b": "b", "value": .8},
                 {"subject": "s2", "pair_id": "p2", "model_a": "b", "model_b": "c", "value": .7}]
        result = paired_bootstrap_deltas(human, human, {"same": human, "absent": []},
                                         models=["a", "b", "c"], n_bootstrap=100, seed=5)
        valid = result["comparisons"]["same"]["n_valid_fits"]
        self.assertGreater(valid, 20)
        self.assertLess(valid, 80)
        self.assertEqual(result["comparisons"]["same"]["n_disconnected_draws"], 100 - valid)
        missing = result["comparisons"]["absent"]
        self.assertEqual(result["n_subjects"], 2)
        self.assertEqual(missing["coverage"]["candidate_scored_subjects"], 0)
        self.assertEqual(missing["n_disconnected_draws"], 100)
        self.assertEqual(missing["n_valid_fits"], 0)
        self.assertTrue(all(v["delta"] is None and v["low"] is None for v in missing["metrics"].values()))

    def test_undefined_correlations_do_not_become_zero_differences(self):
        tied = [{**row, "value": .5} for row in pairs()]
        result = paired_bootstrap_deltas(tied, tied, {"tied": tied}, models=["a", "b", "c"], n_bootstrap=5)
        comparison = result["comparisons"]["tied"]
        self.assertEqual(comparison["n_valid_fits"], 5)
        self.assertEqual(comparison["metrics"]["kendall_tau_b"]["n_valid"], 0)
        self.assertIsNone(comparison["metrics"]["spearman"]["delta"])
        self.assertEqual(comparison["metrics"]["rank_mae"]["n_valid"], 5)

    def test_unaggregated_duplicates_and_outside_pairs_are_rejected(self):
        rows = pairs()
        with self.assertRaisesRegex(ValueError, "aggregate criteria"):
            paired_bootstrap_deltas(rows + rows[:1], rows, {"candidate": rows}, models=["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "outside the human reference"):
            paired_bootstrap_deltas(rows, rows, {"candidate": [{**rows[0], "pair_id": "unreviewed"}]},
                                    models=["a", "b", "c"])


class ExtensionPairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ssot, self.study = self.root / "ssot", self.root / "study"
        fixture(self.ssot)
        grades_path = self.ssot / "2_human_ai_vs_ai_lesson_plan_rubric_grading_data/gre_quant/rubric_grades.csv"
        with grades_path.open() as stream:
            grades = list(csv.DictReader(stream))
        for row in grades:
            row["rubric_question_id"] = row["question_variant_id"]
        with grades_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(grades[0]))
            writer.writeheader()
            writer.writerows(grades)
        dataset = Dataset(self.ssot, "gre_quant")
        (self.study / "calls").mkdir(parents=True)
        pair_id = digest(["s1", "a", "z"])[:20]
        entry = {"pair_id": pair_id, "subject": "s1", "plan_ids": ["a", "z"],
                 "model_a": "AUTHOR_a_SECRET", "model_b": "AUTHOR_z_SECRET",
                 "grading_id": "g1", "human_reviewed": True}
        manifest = {"seed": 7, "instrument": "gre_quant", "dataset_sha256": dataset.fingerprint,
                    "splits": {"validation": ["s1"]}, "pair_catalog": {"validation": [entry]},
                    "evaluation_role": "validation_extension_after_original_test_exposure"}
        (self.study / "manifest.json").write_text(json.dumps(manifest))
        names = [BASELINE, EXAMPLE, "provider"]
        (self.study / "variants.json").write_text(json.dumps([{"variant": name} for name in names]))
        for name in names:
            record = {**entry, "variant": name, "draw": 0, "dataset_sha256": dataset.fingerprint,
                      "criteria": [{"id": c["id"], "scale_id": c["scale_id"]} for c in dataset.rubric("g1")],
                      "normalized_ratings": {"format_old": {"score": 2 if name == EXAMPLE else -2},
                                             "accuracy": {"score": 0}}}
            (self.study / "calls" / f"{name}--{pair_id}--0.json").write_text(json.dumps(record))
        for name, file, members in (("provider_validation", "provider_jobs.json", [BASELINE, "provider"]),
                                    ("example_validation", "example_jobs.json", [BASELINE, EXAMPLE])):
            jobs = [{"variant": member, "pair_id": pair_id, "draw": 0} for member in members]
            (self.study / file).write_text(json.dumps(jobs))
            analyze_round(self.study, name, self.study / file, ssot=self.ssot, bootstrap=0)

    def test_wrapper_reproduces_main_point_metrics_and_writes_both_contrasts_offline(self):
        result = run_paired_deltas(self.study, ssot=self.ssot, n_bootstrap=5)
        self.assertEqual(set(result["comparisons"]), {EXAMPLE, "provider"})
        self.assertEqual(result["comparisons"][EXAMPLE]["role"], "primary_example")
        self.assertEqual(result["comparisons"]["provider"]["role"], "exploratory_provider")
        self.assertEqual(result["n_subjects"], 1)
        self.assertTrue((self.study / "paired_deltas.json").exists())
        self.assertEqual(result, json.loads((self.study / "paired_deltas.json").read_text()))

    def test_wrapper_refuses_incomplete_calls_and_changed_main_metrics(self):
        path = self.study / "example_validation_analysis.json"
        analysis = json.loads(path.read_text())
        analysis["judges"][0]["metrics"]["rank_mae"] += .1
        path.write_text(json.dumps(analysis))
        with self.assertRaisesRegex(ValueError, "point metric differs"):
            run_paired_deltas(self.study, ssot=self.ssot, n_bootstrap=0)
        next((self.study / "calls").glob("provider--*.json")).unlink()
        with self.assertRaisesRegex(ValueError, "Executable schedule is incomplete"):
            run_paired_deltas(self.study, ssot=self.ssot, n_bootstrap=0)

    def test_example_only_phase_never_reads_provider_files_and_matches_full_phase(self):
        full = run_paired_deltas(self.study, ssot=self.ssot, n_bootstrap=10)
        for filename in ("provider_validation_analysis.json", "provider_jobs.json"):
            (self.study / filename).unlink()
        next((self.study / "calls").glob("provider--*.json")).unlink()
        example = run_paired_deltas(self.study, ssot=self.ssot, n_bootstrap=10, include_provider=False)
        self.assertEqual(set(example["comparisons"]), {EXAMPLE})
        self.assertEqual(example["comparisons"][EXAMPLE], full["comparisons"][EXAMPLE])
        self.assertEqual(example["resample_counts_sha256"], full["resample_counts_sha256"])
        self.assertEqual(example["complete_phases"], ["example_validation"])
        self.assertEqual(example["pending_phases"], ["provider_validation"])
        self.assertEqual(example["status"], "example_complete_provider_pending")

    def test_verified_terminal_provider_job_remains_missing_in_paired_metrics(self):
        path = next((self.study / "calls").glob("provider--*.json"))
        record = json.loads(path.read_text())
        key = (record["variant"], record["pair_id"], record.get("draw", 0))
        path.unlink()
        analyze_round(self.study, "provider_validation", self.study / "provider_jobs.json", ssot=self.ssot, bootstrap=0)
        row = {"variant": key[0], "pair_id": key[1], "draw": key[2], "outcome": "missing", "rating_imputed": False}
        def terminal(study, jobs, **kwargs):
            return {key: row} if key in {(job["variant"], job["pair_id"], job.get("draw", 0)) for job in jobs} else {}
        from unittest.mock import patch
        with patch("studentbench_leaderboard.extension_terminal.load_terminal_failures", side_effect=terminal):
            result = run_paired_deltas(self.study, ssot=self.ssot, n_bootstrap=5)
        self.assertEqual(result["status"], "complete_with_terminal_missing_outcomes")
        self.assertEqual(result["terminal_failures"]["n_jobs"], 1)
        self.assertEqual(result["comparisons"]["provider"]["coverage"]["candidate_scored_subjects"], 0)
        self.assertIsNone(result["comparisons"]["provider"]["metrics"]["kendall_tau_b"]["delta"])
        self.assertEqual(result["n_subjects"], 1)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
