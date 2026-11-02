"""Offline study analysis must preserve identities, denominators, and holdouts."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.study_analysis import (
    analyze_legacy, analyze_round, augmentation_analysis, combine_votes, density_analysis,
    load_derived_judges, materialize_judge, scheduled_judges,
)
from studentbench_leaderboard.ranking import fit_bradley_terry
from test_pipeline import fixture


class VoteCombinationTests(unittest.TestCase):
    def test_weighted_soft_votes_preserve_fraction_and_require_every_member(self):
        self.assertEqual(combine_votes([1, 0, 0], "soft", [1, .5, .5]), .5)
        self.assertAlmostEqual(combine_votes([1, 0, 0], "soft"), 1 / 3)
        self.assertIsNone(combine_votes([1, None, 0], "soft", [1, .5, .5]))
        for weights in ([1], [1, 0], [1, -1], [1, float("nan")], [1, float("inf")], [1, True]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                combine_votes([1, 0], "soft", weights)
        with self.assertRaises(ValueError):
            combine_votes([1, 0], "strict", [1, 1])

    def test_soft_disagreement_retains_fraction_and_missing_never_becomes_tie(self):
        self.assertEqual(combine_votes([1, 0], "soft"), .5)
        self.assertEqual(combine_votes([1, 1, 0, .5], "soft"), .625)
        self.assertIsNone(combine_votes([1, None], "soft"))
        self.assertIsNone(combine_votes([None, None], "soft"))

    def test_strict_majority_counts_missing_in_denominator(self):
        self.assertIsNone(combine_votes([1, 0], "strict"))
        self.assertIsNone(combine_votes([1, 1, None, None], "strict"))
        self.assertEqual(combine_votes([1, 1, 1, None], "strict"), 1)
        self.assertEqual(combine_votes([.5, .5, 0], "strict"), .5)

    def test_missing_panel_call_makes_pair_incomplete(self):
        definition = {"members": [("a", 0), ("b", 0)], "aggregation": "soft"}
        pairs = {"p": {"subject": "s", "model_a": "m1", "model_b": "m2"}}
        units = {"p": [{"criterion": "q|scale", "criterion_family": "q"}]}
        record = {"criteria": [{"id": "q", "scale_id": "scale"}],
                  "normalized_ratings": {"q": {"score": 1}}, "estimated_usd": .1}
        rows, counts = materialize_judge(definition, ["p"], pairs, units, {("a", "p", 0): record})
        self.assertIsNone(rows[0]["value"])
        self.assertEqual(counts["missing_calls"], 1)
        self.assertEqual(counts["criterion_coverage"], 0)
        self.assertEqual(counts["estimated_usd"], .1)

    def test_strategy_schedule_requires_declared_members_and_second_order(self):
        variants = {name: {} for name in ("luna_high", "terra_high", "sol_high", "astra_high")}
        schedule = {(name, "p", 0) for name in variants} | {("luna_high", "p", 1)}
        names = {row["variant"] for row in scheduled_judges(variants, schedule)}
        self.assertIn("panel_high", names)
        self.assertIn("panel_high_strict", names)
        self.assertIn("luna_high_balanced", names)
        self.assertNotIn("panel_low", names)
        self.assertNotIn("terra_high_balanced", names)

    def test_derived_schedule_requires_every_member_draw_on_every_pair(self):
        variants = {"first": {}, "second": {}}
        derived = [{"variant": "custom_panel", "members": [("first", 0), ("second", 1)], "aggregation": "soft"}]
        schedule = {("first", "p1", 0), ("first", "p2", 0), ("second", "p1", 1)}
        self.assertNotIn("custom_panel", {r["variant"] for r in scheduled_judges(variants, schedule, derived)})
        schedule.add(("second", "p2", 1))
        self.assertIn("custom_panel", {r["variant"] for r in scheduled_judges(variants, schedule, derived)})

    def test_weighted_panel_abstention_is_missing_even_when_other_members_agree(self):
        definition = {"members": [("a", 0), ("b", 0)], "aggregation": "soft", "weights": [1, .5]}
        pairs = {"p": {"subject": "s", "model_a": "m1", "model_b": "m2"}}
        units = {"p": [{"criterion": "q|scale", "criterion_family": "q"}]}
        record = {"criteria": [{"id": "q", "scale_id": "scale"}], "normalized_ratings": {"q": {"score": 2}}}
        abstention = {**record, "normalized_ratings": {"q": {"score": None}}}
        rows, counts = materialize_judge(definition, ["p"], pairs, units,
                                         {("a", "p", 0): record, ("b", "p", 0): abstention})
        self.assertIsNone(rows[0]["value"])
        self.assertEqual(counts["completed_pairs"], 1)
        self.assertEqual(counts["missing_calls"], 0)
        self.assertEqual(counts["scored_criteria"], 0)


class DerivedDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.variants = {"a": {"judge_id": "judge_5", "family": "Family A"},
                         "b": {"judge_id": "judge_5", "family": "Family B"}}
        self.valid = '''schema_version = 1
[[judges]]
judge_id = "judge_10"
variant = "panel"
label = "Panel"
aggregation = "soft"
[[judges.members]]
variant = "a"
draw = 0
weight = 1.0
[[judges.members]]
variant = "b"
draw = 0
weight = 0.5
'''

    def test_absent_optional_file_preserves_existing_definition_behavior(self):
        self.assertEqual(load_derived_judges(self.root, self.variants), [])

    def test_known_historical_description_preserves_builtin_panels_but_modified_recipes_fail(self):
        original = Path(__file__).resolve().parents[1] / "experiments/ranking-study/derived_judges.toml"
        historical = original.read_text()
        path = self.root / "derived_judges.toml"
        path.write_text(historical)
        variants = {f"{model}_{effort}": {} for model in ("luna", "terra", "sol", "astra")
                    for effort in ("low", "high")}
        variants["enhanced_luna_high"] = {}
        schedule = {(name, "pair", 0) for name in variants} | {("enhanced_luna_high", "pair", 1)}
        derived = load_derived_judges(self.root, variants)
        self.assertEqual(derived, [])
        self.assertEqual(scheduled_judges(variants, schedule, derived), scheduled_judges(variants, schedule))
        definitions = {entry["variant"]: entry for entry in scheduled_judges(variants, schedule, derived)}
        self.assertEqual(definitions["panel_high"]["members"], [(f"{model}_high", 0)
                                                               for model in ("luna", "terra", "sol", "astra")])
        self.assertEqual(definitions["panel_low_strict"]["aggregation"], "strict")
        self.assertEqual(definitions["enhanced_luna_high_balanced"]["members"],
                         [("enhanced_luna_high", 0), ("enhanced_luna_high", 1)])
        self.assertEqual(path.read_text(), historical)
        for changed in (historical.replace('"astra"', '"unknown"'),
                        historical.replace('draws = [0, 1]', 'draws = [0, 2]'),
                        historical.replace('draws = [0, 1]', 'draws = [false, true]'),
                        historical.replace('draws = [0, 1]', 'draws = [0.0, 1.0]'),
                        historical + '\n[judge_99]\nstrategy = "ignored"\n'):
            path.write_text(changed)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Invalid derived-judge file schema"):
                load_derived_judges(self.root, variants)
        path.write_text(historical)
        (self.root / "derived_judges_manifest.json").write_text(json.dumps({"file_sha256": "incorrect"}))
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            load_derived_judges(self.root, variants)

    def test_definition_records_exact_file_hash_normalized_weights_and_member_families(self):
        path = self.root / "derived_judges.toml"
        path.write_text(self.valid)
        definition, = load_derived_judges(self.root, self.variants)
        self.assertEqual(definition["definition_file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(definition["members"], [("a", 0), ("b", 0)])
        self.assertEqual(definition["weights"], [1, .5])
        self.assertAlmostEqual(sum(definition["normalized_weights"]), 1)
        self.assertEqual(definition["weighted_members"][0]["family"], "Family A")
        self.assertTrue(definition["judge_version"].startswith("judge_10_"))

    def test_unknown_duplicate_bad_draw_and_nonfinite_weights_are_rejected(self):
        cases = [self.valid.replace('variant = "b"', 'variant = "unknown"'),
                 self.valid.replace('variant = "b"', 'variant = "a"'),
                 self.valid.replace('draw = 0', 'draw = -1'),
                 self.valid.replace('draw = 0', 'draw = true'),
                 self.valid.replace('weight = 0.5', 'weight = nan'),
                 self.valid.replace('weight = 0.5', 'weight = inf'),
                 self.valid.replace('weight = 0.5', 'weight = 0'),
                 self.valid.replace('judge_id = "judge_10"', 'judge_id = "judge_5"')]
        for text in cases:
            with self.subTest(text=text):
                (self.root / "derived_judges.toml").write_text(text)
                with self.assertRaises(ValueError):
                    load_derived_judges(self.root, self.variants)

    def test_frozen_recipe_file_cannot_change(self):
        path = self.root / "derived_judges.toml"
        path.write_text(self.valid)
        (self.root / "derived_judges_manifest.json").write_text(json.dumps({
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        path.write_text(self.valid + "\n# changed frozen file\n")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            load_derived_judges(self.root, self.variants)

    def test_availability_amendment_is_additive_with_its_own_frozen_hash(self):
        original = self.root / "derived_judges.toml"
        original.write_text(self.valid)
        original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        amendment = self.root / "derived_judges_available.toml"
        amendment.write_text(self.valid.replace('judge_id = "judge_10"', 'judge_id = "judge_12"')
                             .replace('variant = "panel"', 'variant = "available_panel"'))
        frozen = self.root / "derived_judges_available_manifest.json"
        frozen.write_text(json.dumps({"file_sha256": hashlib.sha256(amendment.read_bytes()).hexdigest()}))
        first, second = load_derived_judges(self.root, self.variants)
        self.assertEqual(first["definition_file_sha256"], original_hash)
        self.assertEqual(first["definition_file"], original.name)
        self.assertEqual(second["definition_file"], amendment.name)
        self.assertEqual(second["judge_id"], "judge_12")
        amendment.write_text(amendment.read_text() + "\n# changed availability recipe\n")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            load_derived_judges(self.root, self.variants)
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), original_hash)

    def test_recipe_names_and_ids_cannot_conflict_across_files(self):
        (self.root / "derived_judges.toml").write_text(self.valid)
        for amendment in (self.valid.replace('judge_id = "judge_10"', 'judge_id = "judge_12"'),
                          self.valid.replace('variant = "panel"', 'variant = "available_panel"')):
            with self.subTest(amendment=amendment):
                (self.root / "derived_judges_available.toml").write_text(amendment)
                with self.assertRaisesRegex(ValueError, "across recipe files"):
                    load_derived_judges(self.root, self.variants)

    def test_unrecognized_experimental_recipe_filenames_are_not_automatically_loaded(self):
        (self.root / "derived_judges_unplanned.toml").write_text(self.valid)
        self.assertEqual(load_derived_judges(self.root, self.variants), [])

    def test_prespecified_nine_member_panels_have_equal_member_or_equal_family_mass(self):
        directory = Path(__file__).resolve().parents[1] / "experiments/ranking-extension"
        variants = {v["variant"]: v for v in json.loads((directory / "variants.json").read_text())}
        definitions = load_derived_judges(directory, variants)
        equal, balanced = definitions[:2]
        self.assertEqual(len(equal["members"]), 9)
        self.assertEqual(equal["members"], balanced["members"])
        self.assertEqual(equal["judge_id"], "judge_10")
        self.assertEqual(balanced["judge_id"], "judge_11")
        self.assertTrue(all(abs(w - 1 / 9) < 1e-12 for w in equal["normalized_weights"]))
        families = {}
        for member in balanced["weighted_members"]:
            families[member["family"]] = families.get(member["family"], 0) + member["normalized_weight"]
        self.assertEqual(set(families), {"Google", "Anthropic", "DeepSeek", "Moonshot", "Z.AI", "Meta"})
        self.assertTrue(all(abs(weight - 1 / 6) < 1e-12 for weight in families.values()))

    def test_availability_panels_omit_only_muse_and_preserve_five_equal_family_masses(self):
        directory = Path(__file__).resolve().parents[1] / "experiments/ranking-extension"
        variants = {v["variant"]: v for v in json.loads((directory / "variants.json").read_text())}
        original, _, equal, balanced = load_derived_judges(directory, variants)
        self.assertEqual(equal["judge_id"], "judge_12")
        self.assertEqual(balanced["judge_id"], "judge_13")
        self.assertEqual(set(original["members"]) - set(equal["members"]), {("or_muse13_high", 0)})
        self.assertEqual(equal["members"], balanced["members"])
        self.assertEqual(len(equal["members"]), 8)
        self.assertTrue(all(abs(w - 1 / 8) < 1e-12 for w in equal["normalized_weights"]))
        families = {}
        for member in balanced["weighted_members"]:
            families[member["family"]] = families.get(member["family"], 0) + member["normalized_weight"]
        self.assertEqual(set(families), {"Google", "Anthropic", "DeepSeek", "Moonshot", "Z.AI"})
        self.assertTrue(all(abs(weight - 1 / 5) < 1e-12 for weight in families.values()))


class StudyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ssot = self.root / "ssot"
        fixture(self.ssot)
        # The compact older fixture omits this real SSOT field.
        import csv
        grades_path = self.ssot / "2_human_ai_vs_ai_lesson_plan_rubric_grading_data/gre_quant/rubric_grades.csv"
        with grades_path.open() as stream:
            grades = list(csv.DictReader(stream))
        for row in grades:
            row["rubric_question_id"] = row["question_variant_id"]
        with grades_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(grades[0]))
            writer.writeheader()
            writer.writerows(grades)
        self.dataset = Dataset(self.ssot, "gre_quant")
        self.study = self.root / "study"
        (self.study / "calls").mkdir(parents=True)
        self.pair = digest(["s1", "a", "z"])[:20]
        entry = {"pair_id": self.pair, "subject": "s1", "plan_ids": ["a", "z"],
                 "model_a": "AUTHOR_a_SECRET", "model_b": "AUTHOR_z_SECRET",
                 "grading_id": "g1", "human_reviewed": True}
        self.manifest = {"seed": 1, "instrument": "gre_quant", "dataset_sha256": self.dataset.fingerprint,
                         "splits": {"validation": ["s1"], "test": ["unseen"]},
                         "pair_catalog": {"validation": [entry], "test": []}}
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        (self.study / "variants.json").write_text(json.dumps([{"variant": "complete"}, {"variant": "missing"}]))
        self.jobs = [{"variant": name, "pair_id": self.pair, "draw": 0} for name in ("complete", "missing")]
        (self.study / "jobs.json").write_text(json.dumps(self.jobs))
        criteria = [{"id": row["id"], "scale_id": row["scale_id"]} for row in self.dataset.rubric("g1")]
        self.call = {**entry, **self.jobs[0], "criteria": criteria,
                     "dataset_sha256": self.dataset.fingerprint,
                     "ratings": {"format_old": {"score": 2}, "accuracy": {"score": 0}},
                     "normalized_ratings": {"format_old": {"score": -2}, "accuracy": {"score": 0}},
                     "estimated_usd": .25}
        self.call_path = self.study / "calls" / f"complete--{self.pair}--0.json"
        self.call_path.write_text(json.dumps(self.call))

    def test_normalized_votes_and_human_pair_ids_match_without_reference_drift(self):
        result = analyze_round(self.study, "round", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)
        self.assertFalse(result["complete"])
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["n_human_reviewed_pairs"], 1)
        self.assertEqual(result["human_coverage"]["exact_criteria"], 2)
        complete, missing = result["judges"]
        self.assertEqual(complete["metrics"]["probability_mae"], 0)
        self.assertEqual(complete["metrics"]["kendall_tau_b"], 1)
        self.assertEqual(complete["estimated_usd"], .25)
        self.assertEqual(missing["coverage"]["criterion_coverage"], 0)
        self.assertIsNone(missing["metrics"]["kendall_tau_b"])
        self.assertTrue((self.study / "round_analysis.json").exists())

    def test_optional_frozen_panel_is_materialized_with_definition_and_missing_coverage(self):
        recipe = '''schema_version = 1
[[judges]]
judge_id = "judge_10"
variant = "weighted_panel"
label = "Weighted panel"
aggregation = "soft"
[[judges.members]]
variant = "complete"
draw = 0
weight = 2.0
[[judges.members]]
variant = "missing"
draw = 0
weight = 1.0
'''
        (self.study / "derived_judges.toml").write_text(recipe)
        result = analyze_round(self.study, "partial", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)
        self.assertTrue(result["derived_judge_definitions"][0]["all_members_scheduled"])
        panel = next(j for j in result["judges"] if j["variant"] == "weighted_panel")
        self.assertEqual(panel["judge_id"], "judge_10")
        self.assertEqual(panel["weights"], [2, 1])
        self.assertEqual(panel["coverage"]["missing_calls"], 1)
        self.assertEqual(panel["coverage"]["scored_criteria"], 0)
        self.assertIsNone(panel["metrics"]["kendall_tau_b"])
        self.assertEqual(panel["definition_file_sha256"], hashlib.sha256(recipe.encode()).hexdigest())
        second = {**self.call, "variant": "missing", "normalized_ratings": {
            "format_old": {"score": 2}, "accuracy": {"score": 0}}}
        (self.study / "calls" / f"missing--{self.pair}--0.json").write_text(json.dumps(second))
        complete = analyze_round(self.study, "complete", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)
        panel = next(j for j in complete["judges"] if j["variant"] == "weighted_panel")
        self.assertEqual(panel["coverage"]["scored_criteria"], 2)
        self.assertEqual(panel["coverage"]["missing_calls"], 0)
        self.assertEqual(panel["n_calls"], 2)
        self.assertEqual(panel["coverage"]["disagreeing_member_criteria"], 1)

    def test_wrong_cached_plan_identity_is_rejected(self):
        self.call["plan_ids"] = ["z", "a"]
        self.call_path.write_text(json.dumps(self.call))
        with self.assertRaisesRegex(ValueError, "plan_ids"):
            analyze_round(self.study, "round", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)

    def test_multiple_data_splits_cannot_be_combined(self):
        entry = {**self.manifest["pair_catalog"]["validation"][0], "pair_id": "other", "subject": "unseen"}
        self.manifest["pair_catalog"]["test"] = [entry]
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        self.jobs.append({"variant": "complete", "pair_id": "other", "draw": 0})
        (self.study / "jobs.json").write_text(json.dumps(self.jobs))
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            analyze_round(self.study, "round", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)

    def test_test_analysis_requires_matching_selection_lock_before_loading_labels(self):
        self.manifest["splits"] = {"test": ["s1"]}
        self.manifest["pair_catalog"] = {"test": self.manifest["pair_catalog"]["validation"]}
        for updates in ({}, {"selection_locked": True},
                        {"selection_locked": True, "selection": {"variant": "complete"},
                         "selection_sha256": "wrong"}):
            with self.subTest(updates=updates):
                manifest = {**self.manifest, **updates}
                (self.study / "manifest.json").write_text(json.dumps(manifest))
                with patch("studentbench_leaderboard.study_analysis.Dataset") as dataset_loader:
                    with self.assertRaisesRegex(ValueError, "selection lock hash"):
                        analyze_round(self.study, "test", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)
                    dataset_loader.assert_not_called()
        selection = {"variant": "complete"}
        self.manifest.update(selection_locked=True, selection=selection, selection_sha256=digest(selection))
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        result = analyze_round(self.study, "test", self.study / "jobs.json", bootstrap=0, ssot=self.ssot)
        self.assertEqual(result["split"], "test")
        self.assertIn("exclude_q07", result["judges"][0]["rubric_ablations"])

    def test_density_keeps_full_reference_and_reports_sparse_connectivity(self):
        rows = [{"subject": f"s{i}", "pair_id": str(i), "model_a": a, "model_b": b, "value": 1}
                for i, (a, b) in enumerate((("a", "b"), ("b", "c"), ("a", "c"), ("a", "b")))]
        fit = fit_bradley_terry(rows)
        result = density_analysis(rows, fit, ["a", "b", "c"], repeats=10, seed=1)
        self.assertEqual(result["experiments"][0]["valid_fit_fraction"], 0)
        self.assertEqual(result["experiments"][-1]["summary"]["probability_mae"]["mean"], 0)
        self.assertEqual(result["experiments"][-1]["coverage"]["n_pairs"]["mean"], 4)

    def test_augmentation_retains_reviewed_pairs_and_adds_only_new_pairs(self):
        rows = [{"subject": "s1", "pair_id": str(i), "model_a": a, "model_b": b, "value": value}
                for i, (a, b, value) in enumerate((("a", "b", 1), ("b", "c", 1), ("a", "c", 0)))]
        fit = fit_bradley_terry(rows[:2])
        result = augmentation_analysis(rows[:2], rows[2:], fit, ["a", "b", "c"], repeats=3)
        first, last = result["experiments"][0], result["experiments"][-1]
        self.assertEqual(first["summary"]["probability_mae"]["mean"], 0)
        self.assertEqual(first["coverage"]["n_pairs"]["mean"], 2)
        self.assertEqual(last["coverage"]["n_pairs"]["mean"], 3)
        self.assertEqual(last["coverage"]["n_added_unreviewed_pairs"]["mean"], 1)
        self.assertGreater(last["summary"]["probability_mae"]["mean"], 0)

    def test_legacy_conversion_uses_development_and_original_plan_orientation(self):
        self.manifest["splits"]["development"] = self.manifest["splits"].pop("validation")
        self.manifest["pair_catalog"]["development"] = self.manifest["pair_catalog"].pop("validation")
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        record = {**self.call, "judge_id": "judge_3", "judge_version": "judge_3_old",
                  "instrument": "gre_quant", "plan_ids": ["z", "a"], "calls": [{}] * 5}
        old = self.study / "old.jsonl"
        old.write_text(json.dumps(record) + "\n")
        result = analyze_legacy(self.study, old, bootstrap=0, ssot=self.ssot)
        self.assertEqual(result["split"], "development")
        self.assertEqual(result["judges"][0]["metrics"]["probability_mae"], 0)
        self.assertEqual(result["judges"][0]["n_calls"], 5)


if __name__ == "__main__":
    unittest.main()
