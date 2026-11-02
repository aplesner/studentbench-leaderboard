"""Numerical and sampling invariants for model-level leaderboard evaluation."""

import unittest

from studentbench_leaderboard.ranking import (
    aggregate_pairs, bootstrap_alignment, compare_rankings, fit_bradley_terry,
)


def comparison(a="a", b="b", value=1.0, subject="s1", pair_id="p1", **kwargs):
    return {"subject": subject, "pair_id": pair_id, "model_a": a,
            "model_b": b, "value": value, **kwargs}


def synthetic_fit(scores):
    return {"connected": True, "converged": True,
            "leaderboard": [{"model": model, "score": score} for model, score in scores.items()]}


class AggregationTests(unittest.TestCase):
    def test_repeated_reviews_do_not_overweight_a_criterion(self):
        rows = [comparison(criterion="q1", value=1) for _ in range(12)]
        rows += [comparison(criterion="q2", value=0)]
        result = aggregate_pairs(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], .5)
        self.assertEqual(result[0]["weight"], 1)
        self.assertEqual(result[0]["n_criteria"], 2)

    def test_family_weighting_preserves_exact_variants_before_averaging(self):
        rows = [comparison(criterion="q1_v1|scale1", criterion_family="q1", value=1),
                comparison(criterion="q1_v2|scale2", criterion_family="q1", value=0),
                comparison(criterion="q2_v1|scale1", criterion_family="q2", value=1)]
        result = aggregate_pairs(rows)[0]
        self.assertEqual(result["value"], .75)
        self.assertEqual(result["n_criteria"], 2)
        self.assertEqual(result["n_exact_criteria"], 3)

    def test_reversed_slots_and_abstentions(self):
        rows = [comparison(value=1), comparison(a="b", b="a", value=0), comparison(value=None)]
        self.assertEqual(aggregate_pairs(rows)[0]["value"], 1)
        self.assertEqual(aggregate_pairs(rows)[0]["n_criterion_ratings"], 2)

    def test_inconsistent_pair_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent model"):
            aggregate_pairs([comparison(), comparison(a="c")])


class BradleyTerryTests(unittest.TestCase):
    def test_complete_separation_is_finite_and_slot_invariant(self):
        rows = [comparison(pair_id=f"p{i}", subject=f"s{i}") for i in range(10)]
        fit = fit_bradley_terry(rows)
        swapped = fit_bradley_terry([{**r, "model_a": r["model_b"], "model_b": r["model_a"],
                                     "value": 1 - r["value"]} for r in rows])
        self.assertTrue(fit["converged"])
        self.assertTrue(fit["connected"])
        self.assertEqual(fit["leaderboard"][0]["model"], "a")
        self.assertAlmostEqual(sum(r["score"] for r in fit["leaderboard"]), 0)
        self.assertGreater(fit["pairwise_probabilities"][0]["probability_a"], .8)
        self.assertLess(fit["pairwise_probabilities"][0]["probability_a"], 1)
        self.assertEqual(len(fit["leaderboard"]), len(swapped["leaderboard"]))
        for original, reversed_slots in zip(fit["leaderboard"], swapped["leaderboard"]):
            self.assertEqual({k: v for k, v in original.items() if k != "score"},
                             {k: v for k, v in reversed_slots.items() if k != "score"})
            self.assertAlmostEqual(original["score"], reversed_slots["score"], delta=1e-12)

    def test_ties_have_equal_scores_and_midranks(self):
        fit = fit_bradley_terry([comparison(value=.5)])
        self.assertEqual([r["rank"] for r in fit["leaderboard"]], [1.5, 1.5])
        self.assertEqual(fit["pairwise_probabilities"][0]["probability_a"], .5)

    def test_weighted_observation_equals_repeated_observations(self):
        weighted = fit_bradley_terry([comparison(weight=7)])
        repeated = fit_bradley_terry([comparison(pair_id=f"p{i}") for i in range(7)])
        for a, b in zip(weighted["leaderboard"], repeated["leaderboard"]):
            self.assertAlmostEqual(a["score"], b["score"])

    def test_disconnected_graph_never_fabricates_global_ranks(self):
        fit = fit_bradley_terry([comparison(), comparison(a="c", b="d", pair_id="p2")],
                               models=["a", "b", "c", "d", "e"])
        self.assertFalse(fit["connected"])
        self.assertEqual(fit["components"], [["a", "b"], ["c", "d"], ["e"]])
        self.assertTrue(all(r["rank"] is None for r in fit["leaderboard"]))
        self.assertEqual(fit["models_without_comparisons"], ["e"])
        self.assertEqual(fit["model_coverage"], .8)
        self.assertEqual(len(fit["pairwise_probabilities"]), 2)
        self.assertIsNone(compare_rankings(fit, fit)["spearman"])

    def test_input_validation(self):
        for value in (float("nan"), float("inf"), -1, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_bradley_terry([comparison(value=value)])
        with self.assertRaises(ValueError):
            fit_bradley_terry([comparison()], models=["a"])


class AlignmentTests(unittest.TestCase):
    def test_reverse_order_and_probability_discrepancy(self):
        left = synthetic_fit({"a": 1, "b": 0, "c": -1})
        right = synthetic_fit({"a": -1, "b": 0, "c": 1})
        metrics = compare_rankings(left, right)
        self.assertAlmostEqual(metrics["spearman"], -1)
        self.assertEqual(metrics["kendall_tau_b"], -1)
        self.assertEqual(metrics["pair_order_agreement"], 0)
        self.assertEqual(metrics["top1_overlap"], 0)
        self.assertAlmostEqual(metrics["rank_mae"], 4 / 3)
        self.assertGreater(metrics["probability_mae"], .5)

    def test_tie_metrics_and_no_invented_constant_correlation(self):
        all_ties = synthetic_fit({"a": 0, "b": 0, "c": 0})
        metrics = compare_rankings(all_ties, all_ties)
        self.assertIsNone(metrics["spearman"])
        self.assertIsNone(metrics["kendall_tau_b"])
        self.assertEqual(metrics["pair_order_agreement"], 1)
        self.assertEqual(metrics["top1_overlap"], 1)
        self.assertEqual(metrics["rank_mae"], 0)
        self.assertEqual(metrics["probability_mae"], 0)
        tied_top = synthetic_fit({"a": 1, "b": 1, "c": 0})
        ordered = synthetic_fit({"a": 2, "b": 1, "c": 0})
        self.assertEqual(compare_rankings(tied_top, ordered)["top1_overlap"], .5)
        self.assertAlmostEqual(compare_rankings(tied_top, ordered)["kendall_tau_b"], 2 / (6 ** .5))

    def test_mismatched_model_universe_is_not_scored_as_complete(self):
        metric = compare_rankings(synthetic_fit({"a": 1, "b": 0}),
                                  synthetic_fit({"a": 1, "b": 0, "c": -1}))
        self.assertIsNone(metric["spearman"])
        self.assertEqual(metric["missing_reference_models"], ["c"])
        self.assertAlmostEqual(metric["reference_model_coverage"], 2 / 3)

    def test_paired_cluster_bootstrap_preserves_identical_judges(self):
        rows = []
        for subject in range(5):
            for a, b, value in (("a", "b", .8), ("b", "c", .6), ("a", "c", .9)):
                rows.append(comparison(a=a, b=b, value=value if subject < 4 else 1 - value,
                                       subject=f"s{subject}", pair_id=a + b))
        result = bootstrap_alignment(rows, rows, n_bootstrap=30, seed=3)
        self.assertEqual(result, bootstrap_alignment(rows, rows, n_bootstrap=30, seed=3))
        self.assertEqual(result["valid_bootstrap_fits"]["paired"], 30)
        for name in ("rank_mae", "probability_mae"):
            self.assertEqual(result["metric_intervals"][name]["low"], 0)
            self.assertEqual(result["metric_intervals"][name]["high"], 0)

    def test_sparse_resampling_reports_disconnected_draws(self):
        rows = [comparison(subject="s1"),
                comparison(a="b", b="c", subject="s2", pair_id="p2")]
        result = bootstrap_alignment(rows, rows, n_bootstrap=100, seed=5)
        self.assertTrue(result["human"]["connected"])
        valid = result["valid_bootstrap_fits"]["paired"]
        self.assertGreater(valid, 20)
        self.assertLess(valid, 80)
        self.assertEqual(result["rank_intervals"]["human"]["a"]["n_valid"], valid)


if __name__ == "__main__":
    unittest.main()
