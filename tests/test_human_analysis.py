import unittest
from types import SimpleNamespace

import numpy as np

from studentbench_leaderboard.human_analysis import (
    _nested_samples, direct_human_agreement, human_criterion_rows, reviewer_halves,
)


def vote(pair, reviewer, preference, criterion="q1", subject="s1"):
    return {"subject": subject, "pair_id": pair, "reviewer": reviewer,
            "grading_id": pair + reviewer, "criterion": criterion, "criterion_family": criterion,
            "model_a": "model_a", "model_b": "model_b", "preference": preference,
            "value": ((preference > 0) - (preference < 0) + 1) / 2}


class HumanAnalysisTests(unittest.TestCase):
    def test_human_adapter_canonicalizes_orientation_and_excludes_other_subjects(self):
        session = {"review_subject_id": "kept", "lesson_plan_a_id": "z", "lesson_plan_b_id": "a",
                   "human_reviewer_anon_id": "reviewer"}
        grade = {"preference_score": "2", "question_variant_id": "variant",
                 "rubric_question_id": "family"}
        dataset = SimpleNamespace(sessions={"g1": session, "g2": {**session, "review_subject_id": "heldout"}},
                                  grades_by_session={"g1": [grade], "g2": [grade]},
                                  plans={"a": {"ai_model_preset_id": "left"},
                                         "z": {"ai_model_preset_id": "right"}},
                                  rating_scale=lambda g: "scale")
        rows = human_criterion_rows(dataset, ["kept"])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["model_a"], rows[0]["model_b"]), ("left", "right"))
        self.assertEqual(rows[0]["preference"], -2)
        self.assertEqual(rows[0]["value"], 0)
        self.assertEqual(rows[0]["criterion"], "variant|scale")

    def test_global_reviewer_halves_never_share_a_reviewer_and_match_exact_units(self):
        rows = [vote(pair, reviewer, 1, criterion)
                for pair in ("p1", "p2") for reviewer in ("r1", "r2", "r3", "r4")
                for criterion in ("old|scale1", "new|scale2")]
        rows.append(vote("unique", "r1", -1))
        left, right, meta = reviewer_halves(rows, np.random.default_rng(10))
        self.assertFalse({r["reviewer"] for r in left} & {r["reviewer"] for r in right})
        self.assertEqual({(r["pair_id"], r["criterion"]) for r in left},
                         {(r["pair_id"], r["criterion"]) for r in right})
        self.assertEqual(meta["left"]["pairs"], 2)

    def test_raw_human_agreement_gives_equal_weight_to_units(self):
        # p1: two reviewers completely disagree. p2: three completely agree.
        rows = [vote("p1", "r1", -1), vote("p1", "r2", 1),
                vote("p2", "r1", 1), vote("p2", "r2", 1), vote("p2", "r3", 1)]
        result = direct_human_agreement(rows)
        self.assertEqual(result["direction_agreement"], .5)
        self.assertEqual(result["reviewer_pair_comparisons"], 4)
        self.assertAlmostEqual(np.asarray(result["three_way_confusion_unit_weighted"]).sum(), 2)

    def test_sampling_is_nested_and_retains_complete_review_criterion_sets(self):
        rows = [vote(pair, reviewer, 1, criterion, subject)
                for subject in ("s1", "s2", "s3", "s4") for pair in (subject + "p1", subject + "p2")
                for reviewer in ("r1", "r2", "r3") for criterion in ("q1", "q2")]
        previous = {}
        for axis, density, sampled in _nested_samples(rows, np.random.default_rng(1)):
            identities = {(r["grading_id"], r["criterion"]) for r in sampled}
            self.assertTrue(previous.get(axis, set()) <= identities)
            previous[axis] = identities
            for grading_id in {r["grading_id"] for r in sampled}:
                self.assertEqual({r["criterion"] for r in sampled if r["grading_id"] == grading_id}, {"q1", "q2"})


if __name__ == "__main__":
    unittest.main()
