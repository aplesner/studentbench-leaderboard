import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from studentbench_leaderboard.evaluation import run_evaluation, run_multi_evaluation
from tests.test_evaluation import EvaluationTests


class MultiEvaluationTests(TestCase):
    def setUp(self):
        fixture = EvaluationTests('test_budget_resume_and_connected_ranking')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.path, self.benchmark, self.client = fixture.path, fixture.benchmark, fixture.client
        self.dirs = [self.path / 'g1', self.path / 'g2']
        self.plans = []
        for index, directory in enumerate(self.dirs):
            directory.mkdir()
            model = f'candidate-{index}'
            (directory / 'generation_manifest.json').write_text(json.dumps({
                'model': model, 'configuration': {'model': model, 'reasoning_effort': 'high'},
                'settings': {}, 'selected_subjects': ['s1']}))
            self.plans.append([{'plan_id': f'new-{index}', 'lesson_plan_id': f'new-{index}',
                'ai_model_preset_id': model, 'review_subject_ids': ['s1'],
                'plan_as_stored': {'text': model}}])
        fixture.loader.stop()
        self.loader = patch('studentbench_leaderboard.evaluation.load_generation',
                            side_effect=lambda path, benchmark: self.plans[self.dirs.index(Path(path))])
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def run_multi(self, **kwargs):
        return run_multi_evaluation(self.benchmark, self.dirs, self.client,
            output=self.path / 'combined', bootstrap=0, **kwargs)

    def test_joint_pairs_and_cache_reuse(self):
        cache = self.path / 'cache'
        single = run_evaluation(self.benchmark, self.dirs[0], self.client,
            output=self.path / 'single', cache_dir=cache, bootstrap=0)
        self.assertTrue(single['complete'])
        preview = self.run_multi(dry_run=True, cache_dir=cache)
        self.assertEqual(preview['planned_pairs'], {'new_baseline': 4, 'new_new': 1,
            'baseline_rejudged': 0, 'baseline_imported': 0, 'total': 5})
        self.assertEqual(preview['coverage']['shared_cache_reused'], 2)
        self.assertFalse((self.path / 'combined').exists())
        result = self.run_multi(cache_dir=cache)
        self.assertTrue(result['complete'])
        self.assertEqual(self.client.calls, 5)
        self.assertEqual(len(result['ranking']['leaderboard']), 4)
        self.assertIsNone(result['human_reference']['candidate_human_scores'])

    def test_incomplete_and_invalid_cohorts_do_not_freeze(self):
        self.fixture.mock_coverage.side_effect = [
            {'selected_subjects': ['s1'], 'completed_subjects': ['s1'], 'pending_subjects': []},
            {'selected_subjects': ['s1'], 'completed_subjects': [], 'pending_subjects': ['s1']}]
        result = self.run_multi()
        self.assertEqual(result['status'], 'generation_incomplete')
        self.assertFalse((self.path / 'combined' / 'schedule.json').exists())
        self.fixture.mock_coverage.side_effect = None
        manifest = self.dirs[1] / 'generation_manifest.json'
        value = json.loads(manifest.read_text())
        value['selected_subjects'] = ['s2']
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'subject coverage mismatch'):
            self.run_multi()
        value['selected_subjects'] = ['s1']
        value['model'] = 'candidate-0'
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate'):
            self.run_multi()

    def test_missing_manifest_is_incomplete(self):
        (self.dirs[1] / 'generation_manifest.json').unlink()
        result = self.run_multi()
        self.assertEqual(result['status'], 'generation_incomplete')
        self.assertFalse(result['generation_coverage'][1]['manifest_present'])
        self.assertFalse((self.path / 'combined').exists())
