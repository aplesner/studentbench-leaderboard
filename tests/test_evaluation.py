import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from studentbench_leaderboard.evaluation import run_evaluation


class Client:
    def __init__(self, route='direct'):
        self.route = route
        self.calls = 0
    def configuration(self, model, reasoning_effort=None):
        return {'provider': self.route, 'model': model, 'reasoning_effort': reasoning_effort}
    def complete(self, model, messages, schema, **kwargs):
        self.calls += 1
        payload = json.loads(messages[1]['content'])
        assert set(payload['plans']) == {'A', 'B'}
        return json.dumps({key: {'score': 1, 'reason': 'Evidence supports this.'} for key in schema['required']}), {}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'system.txt').write_text('Judge the plans')
        (self.path / 'user.txt').write_text('$payload')
        (self.path / 'judge_1.toml').write_text('''schema_version = 1
judge_id = "judge_1"
strategy = "single"
[prompts]
system = "system.txt"
user = "user.txt"
[[members]]
name = "one"
model = "openai/test"
''')
        def plan(model, subject):
            return {'ai_model_preset_id': model, 'review_subject_ids': [subject], 'plan_as_stored': {'text': model}}
        self.generated = [{'plan_id': 'new', 'lesson_plan_id': 'new', **plan('candidate', 's1')}]
        self.benchmark = SimpleNamespace(instrument='quant', fingerprint='fp', judge_id='judge_1',
            judge_dir=self.path, plans={'a': plan('old-a', 's1'), 'b': plan('old-b', 's1')},
            contexts={'s1': {'pretest': [{'question': 'Q'}], 'rubric': [{'id': 'q', 'scale_id': 's'}]}},
            baseline_records=[])
        self.client = Client()
        self.loader = patch('studentbench_leaderboard.evaluation.load_generation', return_value=self.generated)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.coverage = patch('studentbench_leaderboard.evaluation.generation_coverage', return_value={
            'selected_subjects': ['s1'], 'completed_subjects': ['s1'], 'pending_subjects': []})
        self.mock_coverage = self.coverage.start()
        self.addCleanup(self.coverage.stop)
    def run_eval(self, **kwargs):
        return run_evaluation(self.benchmark, self.path, self.client, output=self.path / 'out', bootstrap=5, **kwargs)
    def test_budget_resume_and_connected_ranking(self):
        dry = self.run_eval(dry_run=True)
        self.assertEqual(dry['coverage']['scheduled_calls'], 2)
        self.assertEqual(self.client.calls, 0)
        partial = self.run_eval(max_new_calls=1)
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['coverage']['missing_comparisons'], 1)
        result = self.run_eval()
        self.assertTrue(result['complete'])
        self.assertTrue(result['ranking']['connected'])
        self.assertIn('candidate', result['rank_intervals'])
        self.assertEqual(self.client.calls, 2)
        self.run_eval()
        self.assertEqual(self.client.calls, 2)
        self.assertTrue((self.path / 'out' / 'report.html').exists())
        self.assertIsNone(result['human_reference']['candidate_human_scores'])
    def test_changed_route_and_tampered_cache_rejected(self):
        self.run_eval()
        self.client.route = 'other'
        with self.assertRaisesRegex(ValueError, 'schedule'):
            self.run_eval()
        self.client.route = 'direct'
        file = next((self.path / 'out' / 'calls').glob('*.json'))
        record = json.loads(file.read_text())
        record['ratings']['q']['score'] = 0
        file.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Cached call result mismatch'):
            self.run_eval()
    def test_incompatible_baseline_is_scheduled_and_budgeted(self):
        self.benchmark.baseline_records = [{'pair': {'subject': 's1', 'plan_ids': ['a', 'b']},
                                            'request_id': 'unverified'}]
        dry = self.run_eval(dry_run=True)
        self.assertEqual(dry['coverage']['scheduled_calls'], 3)
        self.assertEqual(len(dry['baseline_import']['rejected']), 1)
        result = self.run_eval(max_new_calls=2)
        self.assertEqual(result['coverage']['missing_calls'], 1)
        self.assertFalse(result['complete'])

    def test_real_verified_study_call_reconstruction(self):
        import hashlib
        from studentbench_leaderboard.data import Dataset
        from studentbench_leaderboard.pretest_context import PretestContextBank
        from studentbench_leaderboard.comparison import ConfiguredJudge
        from studentbench_leaderboard.judge_config import load_definition
        from studentbench_leaderboard.evaluation import _baseline
        study = Path('results/ranking-study-20260910')
        paths = sorted((study / 'calls').glob('enhanced_terra_high--*.json'))
        if not paths:
            self.skipTest('Historical study fixture is not present')
        path = paths[0]
        record = json.loads(path.read_text())
        config = study / 'configs/enhanced_terra_high'
        data = Dataset(Path('studentbench_data_ssot'), 'gre_quant')
        bank = PretestContextBank.load(data.root, source_bank=config / 'question_bank.json',
            expected_source_sha256='70bc1f1d4d85b93d6f6b23f035a49ac912d4aa5f29787a9510f68072d0a7f336')
        subject = record['subject']
        record['pair'] = {'subject': subject, 'plan_ids': record['plan_ids']}
        record['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        benchmark = SimpleNamespace(fingerprint='bundle', source_dataset_sha256=data.fingerprint,
            instrument=data.instrument, plans=data.plans,
            contexts={subject: {'pretest': bank.enrich_pretest(data.pretests[subject]),
                                'rubric': data.rubric(record['grading_id'])}})
        definition = load_definition('judge_5', config)
        route = {'provider': 'openai', 'transport': 'direct', 'effective_model': 'openai/gpt-5.6-terra',
                 'api_base_sha256': hashlib.sha256(b'https://api.openai.com/v1').hexdigest()}
        rows = _baseline(record, benchmark, data.plans, ConfiguredJudge(definition), definition, [route])
        self.assertEqual(len(rows), 8)
        route['transport'] = 'openrouter'
        with self.assertRaisesRegex(ValueError, 'route provenance'):
            _baseline(record, benchmark, data.plans, ConfiguredJudge(definition), definition, [route])

    def test_incomplete_generation_does_not_freeze_schedule(self):
        self.mock_coverage.return_value = {'selected_subjects': ['s1', 's2'],
                                           'completed_subjects': ['s1'], 'pending_subjects': ['s2']}
        result = self.run_eval()
        self.assertFalse(result['complete'])
        self.assertEqual(result['generation_coverage']['pending_subjects'], ['s2'])
        self.assertEqual(self.client.calls, 0)
        self.assertFalse((self.path / 'out' / 'schedule.json').exists())
        self.mock_coverage.return_value = {'selected_subjects': ['s1'], 'completed_subjects': ['s1'], 'pending_subjects': []}
        self.assertTrue(self.run_eval()['complete'])

    def test_rejects_symlink_and_generation_artifact_outputs(self):
        output = self.path / 'out'
        output.mkdir()
        protected = self.path / 'protected.json'
        protected.write_text('protected')
        (output / 'report.html').symlink_to(protected)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.run_eval()
        self.assertEqual(protected.read_text(), 'protected')
        with self.assertRaisesRegex(ValueError, 'generation artifacts'):
            run_evaluation(self.benchmark, self.path, self.client, output=self.path / 'generations', bootstrap=0)

    def test_transient_retry_respects_attempt_budget(self):
        class Temporary(Exception):
            transient = True
        attempts = []
        def fail(*args, **kwargs):
            attempts.append(1)
            raise Temporary('temporary')
        self.client.complete = fail
        result = self.run_eval(max_new_calls=3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(result['coverage']['new_calls_attempted'], 3)
        self.assertFalse(result['complete'])

    def test_shared_baseline_cache_reuses_across_candidates(self):
        self.benchmark.baseline_records = [{'pair': {'subject': 's1', 'plan_ids': ['a', 'b']},
                                            'request_id': 'unverified'}]
        cache = self.path / 'shared-cache'
        result = self.run_eval(cache_dir=cache)
        self.assertTrue(result['complete'])
        self.assertEqual(self.client.calls, 3)
        self.generated[0] = {**self.generated[0], 'plan_id': 'new-2', 'lesson_plan_id': 'new-2',
                             'ai_model_preset_id': 'candidate-2', 'plan_as_stored': {'text': 'candidate-2'}}
        out2 = self.path / 'out-2'
        preview = run_evaluation(self.benchmark, self.path, self.client, output=out2, cache_dir=cache,
                                 bootstrap=0, dry_run=True)
        self.assertEqual(preview['coverage']['shared_cache_reused'], 1)
        self.assertEqual(preview['coverage']['new_calls_planned'], 2)
        self.assertFalse(out2.exists())
        second = run_evaluation(self.benchmark, self.path, self.client, output=out2, cache_dir=cache, bootstrap=0)
        self.assertTrue(second['complete'])
        self.assertEqual(second['coverage']['shared_cache_reused'], 1)
        self.assertEqual(self.client.calls, 5)

    def test_shared_cache_refuses_tampering_and_symlinks(self):
        cache = self.path / 'shared-cache'
        self.run_eval(cache_dir=cache)
        entry = next(cache.glob('*.json'))
        record = json.loads(entry.read_text())
        record['ratings']['q']['score'] = 0
        entry.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Cached call result mismatch'):
            run_evaluation(self.benchmark, self.path, self.client, output=self.path / 'out-2',
                           cache_dir=cache, bootstrap=0, dry_run=True)
        entry.unlink()
        (cache / 'bad').symlink_to(self.path / 'out')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            run_evaluation(self.benchmark, self.path, self.client, output=self.path / 'out-2',
                           cache_dir=cache, bootstrap=0)

    def test_failed_call_is_missing(self):
        self.client.complete = lambda *a, **k: (_ for _ in ()).throw(ValueError('bad output'))
        result = self.run_eval()
        self.assertFalse(result['complete'])
        self.assertEqual(result['ranking']['n_comparisons'], 0)
        self.assertEqual(result['coverage']['missing_calls'], 2)

if __name__ == '__main__':
    unittest.main()
