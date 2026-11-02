import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'experiments/public-release/run.py'
spec = importlib.util.spec_from_file_location('release_runner', MODULE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class FakeClient:
    profile = 'tailscale'
    def __init__(self, usage=None, fail=False):
        self.usage, self.fail, self.calls = usage, fail, 0
    def configuration(self, model, reasoning_effort=None, **settings):
        return {'profile':'tailscale', 'transport':'litellm_proxy', 'settings':settings}
    def complete(self, model, messages, schema, **kwargs):
        self.calls += 1
        if self.fail: raise RuntimeError('unknown outcome')
        return '{}', {'usage': self.usage, 'routing': self.configuration(model, **{k:v for k,v in kwargs.items() if k not in ('timeout', 'reasoning_effort')})}


class ReleaseRunnerTests(unittest.TestCase):
    def test_cumulative_cap_and_unused_reservation_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = release.Ledger(Path(tmp)/'spend.jsonl', cap=Decimal('1'))
            fake = FakeClient({'prompt_tokens':1, 'completion_tokens':1})
            client = release.BudgetedClient(fake, ledger, {'m':(Decimal('0'), Decimal('0.01'))})
            for _ in range(2): client.complete('m', [], {}, max_tokens=50)
            self.assertEqual(ledger.summary()['estimated_committed_usd'], '0.02')
            self.assertEqual(ledger.summary()['attempts'], 2)
            self.assertEqual(client.configuration('m')['settings']['max_tokens'], 8192)
            self.assertEqual(client.configuration('m', max_tokens=50)['settings']['max_tokens'], 50)
            with self.assertRaises(release.BudgetExceeded):
                client.complete('m', [], {}, max_tokens=99)
            self.assertEqual(fake.calls, 2)

    def test_failed_unknown_and_missing_usage_keep_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = release.Ledger(Path(tmp)/'spend.jsonl', cap=1)
            fake = FakeClient(fail=True)
            client = release.BudgetedClient(fake, ledger, {'m':(Decimal('0'), Decimal('0.01'))})
            for _ in range(2):
                with self.assertRaises(RuntimeError): client.complete('m', [], {}, max_tokens=40)
            self.assertEqual(ledger.summary()['estimated_committed_usd'], '0.80')
            self.assertEqual(ledger.summary()['attempts'], 2)
            with self.assertRaises(release.BudgetExceeded): client.complete('m', [], {}, max_tokens=40)
            self.assertEqual(fake.calls, 2)

    def test_concurrent_ledgers_cannot_overbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spend.jsonl'
            def reserve(_):
                try: release.Ledger(path, cap=1).reserve('m', Decimal('0.3')); return True
                except release.BudgetExceeded: return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                accepted = list(pool.map(reserve, range(8)))
            self.assertEqual(sum(accepted), 3)
            self.assertEqual(release.Ledger(path).summary()['estimated_committed_usd'], '0.9')

    def test_success_without_usage_keeps_reservation_and_price_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'spend.jsonl'
            ledger = release.Ledger(path, cap=1)
            client = release.BudgetedClient(FakeClient(None), ledger,
                {'m': (Decimal('0'), Decimal('0.01'))}, catalog_sha256='catalog-hash')
            client.complete('m', [], {}, max_tokens=40)
            self.assertEqual(ledger.summary()['estimated_committed_usd'], '0.40')
            event = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(event['pricing_revision'], release.PRICING_REVISION)
            self.assertEqual(event['catalog_sha256'], 'catalog-hash')
            self.assertEqual(ledger.summary()['unfinished'], 1)

    def test_catalog_uses_highest_cache_write_price_across_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'catalog.json'
            path.write_text(json.dumps({'data':[{'id':'m', 'pricing':{
                'prompt':'0.1', 'completion':'0.2', 'input_cache_write':'0.3',
                'overrides':[{'prompt':'0.2', 'completion':'0.4', 'input_cache_write_1h':'0.5'}]}}]}))
            self.assertEqual(release.prices_from_catalog(path)['m'], (Decimal('0.5'), Decimal('0.4')))

    def test_subprocess_reservations_and_resume_share_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'spend.jsonl'
            code = """import importlib.util, sys
from decimal import Decimal
s=importlib.util.spec_from_file_location('release', sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
try: m.Ledger(sys.argv[2], cap=1).reserve('m', Decimal('0.3'))
except m.BudgetExceeded: sys.exit(2)
"""
            procs = [subprocess.Popen([sys.executable, '-c', code, str(MODULE), str(path)]) for _ in range(6)]
            statuses = [proc.wait() for proc in procs]
            self.assertEqual(statuses.count(0), 3)
            self.assertEqual(statuses.count(2), 3)
            ledger = release.Ledger(path, cap=1)
            self.assertEqual(ledger.summary()['attempts'], 3)
            with self.assertRaises(release.BudgetExceeded): ledger.reserve('m', Decimal('0.3'))

    def test_returned_error_usage_reconciles_reservation(self):
        from studentbench_leaderboard.llm_client import LLMResponseError
        class FailedResponse(FakeClient):
            def complete(self, *args, **kwargs):
                raise LLMResponseError('no content', raw_response=None,
                    response_metadata={'usage':{'prompt_tokens':1, 'completion_tokens':2}, 'finish_reason':'length'})
        with tempfile.TemporaryDirectory() as tmp:
            ledger = release.Ledger(Path(tmp)/'spend.jsonl', cap=1)
            client = release.BudgetedClient(FailedResponse(), ledger, {'m':(Decimal('0.01'), Decimal('0.01'))})
            # Reservation includes conservative prompt byte allowance.
            ledger.cap = Decimal('100')
            with self.assertRaises(LLMResponseError): client.complete('m', [], {}, max_tokens=40)
            self.assertEqual(ledger.summary()['estimated_committed_usd'], '0.03')
            self.assertEqual(ledger.summary()['unfinished'], 0)

    def test_per_model_generation_settings(self):
        protocol = {'generation_settings':{'max_tokens':32768, 'temperature':0}}
        self.assertEqual(release.generation_settings(protocol, {'generation_settings':{'max_tokens':65536}}),
                         {'max_tokens':65536, 'temperature':0})
        self.assertEqual(release.generation_settings(protocol, {})['max_tokens'], 32768)
        with self.assertRaisesRegex(ValueError, 'Generation settings'):
            release.generation_settings(protocol, {'generation_settings':'invalid'})

    def test_worker_count_bounds(self):
        import argparse
        self.assertEqual(release.worker_count('64'), 64)
        for invalid in ('65', '0', 'bad'):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                release.worker_count(invalid)

    def test_fresh_ledger_initialization_and_unknown_history(self):
        protocol = {'models':[{'id':'a', 'model':'provider/model'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = release.Ledger(root / 'spend.jsonl')
            release.seed_ledger(ledger, None, protocol, {}, 'catalog-hash', run_dir=root)
            release.seed_ledger(ledger, None, protocol, {}, 'catalog-hash', run_dir=root)
            self.assertEqual(ledger.summary()['attempts'], 0)
            self.assertEqual(len(ledger._read()), 1)
            self.assertEqual(ledger._read()[0]['catalog_sha256'], 'catalog-hash')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical = root / 'generations' / 'a' / 'generation_errors'
            historical.mkdir(parents=True)
            (historical / 'failed.json').write_text('{}')
            ledger = release.Ledger(root / 'spend.jsonl')
            with self.assertRaisesRegex(ValueError, 'Historical run artifacts'):
                release.seed_ledger(ledger, None, protocol, {}, 'catalog-hash', run_dir=root)
            self.assertFalse(ledger.path.exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a-smoke.json').write_text('{}')
            ledger = release.Ledger(root / 'spend.jsonl')
            with self.assertRaises(ValueError):
                release.seed_ledger(ledger, None, protocol, {}, 'catalog-hash', run_dir=root)
            self.assertFalse(ledger.path.exists())

    def test_legacy_seed_refuses_extra_artifacts_and_accepts_original_only(self):
        protocol = {'models':[{'id':'a', 'model':'provider/model'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a-smoke.json').write_text(json.dumps({'calls_started':1}))
            folder = root / 'generations' / 'a'
            (folder / 'generation_errors').mkdir(parents=True)
            (folder / 'generation_manifest.json').write_text(json.dumps({
                'model':'provider/model', 'selected_subjects':['s'],
                'configuration':{}, 'settings':{'max_tokens':10}}))
            (folder / 'generation_errors' / 'failure.json').write_text('{}')
            extra = root / 'evaluation-old' / 'calls' / 'paid.json'
            extra.parent.mkdir(parents=True)
            extra.write_text('{}')
            ledger = release.Ledger(root / 'spend.jsonl')
            request = {'messages':[], 'schema':{}}
            with patch.object(release, '_request', return_value=request):
                with self.assertRaisesRegex(ValueError, 'Unaccounted historical artifacts'):
                    release.seed_ledger(ledger, None, protocol, {'provider/model':(Decimal(0), Decimal('0.01'))}, 'hash', run_dir=root)
                self.assertFalse(ledger.path.exists())
                extra.unlink()
                release.seed_ledger(ledger, None, protocol, {'provider/model':(Decimal(0), Decimal('0.01'))}, 'hash', run_dir=root)
            self.assertEqual(ledger.summary()['attempts'], 1)
            self.assertEqual(ledger.summary()['estimated_committed_usd'], '0.10')

    def test_judge_timeout_override_and_legacy_fallback(self):
        self.assertEqual(release.judge_timeout({'generation_timeout':1200, 'judge_timeout':420}), 420)
        self.assertEqual(release.judge_timeout({'generation_timeout':180}), 180)
        with self.assertRaises(ValueError):
            release.judge_timeout({'generation_timeout':180, 'judge_timeout':0})

    def test_main_wires_judge_timeout_and_generation_timeout_separately(self):
        protocol = {'authorized_inference_cap_usd':1000, 'benchmark':'benchmark',
            'models':[{'id':'m', 'model':'provider/model', 'profile':'tailscale'}],
            'generation_timeout':1200, 'judge_timeout':420, 'reasoning_effort':None,
            'generation_settings':{'max_tokens':32768}}
        prices = {key:(Decimal('0'), Decimal('0')) for key in
                  ('provider/model', 'anthropic/claude-haiku-5.5', 'openai/gpt-5.6-terra')}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'protocol.json').write_text(json.dumps(protocol))
            catalog = root/'catalog.json'
            catalog.write_text('{}')
            with patch.object(release, 'RUN', root), patch.object(release, 'CATALOG', catalog), \
                 patch.object(release, 'prices_from_catalog', return_value=prices), \
                 patch.object(release.Benchmark, 'load', return_value=object()), \
                 patch.object(release, 'seed_ledger'), patch.object(release, 'LLMClient', return_value=FakeClient()), \
                 patch.object(release, 'run_multi_evaluation', return_value={'complete':False}) as evaluate, \
                 patch.object(release, 'run_generation', return_value={}) as generate, \
                 redirect_stdout(io.StringIO()):
                release.main(['evaluate', '--max-new-calls', '1'])
                release.main(['generate', '--max-new-calls', '1'])
            self.assertEqual(evaluate.call_args.kwargs['timeout'], 420)
            self.assertEqual(generate.call_args.kwargs['timeout'], 1200)

    def test_active_model_selection_and_joint_evaluation(self):
        all_enabled = {'models':[{'id':str(i)} for i in range(8)]}
        self.assertEqual(len(release.select_models(all_enabled, require_all=True)), 8)
        protocol = {'models':[{'id':str(i), 'enabled':i < 5} for i in range(8)]}
        self.assertEqual([m['id'] for m in release.select_models(protocol)], list('01234'))
        self.assertEqual(len(release.select_models(protocol, list('01234'), require_all=True)), 5)
        with self.assertRaisesRegex(ValueError, 'Deferred'):
            release.select_models(protocol, ['7'])
        with self.assertRaisesRegex(ValueError, 'all 5 active'):
            release.select_models(protocol, ['0'], require_all=True)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            release.select_models(protocol, ['unknown'])

    def test_profile_restricted_and_invocation_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = release.Ledger(Path(tmp)/'spend.jsonl')
            fake = FakeClient({'prompt_tokens':0,'completion_tokens':0})
            client = release.BudgetedClient(fake, ledger, {'m':(Decimal('0'), Decimal('0'))}, max_calls=1)
            client.complete('m', [], {})
            with self.assertRaises(release.BudgetExceeded): client.complete('m', [], {})
            fake.profile = 'auto'
            with self.assertRaises(ValueError): release.BudgetedClient(fake, ledger, {})

if __name__ == '__main__': unittest.main()
