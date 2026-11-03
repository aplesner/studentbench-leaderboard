import importlib.util
import json
import os
import tempfile
from contextlib import ExitStack, redirect_stdout
import io
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from decimal import Decimal
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'experiments/public-release/extend.py'
spec = importlib.util.spec_from_file_location('extension_driver', MODULE)
extend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extend)


class ExtensionTests(unittest.TestCase):
    def test_roster_active_and_unique(self):
        config = {'models':[{'id':'a', 'model':'provider/a', 'price_id':'provider/a'},
                            {'id':'b', 'model':'provider/b', 'price_id':'provider/b', 'enabled':False}]}
        self.assertEqual([m['id'] for m in extend.validate_roster(config, ['provider/old'])], ['a'])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            extend.validate_roster(config, ['provider/a'])
        with self.assertRaisesRegex(ValueError, 'directory'):
            extend.validate_roster({'models':[{'id':'../x', 'model':'p/x', 'price_id':'p/x'}]}, [])
        with self.assertRaises(ValueError):
            extend.validate_roster({**config, 'credential':'no'}, [])

    def test_pricing_explicit_and_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'catalog.json'
            path.write_text(json.dumps({'data':[{'id':'catalog/a', 'pricing':{'prompt':'0.001', 'completion':'0.002'}},
                                                {'id':'judge', 'pricing':{'prompt':'0.001', 'completion':'0.002'}}]}))
            entries = [{'model':'proxy/a','price_id':'catalog/a'}]
            prices = extend.price_map(path, entries, 'judge')
            request = {'configuration':{'requested_model':'proxy/a','settings':{'max_tokens':10}},
                       'messages':[], 'schema':{}}
            self.assertGreater(extend.estimate_requests([request], prices), Decimal('0.02'))
            with self.assertRaisesRegex(ValueError, 'Missing frozen catalog price'):
                extend.price_map(path, [{'model':'x','price_id':'unknown'}], 'judge')

    def test_main_checks_estimated_cost_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); v1 = root/'v1'; v1.mkdir()
            (v1/'evaluation').mkdir(); (v1/'judge-cache').mkdir()
            catalog = root/'catalog.json'
            catalog.write_text(json.dumps({'data':[{'id':'new/model', 'pricing':{'prompt':'1','completion':'1'}},
                {'id':'judge/model', 'pricing':{'prompt':'1','completion':'1'}}]}))
            protocol = {'models':[{'id':'old','model':'old/model'}], 'benchmark':'bundle',
                        'generation_settings':{'max_tokens':20}, 'generation_timeout':180, 'judge_timeout':420}
            (v1/'protocol.json').write_text(json.dumps(protocol))
            route = {'profile':'tailscale','transport':'litellm_proxy','requested_model':'judge/model',
                     'reasoning_effort':'high','settings':{'max_tokens':8192}}
            request = {'timeout':420}
            call_id = extend.digest(request)
            (v1/'judge-cache'/f'{call_id}.json').write_text(json.dumps({'request_id':call_id,'request':request}))
            schedule = {'generation_models':['old/model'], 'include_new_pairs':True,
                        'comparisons_per_student':None,'benchmark':'benchmark-id','judge':{'id':'judge'},
                        'routes':[route], 'requests':[call_id], 'seed':1}
            schedule['identity'] = extend.digest(schedule)
            (v1/'evaluation'/'schedule.json').write_text(json.dumps(schedule))
            (v1/'evaluation'/'evaluation.json').write_text(json.dumps({'complete':True, 'schedule_identity':schedule['identity']}))
            old_dir = v1/'generations'/'old'
            old_dir.mkdir(parents=True)
            (old_dir/'generation_manifest.json').write_text(json.dumps({'model':'old/model'}))
            ledger = v1/'spend-ledger.jsonl'
            ledger.write_text(json.dumps({'event':'seed','catalog_sha256':extend.sha(catalog),'attempts':[]})+'\n')
            config = root/'config.json'
            config.write_text(json.dumps({'models':[{'id':'new', 'model':'new/model', 'price_id':'new/model'}]}))
            class FakeClient:
                profile = 'tailscale'
                profile_config = {'api_base_env':'LITELLM_BASE_URL', 'api_key_env':'LITELLM_API_KEY'}
                def configuration(self, model, reasoning_effort=None, **settings):
                    return {'profile':'tailscale','transport':'litellm_proxy','requested_model':model,
                            'reasoning_effort':reasoning_effort,'settings':settings}
            definition = SimpleNamespace(members=[SimpleNamespace(model='judge/model', reasoning_effort='high')])
            benchmark = SimpleNamespace(path=root/'bundle', fingerprint='benchmark-id', contexts={'s':{}},
                                        plans={}, judge_id='j', judge_dir=root/'judge')
            preview = {'requests':[{'configuration':FakeClient().configuration('new/model', max_tokens=20),
                                     'messages':[], 'schema':{}}]}
            with ExitStack() as stack:
                stack.enter_context(patch.object(extend.release, 'RUN', v1))
                stack.enter_context(patch.object(extend.release, 'ROOT', root))
                stack.enter_context(patch.object(extend.release, 'CATALOG', catalog))
                stack.enter_context(patch.object(extend.Benchmark, 'load', return_value=benchmark))
                stack.enter_context(patch.object(extend, 'load_definition', return_value=definition))
                stack.enter_context(patch.object(extend, 'ConfiguredJudge', return_value=SimpleNamespace(configuration=lambda:{'id':'judge'})))
                stack.enter_context(patch.object(extend, 'generation_coverage', return_value={'selected_subjects':['s'],'pending_subjects':[]}))
                stack.enter_context(patch.object(extend, 'LLMClient', return_value=FakeClient()))
                generate = stack.enter_context(patch.object(extend, 'run_generation', return_value=preview))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaises(extend.release.BudgetExceeded):
                    extend.main(['generate','--config',str(config),'--catalog',str(catalog),
                                 '--extension-dir',str(root/'extension'),'--execute','--max-new-calls','1'])
                extension_ledger = root/'extension'/'spend-ledger.jsonl'
                self.assertEqual(json.loads(extension_ledger.read_text())['attempts'], [])
                extension_ledger.unlink()
                with self.assertRaisesRegex(ValueError, 'ledger missing'):
                    extend.main(['summary','--config',str(config),'--catalog',str(catalog),
                                 '--extension-dir',str(root/'extension')])
                self.assertFalse(extension_ledger.exists())
                protected = root/'protected'
                protected.mkdir()
                with patch.dict(os.environ, {'STUDENTBENCH_SSOT':str(protected)}), \
                     self.assertRaisesRegex(ValueError, 'overlap benchmark or SSOT'):
                    extend.main(['summary','--config',str(config),'--catalog',str(catalog),
                                 '--extension-dir',str(protected/'bad')])
                self.assertFalse((protected/'bad').exists())
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(generate.call_args.kwargs['dry_run'])
            self.assertEqual(len(ledger.read_text().splitlines()), 1)

    def test_cache_is_copied_once_and_source_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root/'source', root/'target'
            source.mkdir()
            (source/'one.json').write_text('original')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                extend.prepare_cache(source, target, ['one', 'missing'], {'one':extend.sha(source/'one.json'), 'missing':'missing'})
            self.assertFalse(target.exists())
            inventory = {'one':extend.sha(source/'one.json')}
            extend.prepare_cache(source, target, ['one'], inventory)
            (target/'one.json').write_text('extension')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                extend.prepare_cache(source, target, ['one'], inventory)
            self.assertEqual((source/'one.json').read_text(), 'original')
            (target/'one.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                extend.prepare_cache(source, target, ['one'], inventory)

if __name__ == '__main__': unittest.main()
