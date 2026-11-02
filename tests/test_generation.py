import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from studentbench_leaderboard.generation import run_generation, load_generation
from test_benchmark import make_benchmark

PLAN={'summary':'Study','concepts':[{'concept_id':'c1','name':'Algebra','why_it_matters':'Practice','pretest_evidence':[], 'priority_rank':1,'allotted_minutes':60,'practice_problems':[{'stem':'1+1?', 'format':'MCQ','options':{'A':'2','B':'3','C':'','D':''},'answer':'A','solution':'1+1=2','difficulty':'Easy'}]}]}
class FakeClient:
    def __init__(self): self.calls=[]
    def configuration(self,model,reasoning_effort=None,**settings): return {'model':model,'effort':reasoning_effort,'settings':settings}
    def complete(self,model,messages,schema,**kwargs):
        self.calls.append(messages)
        return json.dumps(PLAN), {'usage':{}}

class GenerationTests(unittest.TestCase):
    def test_resume_and_input_privacy(self):
        with TemporaryDirectory() as temp:
            tmp_path=Path(temp); b=make_benchmark(tmp_path/'bundle'); c=FakeClient(); out=tmp_path/'run'
            r=run_generation(b,out,'model',c,max_new_calls=1)
            self.assertEqual((r['completed'],r['pending']),(1,1))
            self.assertNotIn('secret',json.dumps(c.calls))
            self.assertEqual(run_generation(b,out,'model',c)['completed'],1)
            self.assertEqual(run_generation(b,out,'model',c)['reused'],2)
            self.assertEqual(len(c.calls),2); self.assertEqual(len(load_generation(out,b)),2)
            self.assertEqual(len(list((out/'generation_attempts').glob('*.json'))),2)
            with self.assertRaisesRegex(ValueError,'different request'): run_generation(b,out,'other',c)

    def test_invalid_response_is_failure_and_dry_run_is_offline(self):
        with TemporaryDirectory() as temp:
            tmp_path=Path(temp); b=make_benchmark(tmp_path/'bundle'); c=FakeClient(); out=tmp_path/'run'
            self.assertEqual(run_generation(b,out,'model',c,dry_run=True)['new_calls_planned'],2)
            self.assertFalse(c.calls); self.assertFalse(out.exists())
            c.complete=lambda *args,**kwargs: ('{}',{})
            self.assertEqual(run_generation(b,out,'model',c,limit=1)['failed'],1)
            self.assertFalse(load_generation(out,b))

    def test_attempt_budget_and_saved_request_validation(self):
        with TemporaryDirectory() as temp:
            tmp_path=Path(temp); b=make_benchmark(tmp_path/'bundle'); c=FakeClient(); out=tmp_path/'run'
            class RateLimitError(Exception): pass
            calls=[]
            def fail(*args, **kwargs):
                calls.append(1)
                raise RateLimitError('temporary')
            c.complete=fail
            result=run_generation(b,out,'model',c,max_new_calls=1)
            self.assertEqual(len(calls),1)
            self.assertEqual(result['calls_started'],1)
            self.assertFalse(result['complete'])
            c=FakeClient(); run_generation(b,out,'model',c)
            files=sorted((out/'generations').glob('*.json'))
            data=json.loads(files[0].read_text())
            data['request']['model']='other'
            files[0].write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_generation(out,b)

    def test_rejects_output_symlinks_and_freezes_selection(self):
        from studentbench_leaderboard.generation import generation_coverage
        with TemporaryDirectory() as temp:
            root=Path(temp); b=make_benchmark(root/'bundle'); out=root/'run'; out.mkdir()
            (b.path/'cache').mkdir()
            (out/'generations').symlink_to(b.path/'cache', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Symlink'): run_generation(b,out,'model',FakeClient())
            self.assertEqual(list((b.path/'cache').iterdir()),[])
        with TemporaryDirectory() as temp:
            root=Path(temp); b=make_benchmark(root/'bundle'); out=root/'run'; c=FakeClient()
            run_generation(b,out,'model',c,limit=1,max_new_calls=0)
            coverage=generation_coverage(out,b)
            self.assertEqual(coverage['selected_subjects'],['s1'])
            self.assertEqual(coverage['pending_subjects'],['s1'])
            with self.assertRaisesRegex(ValueError, 'different request'): run_generation(b,out,'model',c,limit=2)

    def test_invalid_responses_and_usage_survive_repeated_invocations(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); b=make_benchmark(root/'bundle'); out=root/'run'; c=FakeClient()
            raw='{"unparseable":"response"}'
            c.complete=lambda *args,**kwargs: (raw, {'usage':{'completion_tokens':123}})
            for _ in range(2):
                result=run_generation(b,out,'model',c,limit=1)
                self.assertEqual(result['failed'],1)
                self.assertFalse(result['complete'])
                self.assertEqual(load_generation(out,b),[])
            attempts=sorted((out/'generation_attempts').glob('*.json'))
            errors=sorted((out/'generation_errors').glob('*.json'))
            self.assertEqual(len(attempts),2); self.assertEqual(len(errors),2)
            for attempt in attempts:
                data=json.loads(attempt.read_text())
                self.assertEqual(data['raw_response'],raw)
                self.assertEqual(data['metadata']['usage']['completion_tokens'],123)
                self.assertTrue(data['request_id']); self.assertTrue(data['request']['messages'])
                self.assertTrue(data['received_at']); self.assertTrue(data['invocation_id'])
            self.assertNotEqual(json.loads(attempts[0].read_text())['invocation_id'], json.loads(attempts[1].read_text())['invocation_id'])
            for error in errors:
                self.assertTrue((out/json.loads(error.read_text())['attempt_file']).exists())

    def test_transient_error_at_budget_counts_as_failure(self):
        from unittest.mock import patch
        from studentbench_leaderboard.llm_client import LLMRequestError
        with TemporaryDirectory() as temp:
            root=Path(temp); b=make_benchmark(root/'bundle'); out=root/'run'; c=FakeClient()
            for i in range(81): b.contexts[f'extra{i:02d}']=b.contexts['s1']
            def fail(*args,**kwargs):
                raise LLMRequestError('timeout',original_type='APITimeoutError',status_code=None,transient=True)
            c.complete=fail
            with patch('studentbench_leaderboard.generation.time.sleep'):
                result=run_generation(b,out,'model',c,max_new_calls=1)
            self.assertEqual(result['failed'],1)
            self.assertEqual(result['pending'],83)
            self.assertEqual(result['calls_started'],1)
            self.assertEqual(len(list((out/'generation_errors').glob('*.json'))),1)
            self.assertFalse(result['complete'])

    def test_provider_response_without_text_preserves_usage(self):
        from studentbench_leaderboard.llm_client import LLMResponseError
        with TemporaryDirectory() as temp:
            root=Path(temp); b=make_benchmark(root/'bundle'); out=root/'run'; c=FakeClient()
            def fail(*args,**kwargs):
                raise LLMResponseError('no text', response_metadata={'usage':{'completion_tokens':99}, 'finish_reason':'length'}, raw_response=None)
            c.complete=fail
            result=run_generation(b,out,'model',c,limit=1)
            self.assertEqual(result['failed'],1)
            self.assertFalse(result['complete'])
            self.assertEqual(load_generation(out,b),[])
            attempts=list((out/'generation_attempts').glob('*.json'))
            self.assertEqual(len(attempts),1)
            record=json.loads(attempts[0].read_text())
            self.assertIsNone(record['raw_response'])
            self.assertEqual(record['metadata']['usage']['completion_tokens'],99)
            self.assertEqual(record['metadata']['finish_reason'],'length')
            error=json.loads(next((out/'generation_errors').glob('*.json')).read_text())
            self.assertEqual(out/error['attempt_file'],attempts[0])
