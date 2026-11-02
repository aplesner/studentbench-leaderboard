import json
import unittest
from studentbench_leaderboard.judge import Task
from studentbench_leaderboard.structured_judge import response_task,parse_judge_response,project_plan

class StructuredJudgeTests(unittest.TestCase):
    def setUp(self):
        self.task=Task('gre_quant','s',['p1','p2'],[{'text':'first'},{'text':'second'}],[],
            [{'id':'q','question':'Which is better?','scale_id':'s','rating_labels':{1:'poor'}}],'hash')
    def test_explicit_winner_conversion_and_no_quality_scale(self):
        task=response_task(self.task,'evidence_winner')
        r={'q':dict(evidence_a='a',evidence_b='b',reason='B better',winner='B',strength='clear')}
        self.assertEqual(parse_judge_response(json.dumps(r),task)['q']['score'],-2)
        self.assertNotIn('rating_labels',task.messages()[1]['content'])
        r['q'].update(winner='tie',strength='none')
        self.assertEqual(parse_judge_response(json.dumps(r),task)['q']['score'],0)
        r['q'].update(winner='insufficient')
        self.assertIsNone(parse_judge_response(json.dumps(r),task)['q']['score'])
    def test_rejects_inconsistent_winner_strength(self):
        r={'q':dict(evidence_a='a',evidence_b='b',reason='equal',winner='tie',strength='clear')}
        with self.assertRaises(ValueError):parse_judge_response(json.dumps(r),response_task(self.task,'evidence_winner'))
    def test_absolute_ratings_difference_and_missing(self):
        task=response_task(self.task,'absolute_ratings')
        r={'q':dict(evidence_a='a',evidence_b='b',reason='B stronger',rating_a=2,rating_b=5)}
        self.assertEqual(parse_judge_response(json.dumps(r),task)['q']['score'],-2)
        r['q']['rating_a']=None
        self.assertIsNone(parse_judge_response(json.dumps(r),task)['q']['score'])
    def test_duplicate_and_missing_fields_rejected(self):
        task=response_task(self.task,'evidence_winner')
        for raw in ['{"q":{},"q":{}}','{}']:
            with self.assertRaises(ValueError):parse_judge_response(raw,task)
    def test_human_display_projection_preserves_presented_key_and_original_source(self):
        plan={'items':[{'answer':'1','answer_original':'24','solution':'24'}]}
        projected=project_plan(plan,'human_display')
        self.assertEqual(projected,{'items':[{'answer':'1','solution':'24'}]})
        self.assertEqual(plan['items'][0]['answer_original'],'24')


class ConfiguredEvidenceIntegrationTests(unittest.TestCase):
    def test_named_judge_preserves_raw_evidence_and_maps_b_winner(self):
        from unittest.mock import patch
        from studentbench_leaderboard.comparison import compare_plans,ComparisonContext
        raw=json.dumps({'q':dict(evidence_a='First plan has no explanation.',evidence_b='Second plan explains the step.',
            reason='B provides the useful explanation.',winner='B',strength='clear')})
        context=ComparisonContext('gre_quant','subject',[{'question':'Explain the step','is_correct':False}],[{'id':'q','scale_id':'s','question':'Which explains better?'}])
        with patch('studentbench_leaderboard.judge.LiteLLMJudge.complete',return_value=(raw,{})):
            result=compare_plans({'steps':['do it']},{'steps':['explain it']},'judge_5',context=context)
        self.assertEqual(result['ratings']['q']['score'],-2)
        self.assertEqual(result['calls'][0]['raw_response'],raw)
        self.assertEqual(result['judge']['structure']['response_style'],'evidence_winner')

if __name__=='__main__':unittest.main()
