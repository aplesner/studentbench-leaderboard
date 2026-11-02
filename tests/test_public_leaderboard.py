"""The optional public view must stay aggregate and never borrow old ranks."""
import json
import copy
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from studentbench_leaderboard.study_report import render_report, render_public_report


class PublicLeaderboardTests(unittest.TestCase):
    def test_absent_pointer_preserves_legacy_view(self):
        with TemporaryDirectory() as tmp:
            output = render_report(Path(tmp)).read_text()
            self.assertNotIn('id="public-leaderboard"', output)
            self.assertNotIn('"public_leaderboard"', output)

    def test_pending_missing_and_unsafe_pointer(self):
        with TemporaryDirectory() as tmp:
            study = Path(tmp) / 'study'
            study.mkdir()
            outside = Path(tmp).parent / 'secret-leaderboard.json'
            for filename in ('missing.json', str(outside), '../../outside.json'):
                (study / 'report_leaderboard.json').write_text(json.dumps({'evaluation_file': filename}))
                output = render_report(study).read_text()
                self.assertIn('"public_leaderboard": {"status": "pending"', output)
                self.assertLess(output.index('id="public-leaderboard"'), output.index('id="overview"'))
                self.assertLess(output.index('href="#public-leaderboard"'), output.index('href="#overview"'))
                self.assertIn('No ranks are inferred from the earlier 12-system study', output)

    def test_symlink_escape_and_partial_fit_remain_pending(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            study = Path(tmp) / 'study'
            study.mkdir()
            external = Path(outside) / 'evaluation.json'
            external.write_text(json.dumps({'complete': True, 'raw_response': 'EXTERNAL-SECRET'}))
            (study / 'link.json').symlink_to(external)
            (study / 'report_leaderboard.json').write_text(json.dumps({'evaluation_file': 'link.json'}))
            output = render_report(study).read_text()
            self.assertNotIn('EXTERNAL-SECRET', output)
            self.assertIn('"status": "pending"', output)
            (study / 'partial.json').write_text(json.dumps({'complete': False, 'ranking': {'connected': True, 'leaderboard': [{'model': str(i), 'rank': i} for i in range(20)]}}))
            (study / 'report_leaderboard.json').write_text(json.dumps({'evaluation_file': 'partial.json'}))
            output = render_report(study).read_text()
            self.assertIn('"status": "pending"', output)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to execute the report JavaScript')
    def test_runtime_requires_exact_roster_and_convergence(self):
        with TemporaryDirectory() as tmp:
            study = Path(tmp) / 'study'
            study.mkdir()
            baseline = [f'old-{i}' for i in range(12)]
            new = [f'new-{i}' for i in range(8)]
            roster = baseline + new
            (study / 'test_analysis.json').write_text(json.dumps({'models': baseline}))
            (study / 'report_leaderboard.json').write_text(json.dumps({'evaluation_file': 'evaluation.json'}))
            fit = {'connected': True, 'converged': True, 'models': roster,
                   'leaderboard': [{'model': model, 'rank': i + 1, 'score': .1} for i, model in enumerate(roster)]}
            base = {'complete': True, 'ranking': fit, 'requested_models': new,
                    'generation_coverage': [{'model': model, 'label': 'Friendly '+model, 'status': 'ready', 'settings': {'max_tokens': 32768}} for model in new],
                    'evaluation_conditions': {'routes': [{'model': 'judge-model', 'reasoning_effort': 'low', 'settings': {'max_tokens': 8192, 'api_key': 'SECRET-KEY'}, 'endpoint': 'SECRET-ENDPOINT', 'profile': 'SECRET-PROFILE'}]}}
            cases = []
            cases.append(('complete', base, True))
            active17 = copy.deepcopy(base)
            active17['requested_models'] = new[:5]
            active17['generation_coverage'] = active17['generation_coverage'][:5]
            active17['ranking']['models'] = roster[:17]
            active17['ranking']['leaderboard'] = active17['ranking']['leaderboard'][:17]
            active17['deferred_models'] = [{'model': model, 'label': model} for model in new[5:]]
            cases.append(('17 active models ready', active17, True))
            pending17 = copy.deepcopy(active17)
            pending17['complete'] = False
            cases.append(('17 active models pending', pending17, False))
            missing_explicit = copy.deepcopy(active17)
            del missing_explicit['requested_models']
            cases.append(('17 named generation coverage fallback', missing_explicit, True))
            missing_active = copy.deepcopy(active17)
            missing_active['generation_coverage'].pop()
            cases.append(('17 missing active generation', missing_active, False))
            empty_active = copy.deepcopy(active17)
            empty_active['requested_models'] = []
            cases.append(('empty requested list', empty_active, False))
            missing_convergence = copy.deepcopy(base)
            del missing_convergence['ranking']['converged']
            cases.append(('missing convergence', missing_convergence, False))
            wrong_fit_roster = copy.deepcopy(base)
            wrong_fit_roster['ranking']['models'][0] = 'intruder'
            cases.append(('wrong fit roster', wrong_fit_roster, False))
            wrong_rows = copy.deepcopy(base)
            wrong_rows['ranking']['leaderboard'][0]['model'] = 'intruder'
            cases.append(('wrong row roster', wrong_rows, False))
            duplicate = copy.deepcopy(base)
            duplicate['ranking']['leaderboard'][0]['model'] = roster[1]
            cases.append(('duplicate row', duplicate, False))
            missing_generation = copy.deepcopy(base)
            missing_generation['generation_coverage'].pop()
            cases.append(('missing generation identity', missing_generation, False))
            malformed = copy.deepcopy(base)
            malformed['evaluation_conditions'] = {'judge': {'members': None}}
            malformed['ranking']['models'] = None
            malformed['coverage'] = []
            malformed['bootstrap'] = 'not a dictionary'
            malformed['generation_coverage'][0]['settings'] = None
            cases.append(('malformed nested data', malformed, False))
            harness = r"""
const fs=require('fs'),vm=require('vm'),text=fs.readFileSync(0,'utf8');
const data=text.split('<script type="application/json" id="study-data">')[1].split('</script>')[0];
const script=text.split('<script>')[1].split('</script>')[0],nodes={};
const get=id=>nodes[id]??={textContent:id==='study-data'?data:'',innerHTML:'',value:'overall',classList:{toggle(){}},addEventListener(){}};
vm.runInNewContext(script,{document:{getElementById:get,addEventListener(){}}});
process.stdout.write(JSON.stringify({status:get('public-status').textContent,ranking:get('public-ranking').innerHTML,settings:get('public-model-settings').innerHTML,coverage:get('public-coverage').innerHTML}));
"""
            for label, artifact, ready in cases:
                with self.subTest(label=label):
                    (study / 'evaluation.json').write_text(json.dumps(artifact))
                    document = render_public_report(study).read_text()
                    result = subprocess.run(['node', '-e', harness], input=document, text=True, capture_output=True, check=True)
                    displayed = json.loads(result.stdout)
                    self.assertEqual(displayed['status'], 'Complete' if ready else 'Pending / unavailable')
                    self.assertIn('max tokens: 32768', displayed['settings'])
                    self.assertIn('83 frozen benchmark student contexts', document)
                    self.assertIn('delivered schema-valid plans', document)
                    self.assertIn('Historical baseline plans can include answer-verifier edits', document)
                    self.assertIn('without that historical answer verifier', document)
                    self.assertIn('generation pipelines are not historically equivalent', document)
                    if label != 'malformed nested data':
                        self.assertIn('max tokens: 8192', displayed['coverage'])
                        for secret in ('SECRET-KEY', 'SECRET-ENDPOINT', 'SECRET-PROFILE'):
                            self.assertNotIn(secret, document)
                    if ready:
                        self.assertIn('95% student bootstrap rank range', displayed['ranking'])
                        self.assertIn('Friendly new-0', displayed['ranking'])
                    else:
                        self.assertIn('not available yet', displayed['ranking'])
                        self.assertNotIn('<strong>old', displayed['ranking'])

    def test_complete_artifact_whitelists_aggregates(self):
        with TemporaryDirectory() as tmp:
            study = Path(tmp) / 'study'
            study.mkdir()
            rows = [{'model': f'model-{i}', 'rank': i + 1, 'score': .2} for i in range(20)]
            data = {'complete': True, 'ranking': {'connected': True, 'converged': True, 'leaderboard': rows},
                    'per_rubric': {'q01': {'connected': True, 'leaderboard': rows}},
                    'rank_intervals': {'model-0': {'low': 1, 'high': 4}},
                    'coverage': {'completed_comparisons': 10, 'scheduled_comparisons': 10},
                    'generation_coverage': [{'model': 'model-0', 'reasoning_effort': 'high',
                        'selected_subjects': ['PRIVATE-STUDENT'], 'completed_subjects': ['PRIVATE-STUDENT'],
                        'settings': {'temperature': .5, 'api_key': 'PRIVATE-KEY'}}],
                    'evaluation_conditions': {'judge': {'model': 'judge-name', 'endpoint': 'PRIVATE-ENDPOINT'}},
                    'raw_response': 'PRIVATE-PLAN'}
            (Path(tmp) / 'evaluation.json').write_text(json.dumps(data))
            (study / 'report_leaderboard.json').write_text(json.dumps({'evaluation_file': '../evaluation.json'}))
            output = render_report(study).read_text()
            payload = json.loads(output.split('<script type="application/json" id="study-data">')[1].split('</script>')[0])['public_leaderboard']
            self.assertEqual(len(payload['ranking']['leaderboard']), 20)
            self.assertEqual(payload['generation_coverage'][0]['selected_subjects'], 1)
            self.assertEqual(payload['generation_coverage'][0]['reasoning_effort'], 'high')
            self.assertEqual(payload['per_rubric']['q01']['leaderboard'], rows_with_nulls(rows))
            for secret in ('PRIVATE-STUDENT', 'PRIVATE-KEY', 'PRIVATE-ENDPOINT', 'PRIVATE-PLAN'):
                self.assertNotIn(secret, output)
            self.assertIn('AI-judged quality', output)
            self.assertIn('student-context bootstrap', output)
            (study / 'audit').mkdir()
            (study / 'audit/findings.json').write_text('[{"finding":"PRIVATE-VALIDATION-EXAMPLE"}]')
            (study / 'selection.json').write_text(json.dumps({'variant': 'judge-a'}))
            (study / 'test_analysis.json').write_text(json.dumps({'n_pairs': 155, 'n_subjects': 83, 'models': ['original'], 'judges': [{'variant': 'judge-a', 'metrics': {'kendall_tau_b': .8}}]}))
            standalone = render_public_report(study).read_text()
            self.assertIn('Judge validation and methods', standalone)
            self.assertIn('"kendall_tau_b": 0.8', standalone)
            self.assertNotIn('PRIVATE-VALIDATION-EXAMPLE', standalone)
            self.assertNotIn('id="overview"', standalone)


def rows_with_nulls(rows):
    return [{**row, 'n_comparisons': None, 'n_subjects': None} for row in rows]


if __name__ == '__main__':
    unittest.main()
