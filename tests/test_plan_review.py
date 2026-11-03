import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studentbench_leaderboard.plan_review import collect_pairs, render_review
from tests import test_evaluation


class PlanReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_evaluation.EvaluationTests('test_budget_resume_and_connected_ranking')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        judge_path = self.fixture.path / 'judge_1.toml'
        judge_path.write_text(judge_path.read_text().replace('strategy = "single"', 'strategy = "single"\nplan_projection = "human_display"'))
        self.fixture.benchmark.plans['a']['plan_as_stored']['practice'] = [{'answer_original': 'ARCHIVAL SECRET', 'answer': 'displayed'}]
        self.fixture.benchmark.contexts['s1']['pretest'][0].update({'question_id': 'question-12', 'skill': 'algebra', 'subskill': 'equations', 'difficulty': 'hard', 'time_spent_seconds': 93, 'answer_changes': 2, 'question_context': 'source context', 'reviewer_id': 'PRIVATE REVIEWER', 'student_id': 'PRIVATE STUDENT'})
        self.fixture.run_eval()
        self.benchmark = self.fixture.benchmark
        self.benchmark.path = self.fixture.path / 'benchmark'
        self.benchmark.path.mkdir()
        self.loader = patch('studentbench_leaderboard.plan_review.load_generation', return_value=self.fixture.generated)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.args = (self.benchmark, [self.fixture.path / 'generation'], self.fixture.path / 'out', 'candidate')

    def test_same_student_random_orientation_and_verified_judgments(self):
        pairs = collect_pairs(*self.args)
        self.assertEqual(len(pairs), 2)
        self.assertNotIn('ARCHIVAL SECRET', json.dumps(pairs))
        self.assertIn('displayed', json.dumps(pairs))
        pretest = pairs[0]['pretest'][0]
        self.assertEqual({k: pretest[k] for k in ('question_id', 'skill', 'subskill', 'difficulty', 'time_spent_seconds', 'answer_changes', 'question_context')}, {'question_id': 'question-12', 'skill': 'algebra', 'subskill': 'equations', 'difficulty': 'hard', 'time_spent_seconds': 93, 'answer_changes': 2, 'question_context': 'source context'})
        self.assertNotIn('reviewer_id', pretest)
        self.assertNotIn('student_id', pretest)
        self.assertEqual(pairs, collect_pairs(*self.args))
        self.assertEqual({m for p in pairs for m in p['models'] if m != 'candidate'}, {'old-a', 'old-b'})
        for pair in pairs:
            self.assertEqual(pair['models'][0 if pair['focus_slot'] == 'A' else 1], 'candidate')
            call = pair['judgments'][0]
            self.assertEqual(call['ratings']['q']['score'], 1 if call['judge_a'] == 'A' else -1)
        with self.assertRaisesRegex(ValueError, 'Focus model'):
            collect_pairs(*self.args[:-1], 'missing')
        path = next((self.fixture.path / 'out/calls').glob('*.json'))
        saved = json.loads(path.read_text())
        saved['ratings']['q']['score'] = 0
        path.write_text(json.dumps(saved))
        with self.assertRaises(ValueError):
            collect_pairs(*self.args)

    def test_output_guard_escaping_and_no_network(self):
        self.benchmark.contexts['s1']['pretest'][0]['question_text'] = '</script><script>bad()</script>'
        # Keep request verification intact while exercising output escaping directly.
        fake = [{'id': 'x', 'plans': [{'text': '</script><script>bad()</script>'}, {}],
                 'models': ['candidate', 'other'], 'focus_slot': 'A', 'pretest': [], 'judgments': []}]
        with patch('studentbench_leaderboard.plan_review.collect_pairs', return_value=fake):
            root = self.fixture.path
            for output in (root / 'benchmark/inside', root / 'out/inside', root / 'generation/inside', root / 'ssot/inside'):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    render_review(*self.args, output, ssot=root / 'ssot')
            with tempfile.TemporaryDirectory() as temp:
                result = render_review(*self.args, Path(temp) / 'viewer', ssot=root / 'ssot')
                output = Path(result['path']).read_text()
                self.assertNotIn('</script><script>bad()', output)
                self.assertNotIn('<script src=', output)
                self.assertNotIn('fetch(', output)
                self.assertNotIn('https://', output)
                self.assertEqual(Path(result['path']).stat().st_mode & 0o777, 0o600)
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    render_review(*self.args, Path(temp) / 'viewer', ssot=root / 'ssot')

    def test_repository_outputs_are_restricted_and_symlinks_cannot_escape(self):
        fake = [{'id': 'x', 'plans': [{}, {}], 'models': ['a', 'b'],
                 'focus_slot': 'A', 'pretest': [], 'judgments': []}]
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            repo = repo.resolve()
            (repo / 'results').mkdir()
            (repo / 'escape').symlink_to(outside, target_is_directory=True)
            (Path(outside) / 'into-repo').symlink_to(repo, target_is_directory=True)
            with patch('studentbench_leaderboard.plan_review.REPO_ROOT', repo), patch('studentbench_leaderboard.plan_review.collect_pairs', return_value=fake):
                for path in (repo / 'reports/private-review', repo / '.github/private',
                             repo / '.git/private', repo / 'escape/private',
                             Path(outside) / 'into-repo/private'):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        render_review(*self.args, path, ssot=self.fixture.path / 'ssot')
                another = Path(outside).resolve() / 'another-checkout'
                (another / '.git').mkdir(parents=True)
                with self.assertRaises(ValueError):
                    render_review(*self.args, another / 'public/private', ssot=self.fixture.path / 'ssot')
                result = render_review(*self.args, repo / 'results/private', ssot=self.fixture.path / 'ssot')
                self.assertTrue(Path(result['path']).is_file())
                (repo / '.git').mkdir()
                with patch('studentbench_leaderboard.plan_review.subprocess.run') as check:
                    check.return_value.returncode = 1
                    with self.assertRaisesRegex(ValueError, 'ignored by Git'):
                        render_review(*self.args, repo / 'results/not-ignored', ssot=self.fixture.path / 'ssot')
                    check.return_value.returncode = 0
                    accepted = render_review(*self.args, repo / 'results/ignored', ssot=self.fixture.path / 'ssot')
                    self.assertTrue(Path(accepted['path']).exists())
                (repo / 'results/escape').symlink_to(outside, target_is_directory=True)
                with self.assertRaises(ValueError):
                    render_review(*self.args, repo / 'results/escape/private', ssot=self.fixture.path / 'ssot')

    def test_assignment_identity_changes_with_orientation_and_seed(self):
        fake = [{'id': 'same-pair', 'plan_ids': ['first', 'second'], 'plans': [{}, {}],
                 'models': ['a', 'b'], 'focus_slot': 'A', 'pretest': [], 'judgments': []}]
        def payload(document):
            return json.loads(document.split('<script type="application/json" id="review-data">')[1].split('</script>')[0])
        with tempfile.TemporaryDirectory() as tmp:
            with patch('studentbench_leaderboard.plan_review.collect_pairs', return_value=fake):
                one = payload(Path(render_review(*self.args, Path(tmp) / 'one', ssot=self.fixture.path / 'ssot', seed=1)['path']).read_text())
                seed = payload(Path(render_review(*self.args, Path(tmp) / 'seed', ssot=self.fixture.path / 'ssot', seed=2)['path']).read_text())
            swapped = [{**fake[0], 'plan_ids': ['second', 'first'], 'models': ['b', 'a'], 'focus_slot': 'B'}]
            with patch('studentbench_leaderboard.plan_review.collect_pairs', return_value=swapped):
                reverse = payload(Path(render_review(*self.args, Path(tmp) / 'reverse', ssot=self.fixture.path / 'ssot', seed=1)['path']).read_text())
            self.assertNotEqual(one['corpus_id'], reverse['corpus_id'])
            self.assertNotEqual(one['corpus_id'], seed['corpus_id'])
            self.assertEqual(one['pairs'][0]['plan_ids'], ['first', 'second'])
            self.assertEqual(reverse['pairs'][0]['plan_ids'], ['second', 'first'])

    @unittest.skipUnless(shutil.which('node'), 'Node required for browser logic test')
    def test_opponent_filter_preserves_blinding_and_assignments(self):
        common = {'pretest': [], 'judgments': [{'judge_a': 'A', 'ratings': {'q01': {'score': 1, 'reason': 'HIDDEN EVIDENCE'}}}]}
        pairs = [
            {**common, 'id': 'one', 'plan_ids': ['p1', 'p2'], 'plans': [{'text': 'first A'}, {'text': 'first B'}], 'models': ['candidate', 'opus-5-high'], 'focus_slot': 'A'},
            {**common, 'id': 'two', 'plan_ids': ['p3', 'p4'], 'plans': [{'text': 'second A'}, {'text': 'second B'}], 'models': ['opus-5-high', 'candidate'], 'focus_slot': 'B'},
            {**common, 'id': 'three', 'plan_ids': ['p5', 'p6'], 'plans': [{'text': 'third A'}, {'text': 'third B'}], 'models': ['candidate', 'other-model'], 'focus_slot': 'A'},
        ]
        with tempfile.TemporaryDirectory() as temp, patch('studentbench_leaderboard.plan_review.collect_pairs', return_value=pairs):
            result = render_review(*self.args, Path(temp) / 'viewer', ssot=self.fixture.path / 'ssot')
            document = Path(result['path']).read_text()
        harness = r"""
const fs=require('fs'),vm=require('vm'),html=fs.readFileSync(0,'utf8');
const data=html.split('<script type="application/json" id="review-data">')[1].split('</script>')[0];
const script=html.split('<script>')[1].split('</script>')[0];
class N{constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.textContent='';this.dataset={};this.value='all';this.classList={add(){},remove(){},toggle(){}}}append(...x){this.children.push(...x)}replaceChildren(){this.children=[]}click(){if(this.onclick)this.onclick()}}
const dump=n=>n.textContent+n.children.map(dump).join('');
function run(search){const nodes={};const get=id=>nodes[id]??=new N();get('review-data').textContent=data;get('opponent').value='';const choices=['A','B','tie','insufficient'].map(c=>{let n=new N('button');n.dataset.choice=c;return n});const document={getElementById:get,createElement:t=>new N(t),querySelectorAll:()=>choices};let blob;vm.runInNewContext(script,{document,window:{scrollTo(){},location:{search}},URLSearchParams,localStorage:{getItem:()=>null,setItem(){}},alert(){},confirm:()=>true,Blob:function(p){blob=p[0]},URL:{createObjectURL:()=> 'blob:local',revokeObjectURL(){}}});return {get,choices,exported:()=>JSON.parse(blob)}}
let a=run('?opponent=opus-5-high');const initial={opponent:a.get('opponent').value,position:a.get('position').textContent,planA:dump(a.get('plan-a')),revealed:dump(a.get('revealed'))};a.get('next').click();const second={position:a.get('position').textContent,planA:dump(a.get('plan-a')),revealed:dump(a.get('revealed'))};a.choices[0].click();a.get('export').click();const exported=a.exported();a.get('filter').value='focus-loss';a.get('filter').onchange();const intersection=a.get('position').textContent;a.get('opponent').value='other-model';a.get('opponent').onchange();const empty=a.get('position').textContent;const unknown=run('?opponent=%3Cscript%3E').get('opponent').value;const defaultPosition=run('').get('position').textContent;process.stdout.write(JSON.stringify({initial,second,exported,intersection,empty,unknown,defaultPosition}));
"""
        result = subprocess.run(['node', '-e', harness], input=document, text=True, capture_output=True, check=True)
        shown = json.loads(result.stdout)
        self.assertEqual(shown['initial']['opponent'], 'opus-5-high')
        self.assertEqual(shown['initial']['position'], 'Pair 1 / 2')
        self.assertIn('first A', shown['initial']['planA'])
        self.assertEqual(shown['initial']['revealed'], '')
        self.assertEqual(shown['second']['position'], 'Pair 2 / 2')
        self.assertIn('second A', shown['second']['planA'])
        self.assertEqual(shown['second']['revealed'], '')
        self.assertEqual(shown['exported']['reviews'][0]['plan_a_id'], 'p3')
        self.assertEqual(shown['exported']['reviews'][0]['model_a'], 'opus-5-high')
        self.assertEqual(shown['intersection'], 'Pair 1 / 1')
        self.assertEqual(shown['empty'], 'No revealed pairs in this filter')
        self.assertEqual(shown['unknown'], '')
        self.assertEqual(shown['defaultPosition'], 'Pair 1 / 3')

    @unittest.skipUnless(shutil.which('node'), 'Node required for browser logic test')
    def test_blind_until_choice_or_reveal(self):
        payload = [{'id': 'x', 'plan_ids': ['plan-1', 'plan-2'], 'plans': [{'summary': 'summary value', 'concepts': [{'name': 'name value', 'why_it_matters': 'why value', 'practice_problems': [{'stem': 'stem value', 'format': 'format value', 'options': ['option value'], 'answer': 'answer value', 'solution': 'solution value'}]}]}, {'concepts': [{'practice_problems': [{'solution': 'solution value', 'answer': 'answer value', 'options': ['option value'], 'format': 'format value', 'stem': 'stem value'}], 'why_it_matters': 'why value', 'name': 'name value'}], 'summary': 'summary value'}],
                    'models': ['SECRET MODEL ONE', 'SECRET MODEL TWO'], 'focus_slot': 'A', 'pretest': [],
                    'judgments': [{'judge_a': 'B', 'ratings': {'q01': {'score': -1, 'reason': 'SECRET EVIDENCE'}}}]}]
        payload.append({**payload[0], 'id': 'y', 'plan_ids': ['plan-3', 'plan-4']})
        with tempfile.TemporaryDirectory() as temp, patch('studentbench_leaderboard.plan_review.collect_pairs', return_value=payload):
            result = render_review(*self.args, Path(temp) / 'viewer', ssot=self.fixture.path / 'ssot')
            document = Path(result['path']).read_text()
        harness = r'''
const fs=require('fs'),vm=require('vm'),html=fs.readFileSync(0,'utf8');
const data=html.split('<script type="application/json" id="review-data">')[1].split('</script>')[0];
const script=html.split('<script>')[1].split('</script>')[0];
class N{constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.textContent='';this.dataset={};this.value='all';this.classList={add(){},remove(){},toggle(){}}}append(...x){this.children.push(...x)}replaceChildren(){this.children=[]}click(){if(this.onclick)this.onclick()}}
const nodes={};const get=id=>nodes[id]??=new N();get('opponent').value='';get('review-data').textContent=data;const choices=['A','B','tie','insufficient'].map(c=>{let n=new N('button');n.dataset.choice=c;return n});const document={getElementById:get,createElement:t=>new N(t),querySelectorAll:()=>choices};
let blob,scrolls=0;vm.runInNewContext(script,{document,window:{scrollTo:()=>scrolls++,location:{search:''}},URLSearchParams,localStorage:{getItem:()=>null,setItem(){}},alert(){},confirm:()=>true,Blob:function(parts){blob=parts[0]},URL:{createObjectURL:()=> 'blob:local',revokeObjectURL(){}}});
const dump=n=>n.textContent+n.children.map(dump).join('');let before=dump(get('revealed')),planA=dump(get('plan-a')),planB=dump(get('plan-b'));get('next').click();get('prev').click();choices[0].click();let after=dump(get('revealed'));get('export').click();process.stdout.write(JSON.stringify({before,after,planA,planB,scrolls,exported:JSON.parse(blob)}));
'''
        result = subprocess.run(['node', '-e', harness], input=document, text=True, capture_output=True, check=True)
        shown = json.loads(result.stdout)
        self.assertNotIn('SECRET', shown['before'])
        self.assertEqual(shown['planA'], shown['planB'])
        for label in ('summary value', 'name value', 'why value', 'stem value', 'format value', 'option value', 'answer value', 'solution value'):
            self.assertIn(label, shown['planA'])
        self.assertEqual(sorted(['summary value', 'name value', 'why value', 'stem value', 'format value', 'option value', 'answer value', 'solution value'], key=shown['planA'].index),
                         ['summary value', 'name value', 'why value', 'stem value', 'format value', 'option value', 'answer value', 'solution value'])
        self.assertEqual(shown['scrolls'], 2)
        self.assertIn('SECRET MODEL ONE', shown['after'])
        self.assertIn('SECRET EVIDENCE', shown['after'])
        self.assertIn('judge’s original A = displayed B', shown['after'])
        self.assertIn('Evidence is shown verbatim', shown['after'])
        self.assertEqual(shown['exported']['reviews'][0]['plan_a_id'], 'plan-1')
        self.assertEqual(shown['exported']['reviews'][0]['model_b'], 'SECRET MODEL TWO')
        self.assertEqual(shown['exported']['reviews'][0]['choice'], 'A')
        self.assertEqual(shown['exported']['seed'], 20261103)


if __name__ == '__main__':
    unittest.main()
