"""Build a private, offline, blinded viewer for saved evaluation comparisons."""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
from pathlib import Path

from .comparison import ConfiguredJudge
from .data import digest
from .evaluation import _read_call
from .generation import load_generation
from .judge import PRETEST_FIELDS, Task
from .judge_config import load_definition
from .structured_judge import project_plan

REPO_ROOT = Path(__file__).resolve().parents[2]


def collect_pairs(benchmark, generation_dirs, evaluation_dir, focus, seed=20261103):
    """Verify saved inputs and calls, and select all judged pairs for a model."""
    plans = dict(benchmark.plans)
    for directory in generation_dirs:
        for plan in load_generation(directory, benchmark):
            if plan['plan_id'] in plans:
                raise ValueError('Duplicate plan ID')
            plans[plan['plan_id']] = plan
    models = {p['ai_model_preset_id'] for p in plans.values()}
    if focus not in models:
        raise ValueError('Focus model not present in verified plans')
    directory = Path(evaluation_dir)
    header = json.loads((directory / 'schedule.json').read_text())
    if header.get('benchmark') != benchmark.fingerprint:
        raise ValueError('Evaluation schedule belongs to a different benchmark')
    if header.get('identity') != digest({k: v for k, v in header.items() if k != 'identity'}):
        raise ValueError('Evaluation schedule fingerprint mismatch')
    definition = load_definition(benchmark.judge_id, benchmark.judge_dir)
    judge = ConfiguredJudge(definition)
    if header.get('judge') != judge.configuration():
        raise ValueError('Evaluation judge does not match the frozen benchmark')
    requests = header.get('requests')
    if (not isinstance(requests, list) or not all(isinstance(r, str) and re.fullmatch(r'[0-9a-f]{64}', r) for r in requests)
            or len(requests) != len(set(requests))):
        raise ValueError('Invalid evaluation request list')
    rng = random.Random(seed)
    pairs, offset = [], 0
    for entry in header['schedule']:
        subject, ids = entry['subject'], entry['plan_ids']
        if (subject not in benchmark.contexts or len(ids) != 2 or len(set(ids)) != 2 or
                any(pid not in plans or subject not in plans[pid]['review_subject_ids'] for pid in ids)):
            raise ValueError('Invalid evaluation pair or student match')
        context = benchmark.contexts[subject]
        rubric = getattr(benchmark, 'rubrics', {}).get(entry.get('grading_id'), context['rubric'])
        task = Task(benchmark.instrument, subject, ids, [plans[p]['plan_as_stored'] for p in ids],
                    context['pretest'], rubric, benchmark.fingerprint)
        calls = judge.schedule(task)
        request_ids = requests[offset:offset + len(calls)]
        offset += len(calls)
        if len(request_ids) != len(calls):
            raise ValueError('Evaluation request list is incomplete')
        names = [plans[p]['ai_model_preset_id'] for p in ids]
        if names.count(focus) != 1:
            continue
        flipped = bool(rng.getrandbits(1))
        order = [1, 0] if flipped else [0, 1]
        judgments = []
        for index, ((member_index, swapped), request_id) in enumerate(zip(calls, request_ids)):
            path = directory / 'calls' / (request_id + '.json')
            if path.is_symlink() or not path.resolve().is_relative_to((directory / 'calls').resolve()):
                raise ValueError('Unsafe saved call path')
            if not path.is_file():
                break
            saved = json.loads(path.read_text())
            req = saved.get('request', {})
            ct = judge.call_task(task, swapped)
            expected = {'messages': ct.messages(), 'schema': ct.schema(), 'judge': judge.configuration(),
                        'route': header['routes'][member_index], 'member': definition.members[member_index].configuration(),
                        'subject': subject, 'plan_ids': ids, 'benchmark_sha256': benchmark.fingerprint,
                        'call_index': index}
            if (digest(req) != request_id or saved.get('request_id') != request_id or
                    any(req.get(k) != v for k, v in expected.items())):
                raise ValueError('Saved judge request does not match the frozen schedule')
            saved = _read_call(path, {'request': req, 'id': request_id, 'task': ct})
            # Scores refer to the judge's presented A; orient them to the viewer's A.
            reverse = swapped != flipped
            scores = {k: {**v, 'score': -v['score'] if reverse and v['score'] is not None else v['score']}
                      for k, v in saved['ratings'].items()}
            judgments.append({'judge_a': 'B' if reverse else 'A', 'ratings': scores})
        if len(judgments) != len(calls):
            continue
        fields = (*PRETEST_FIELDS, 'question_context')
        pretest = [{key: item[key] for key in fields if key in item} for item in context['pretest']]
        pairs.append({'id': digest([benchmark.fingerprint, subject, sorted(ids)]),
                      'plans': [project_plan(plans[ids[i]]['plan_as_stored'], definition.plan_projection) for i in order],
                      'models': [names[i] for i in order], 'plan_ids': [ids[i] for i in order], 'focus_slot': 'B' if flipped == (names[0] == focus) else 'A',
                      'pretest': pretest, 'judgments': judgments})
    if offset != len(requests):
        raise ValueError('Evaluation request count does not match schedule')
    if not pairs:
        raise ValueError('No completed judged pairs for the selected model')
    rng.shuffle(pairs)
    return pairs


def render_review(benchmark, generation_dirs, evaluation_dir, focus, output, *, ssot, seed=20261103):
    """Write a local private HTML viewer without modifying source artifacts."""
    raw_output = Path(os.path.abspath(output))
    output = raw_output.resolve()
    # Also discover checkout roots from the working directory and destination:
    # an installed package's source location may not be the active checkout.
    repositories = {REPO_ROOT.resolve()}
    for start in (Path.cwd().resolve(), raw_output, output):
        repositories.update(parent.resolve() for parent in (start, *start.parents)
                            if (parent / '.git').exists())
    for repo in repositories:
        results = repo / 'results'
        if raw_output.is_relative_to(repo):
            if (not raw_output.is_relative_to(results) or results.is_symlink() or
                    not output.is_relative_to(results)):
                raise ValueError('Repository-local review output must remain in the ignored results directory')
            if (repo / '.git').exists():
                try:
                    ignored = subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q', '--',
                                              str((output / 'index.html').relative_to(repo))],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             check=False).returncode == 0
                except OSError:
                    ignored = False
                if not ignored:
                    raise ValueError('Review output must be ignored by Git')
        elif output.is_relative_to(repo):
            raise ValueError('Review output cannot enter the repository through a symlink')
    protected = [Path(ssot).resolve(), benchmark.path.resolve(), Path(evaluation_dir).resolve(),
                 *(Path(p).resolve() for p in generation_dirs)]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Review output must not overlap source data or SSOT')
    target = output / 'index.html'
    if target.exists() or target.is_symlink():
        raise ValueError('Refusing to overwrite an existing review file')
    pairs = collect_pairs(benchmark, generation_dirs, evaluation_dir, focus, seed)
    corpus_id = digest({'focus': focus, 'seed': seed, 'assignments': [[p['id'], p.get('plan_ids', []), p['models']] for p in pairs]})
    payload = json.dumps({'pairs': pairs, 'focus': focus, 'seed': seed, 'corpus_id': corpus_id}, ensure_ascii=False, allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(HTML.replace('__PAYLOAD__', payload))
    return {'path': str(target), 'pairs': len(pairs)}

HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private blinded plan review</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f7;color:#1b303a;font:15px/1.5 system-ui,sans-serif}header,main{max-width:1500px;margin:auto;padding:18px 22px}header{border-bottom:1px solid #ccd9df}h1{margin:0;font-size:25px}h2{font-size:19px}button,select{font:inherit;padding:8px 12px;border:1px solid #9dafb8;background:#fff;border-radius:6px;cursor:pointer}button:hover{border-color:#087f83}button:focus-visible,select:focus-visible{outline:3px solid #cb9c35}button.selected{background:#087f83;color:white}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card,details{background:white;border:1px solid #d5e0e4;border-radius:8px;padding:14px}details{margin:7px 0}summary{font-weight:650;cursor:pointer}.label{font-weight:650;text-transform:capitalize}.field{margin:8px 0}.value{white-space:pre-wrap;overflow-wrap:anywhere}.item{border-left:3px solid #c9d8dd;padding:4px 10px;margin:8px 0}.small{font-size:13px;color:#536b76}.notice{padding:10px;background:#e8f3f1;border-radius:5px}.hidden{display:none}@media(max-width:800px){.cards{grid-template-columns:1fr}}
</style></head><body><header><h1>Private blinded plan review</h1><p class="small">Same student, two plans. All saved comparisons are shown in a reproducibly shuffled order; A/B assignment is randomized. Results are saved only in this browser. Exported reviews include model identities and A/B assignments. This file contains sensitive student and plan data: keep it private.</p><div class="toolbar"><button id="prev">Previous</button><button id="next">Next</button><button id="random">Random pair</button><strong id="position"></strong><label>View <select id="filter"><option value="all">All pairs</option><option value="focus-win">Revealed · focus favored</option><option value="tie">Revealed · tie</option><option value="focus-loss">Revealed · focus not favored</option></select></label></div></header><main><details><summary>Student pretest and answers</summary><div id="pretest"></div></details><div class="cards"><section class="card"><h2>Plan A</h2><div id="plan-a"></div></section><section class="card"><h2>Plan B</h2><div id="plan-b"></div></section></div><div class="toolbar"><strong>Your preference:</strong><button data-choice="A">A</button><button data-choice="B">B</button><button data-choice="tie">Tie</button><button data-choice="insufficient">Insufficient evidence</button><button id="reveal">Reveal models and judge</button></div><div id="revealed" class="notice hidden"></div><div class="toolbar"><button id="export">Export my reviews (JSON)</button><button id="clear">Clear saved reviews</button></div><p class="small">Math and other source text are shown as saved. The revealed filter is exploratory: direction averages the available criterion votes for this pair, not the fitted leaderboard score. Missing judge answers are not ties.</p></main><script type="application/json" id="review-data">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent),pairs=data.pairs,$=id=>document.getElementById(id);
const storageKey='studentbench-review-v2-'+data.corpus_id;let records={};try{records=JSON.parse(localStorage.getItem(storageKey)||'{}')||{}}catch(e){}let current=0;
function save(){try{localStorage.setItem(storageKey,JSON.stringify(records))}catch(e){alert('Browser storage is unavailable. Export your reviews before closing.')}}
function node(tag,text,cls){const n=document.createElement(tag);if(text!=null)n.textContent=String(text);if(cls)n.className=cls;return n}
const FIELD_ORDER={plan:['summary','concepts'],concept:['name','why_it_matters','pretest_evidence','priority_rank','allotted_minutes','practice_problems'],question:['stem','format','difficulty','options','answer','solution'],pretest:['presented_position','question_id','question_text','question_context','format','options_json','student_answer','correct_answer','is_correct','skill','subskill','difficulty','time_spent_seconds','answer_changes']};
function renderValue(parent,value,key='',kind=''){if(value===null||value===undefined){parent.append(node('div','null','value'));return}if(typeof value!=='object'){if(key==='options_json'&&typeof value==='string'){try{return renderValue(parent,JSON.parse(value),key)}catch(e){}}parent.append(node('div',value===''?'""':value,'value'));return}if(Array.isArray(value)){if(!value.length)parent.append(node('div','[]','value'));value.forEach((v,i)=>{const d=node('div',null,'item');if(value.length>1)d.append(node('strong',String(i+1)+'.'));renderValue(d,v,'',kind);parent.append(d)});return}const order=FIELD_ORDER[kind]||[],keys=Object.keys(value).sort((a,b)=>{const ai=order.indexOf(a),bi=order.indexOf(b);return (ai<0?order.length:ai)-(bi<0?order.length:bi)||a.localeCompare(b)});if(!keys.length)parent.append(node('div','{}','value'));for(const k of keys){const v=value[k],block=/answer|solution|explanation/i.test(k)?node('details'):node('div',null,'field');block.append(node(block.tagName==='DETAILS'?'summary':'div',k.replaceAll('_',' '),'label'));const childKind=k==='concepts'?'concept':k==='practice_problems'?'question':'';renderValue(block,v,k,childKind);parent.append(block)}}
function outcome(p){let values=[];for(const call of p.judgments)for(const v of Object.values(call.ratings)){if(v.score!=null)values.push(Math.sign(v.score)*(p.focus_slot==='A'?1:-1))}if(!values.length)return null;const mean=values.reduce((a,b)=>a+b,0)/values.length;return mean>0?'focus-win':mean<0?'focus-loss':'tie'}
function visible(){const f=$('filter').value;return pairs.map((p,i)=>i).filter(i=>f==='all'||(records[pairs[i].id]?.revealed&&outcome(pairs[i])===f))}
function draw(){const list=visible();if(!list.includes(current))current=list[0]??-1;const p=pairs[current];$('position').textContent=p?'Pair '+(list.indexOf(current)+1)+' / '+list.length:'No revealed pairs in this filter';for(const id of ['pretest','plan-a','plan-b','revealed'])$(id).replaceChildren();$('revealed').classList.add('hidden');document.querySelectorAll('[data-choice]').forEach(b=>{b.disabled=!p;b.classList.toggle('selected',!!p&&records[p.id]?.choice===b.dataset.choice)});if(!p)return;renderValue($('pretest'),p.pretest,'','pretest');renderValue($('plan-a'),p.plans[0],'','plan');renderValue($('plan-b'),p.plans[1],'','plan');if(records[p.id]?.revealed){const box=$('revealed');box.classList.remove('hidden');box.append(node('strong','Plan A: '+p.models[0]+' | Plan B: '+p.models[1]));box.append(node('p','Evidence is shown verbatim; any A/B labels in it refer to the judge’s original order. The mapping to the displayed plans is shown for each call.'));p.judgments.forEach((call,i)=>{const d=node('details');d.append(node('summary','Judge call '+(i+1)+' · judge’s original A = displayed '+call.judge_a));for(const [criterion,v] of Object.entries(call.ratings)){const direction=v.score==null?'Insufficient':v.score>0?'A favored':v.score<0?'B favored':'Tie';d.append(node('p',criterion+' · '+direction+' (score '+(v.score??'—')+')'));d.append(node('div',v.reason,'value'))}box.append(d)})}}
function move(delta){const list=visible();if(!list.length)return;current=list[(list.indexOf(current)+delta+list.length)%list.length];draw();window.scrollTo({top:0,behavior:'auto'})}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('random').onclick=()=>{const list=visible();if(list.length){current=list[Math.floor(Math.random()*list.length)];draw();window.scrollTo({top:0,behavior:'auto'})}};$('filter').onchange=draw;
document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{const p=pairs[current];if(!p)return;records[p.id]={...records[p.id],choice:b.dataset.choice,revealed:true};save();draw()});$('reveal').onclick=()=>{const p=pairs[current];if(!p)return;records[p.id]={...records[p.id],revealed:true};save();draw()};$('export').onclick=()=>{const blob=new Blob([JSON.stringify({version:2,corpus_id:data.corpus_id,seed:data.seed,focus:data.focus,reviews:pairs.filter(p=>records[p.id]).map(p=>({pair_id:p.id,plan_a_id:p.plan_ids[0],plan_b_id:p.plan_ids[1],model_a:p.models[0],model_b:p.models[1],...records[p.id]}))},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='private-plan-reviews.json';a.click();URL.revokeObjectURL(a.href)};$('clear').onclick=()=>{if(confirm('Clear all saved choices and reveal states for this review?')){records={};save();$('filter').value='all';draw()}};draw();
</script></body></html>'''
