"""Versioned, resumable lesson-plan generation with no human-review inputs."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import threading
import uuid
from .data import digest

RENDERER = 'studentbench.raw-pretest.v1'
PARSER = 'studentbench.options-normalization.v1'
STR = {'type':'string'}
PROBLEM = {'type':'object', 'properties':{'stem':STR,'format':{'type':'string','enum':['MCQ','SPR','open']},
 'options':{'type':'object','properties':{x:STR for x in 'ABCD'},'required':list('ABCD'),'additionalProperties':False},
 'answer':STR,'solution':STR,'difficulty':{'type':'string','enum':['Easy','Medium','Hard']}},
 'required':['stem','format','options','answer','solution','difficulty'], 'additionalProperties':False}
CONCEPT = {'type':'object','properties':{'concept_id':STR,'name':STR,'why_it_matters':STR,
 'pretest_evidence':{'type':'array','items':STR},'priority_rank':{'type':'integer'},
 'allotted_minutes':{'type':'integer'},'practice_problems':{'type':'array','items':PROBLEM}},
 'required':['concept_id','name','why_it_matters','pretest_evidence','priority_rank','allotted_minutes','practice_problems'], 'additionalProperties':False}
SCHEMA = {'type':'object','properties':{'summary':STR,'concepts':{'type':'array','items':CONCEPT}},
          'required':['summary','concepts'],'additionalProperties':False}
# The released prompt permits omitted options, but strict provider schemas require
# fixed properties; empty strings represent unused options and are removed below.
GEN_FIELDS = ('presented_position','question_id','question_text','options_json','correct_answer',
              'student_answer','is_correct','time_spent_seconds')

def _safe_output(output):
    output = Path(output)
    if output.is_symlink(): raise ValueError(f'Symlink in generation output: {output}')
    if output.exists():
        for path in output.rglob('*'):
            if path.is_symlink(): raise ValueError(f'Symlink in generation output: {path}')

def _write(path, obj):
    if path.is_symlink(): raise ValueError(f'Symlink output target: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    if temp.is_symlink(): raise ValueError(f'Symlink temporary target: {temp}')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)

def _write_new(path, obj):
    """Atomically publish a new ledger entry without replacing any prior entry."""
    if path.is_symlink(): raise ValueError(f'Symlink output target: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(temp, path, follow_symlinks=False)
    finally:
        temp.unlink(missing_ok=True)

@contextmanager
def _lock(output):
    _safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output / '.generation.lock', os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError(f'Generation output is locked: {output}') from None
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)

def _request(benchmark, subject, config, model, settings):
    pretest = []
    for row in benchmark.contexts[subject]['pretest']:
        item = {k:row[k] for k in GEN_FIELDS if k in row}
        if row.get('question_context'):
            context = row['question_context']
            item['question_context'] = {k:context[k] for k in ('quantity_a','quantity_b','shared_data','figure_status','missing_evidence') if k in context}
        pretest.append(item)
    prompt = benchmark.generation_prompt.replace('{treatment_minutes}', '60').replace('{{','{').replace('}}','}')
    prompt += '\nFor this JSON schema, always include options with A, B, C and D; use empty strings for unused options.\n'
    messages = [{'role':'system','content':prompt},{'role':'user','content':json.dumps({'pretest':pretest},ensure_ascii=False,allow_nan=False)}]
    request = {'benchmark_sha256':benchmark.fingerprint,'subject':subject,'model':model,'configuration':config,
               'messages':messages,'schema':SCHEMA,'renderer':RENDERER,'parser':PARSER,'settings':settings,'draw':0}
    return request

def _unique(pairs):
    d = {}
    for k,v in pairs:
        if k in d: raise ValueError(f'Duplicate JSON key: {k}')
        d[k]=v
    return d

def _validate(raw):
    p = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(p,dict) or set(p) != {'summary','concepts'} or not isinstance(p['summary'],str) or not p['summary'].strip() or not isinstance(p['concepts'],list) or not p['concepts']:
        raise ValueError('Plan must contain a summary and nonempty concepts')
    ranks=[]; total=0; ids=set()
    for c in p['concepts']:
        if not isinstance(c,dict) or set(c) != set(CONCEPT['required']): raise ValueError('Invalid concept fields')
        if any(not isinstance(c[k],str) or not c[k].strip() for k in ('concept_id','name','why_it_matters')): raise ValueError('Empty concept text')
        if c['concept_id'] in ids: raise ValueError('Duplicate concept ID')
        ids.add(c['concept_id'])
        if type(c['priority_rank']) is not int or type(c['allotted_minutes']) is not int or c['allotted_minutes'] <= 0: raise ValueError('Invalid concept rank or duration')
        ranks.append(c['priority_rank']); total += c['allotted_minutes']
        if not isinstance(c['pretest_evidence'],list) or any(not isinstance(x,str) for x in c['pretest_evidence']): raise ValueError('Invalid evidence')
        if not isinstance(c['practice_problems'],list) or not c['practice_problems']: raise ValueError('Concept has no practice problems')
        for q in c['practice_problems']:
            if not isinstance(q,dict) or set(q) != set(PROBLEM['required']): raise ValueError('Invalid problem fields')
            if any(not isinstance(q[k],str) or not q[k].strip() for k in ('stem','answer','solution')): raise ValueError('Empty problem text')
            if q['format'] not in ('MCQ','SPR','open') or q['difficulty'] not in ('Easy','Medium','Hard'): raise ValueError('Invalid problem format or difficulty')
            opts=q['options']
            if not isinstance(opts,dict) or set(opts) != set('ABCD') or any(not isinstance(v,str) for v in opts.values()): raise ValueError('Invalid options')
            opts={k:v for k,v in opts.items() if v.strip()}
            if q['format']=='MCQ':
                if len(opts)<2 or q['answer'] not in opts: raise ValueError('Invalid MCQ options or answer')
                q['options']=opts
            else:
                if opts: raise ValueError('Only MCQ problems may contain options')
                q['options']={}
                if q['format']=='SPR' and not re.fullmatch(r'-?(?:\d+(?:\.\d+)?|\d+/\d+)',q['answer']): raise ValueError('Invalid numeric answer')
    if sorted(ranks)!=list(range(1,len(ranks)+1)) or total != 60: raise ValueError('Ranks or 60-minute allocation invalid')
    return p

def _record(path, benchmark):
    r=json.loads(path.read_text())
    if r.get('request_id') != digest(r.get('request')) or r['request'].get('benchmark_sha256') != benchmark.fingerprint:
        raise ValueError(f'Generation request integrity failure: {path}')
    plan=_validate(r['raw_response'])
    if plan != r.get('plan_as_stored'): raise ValueError(f'Generated plan differs from response: {path}')
    subject=r['request']['subject']; model=r['request']['model']; pid='generated_'+r['request_id'][:24]
    if subject not in benchmark.contexts or r.get('lesson_plan_id') != pid or r.get('review_subject_ids') != [subject] or r.get('ai_model_preset_id') != model:
        raise ValueError(f'Generation identity mismatch: {path}')
    return {'plan_id':pid,'lesson_plan_id':pid,'ai_model_preset_id':model,'review_subject_ids':[subject],'plan_as_stored':plan}

def load_generation(output, benchmark):
    output=Path(output); _safe_output(output); directory=output / 'generations'
    paths=sorted(directory.glob('*.json')) if directory.exists() else []
    manifest_path=output/'generation_manifest.json'
    if not manifest_path.exists():
        if paths: raise ValueError('Generation manifest missing')
        return []
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('benchmark_sha256') != benchmark.fingerprint or manifest.get('schema_version') != 1:
        raise ValueError('Generation manifest does not match benchmark')
    subjects=manifest.get('selected_subjects')
    if not isinstance(subjects,list) or subjects != sorted(set(subjects)) or any(s not in benchmark.contexts for s in subjects):
        raise ValueError('Invalid generation selection')
    expected={digest(_request(benchmark, subject, manifest['configuration'], manifest['model'], manifest['settings']))
              for subject in subjects}
    result=[]
    for path in paths:
        if path.stem not in expected: raise ValueError(f'Unexpected saved generation request: {path}')
        record=_record(path,benchmark)
        if path.stem != record['lesson_plan_id'][10:] and not path.stem.startswith(record['lesson_plan_id'][10:]):
            raise ValueError(f'Generation filename mismatch: {path}')
        result.append(record)
    return result

def generation_coverage(output, benchmark):
    """Return the frozen requested subjects and validated completion coverage."""
    plans = load_generation(output, benchmark)
    manifest_path = Path(output) / 'generation_manifest.json'
    if not manifest_path.exists():
        return {'selected_subjects': [], 'completed_subjects': [], 'pending_subjects': []}
    manifest = json.loads(manifest_path.read_text())
    selected = manifest['selected_subjects']
    completed = sorted({p['review_subject_ids'][0] for p in plans})
    return {'selected_subjects':selected, 'completed_subjects':completed,
            'pending_subjects':sorted(set(selected)-set(completed))}

def run_generation(benchmark, output, model, client, *, dry_run=False, limit=None, reasoning_effort=None,
                   max_new_calls=1000, workers=1, timeout=180, **settings):
    output=Path(output).absolute()
    _safe_output(output)
    output=output.resolve()
    if output.is_relative_to(benchmark.path) or benchmark.path.is_relative_to(output): raise ValueError('Generation output must be separate from benchmark')
    configured_ssot = Path(os.environ.get('STUDENTBENCH_SSOT', 'studentbench_data_ssot')).resolve()
    if output.is_relative_to(configured_ssot) or any((p / 'release_manifest.json').exists() for p in (output, *output.parents)):
        raise ValueError('Generation output must be outside the SSOT')
    if limit is not None and limit < 0 or max_new_calls < 0 or workers < 1 or timeout <= 0: raise ValueError('Invalid generation limits')
    config=client.configuration(model,reasoning_effort=reasoning_effort,**settings)
    subjects=sorted(benchmark.contexts)[:limit] if limit is not None else sorted(benchmark.contexts)
    requests=[_request(benchmark,s,config,model,settings) for s in subjects]
    run_manifest={'schema_version':1,'benchmark_sha256':benchmark.fingerprint,'model':model,
                  'configuration':config,'settings':settings,'renderer':RENDERER,'parser':PARSER,
                  'selected_subjects':subjects}
    def execute():
        manifest_path=output/'generation_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != run_manifest:
            raise ValueError('Output contains generations for a different request configuration')
        load_generation(output,benchmark)
        if not dry_run and not manifest_path.exists(): _write(manifest_path,run_manifest)
        todo=[]; reused=0
        for req in requests:
            path=output/'generations'/f'{digest(req)}.json'
            if path.exists(): _record(path,benchmark); reused+=1
            else: todo.append(req)
        counts={'scheduled':len(requests),'reused':reused,'pending':len(todo),'new_calls_planned':min(len(todo),max_new_calls), 'completed':0,'failed':0,'dry_run':dry_run,'estimated_cost':'unknown'}
        if dry_run:
            counts['requests'] = [{'request_id':digest(req), **req} for req in todo[:max_new_calls]]
            counts['complete'] = not todo
            return counts
        budget_lock = threading.Lock()
        invocation_id = uuid.uuid4().hex
        attempts_started = 0
        def call(req):
            nonlocal attempts_started
            key=digest(req); path=output/'generations'/f'{key}.json'
            for attempt in range(1,4):
                with budget_lock:
                    if attempts_started >= max_new_calls: return False if attempt > 1 else None
                    attempts_started += 1
                start=time.monotonic()
                attempt_path = None
                try:
                    raw, meta=client.complete(model,req['messages'],SCHEMA,reasoning_effort=reasoning_effort,timeout=timeout,**settings)
                    attempt_path = output/'generation_attempts'/f'{key}-{invocation_id}-{attempt}.json'
                    _write_new(attempt_path, {'request_id':key, 'request':req, 'raw_response':raw,
                        'metadata':meta, 'duration_seconds':time.monotonic()-start,
                        'received_at':datetime.now(timezone.utc).isoformat(),
                        'invocation_id':invocation_id, 'attempt':attempt})
                    plan=_validate(raw); pid='generated_'+key[:24]
                    _write(path,{'request':req,'request_id':key,'raw_response':raw,'plan_as_stored':plan,'lesson_plan_id':pid,
                        'ai_model_preset_id':model,'review_subject_ids':[req['subject']],'metadata':meta,
                        'duration_seconds':time.monotonic()-start,'attempts':attempt})
                    return True
                except Exception as exc:
                    if attempt_path is None and hasattr(exc, 'response_metadata'):
                        attempt_path = output/'generation_attempts'/f'{key}-{invocation_id}-{attempt}.json'
                        _write_new(attempt_path, {'request_id':key, 'request':req,
                            'raw_response':getattr(exc, 'raw_response', None),
                            'metadata':exc.response_metadata, 'duration_seconds':time.monotonic()-start,
                            'received_at':datetime.now(timezone.utc).isoformat(),
                            'invocation_id':invocation_id, 'attempt':attempt})
                    msg=str(exc)
                    for k,v in os.environ.items():
                        if any(x in k.upper() for x in ('KEY','TOKEN','SECRET','PASSWORD')) and len(v)>8: msg=msg.replace(v,'[REDACTED]')
                    _write_new(output/'generation_errors'/f'{key}-{invocation_id}-{attempt}.json',
                        {'request_id':key, 'invocation_id':invocation_id, 'attempt':attempt,
                         'attempt_file':str(attempt_path.relative_to(output)) if attempt_path and attempt_path.exists() else None,
                         'type':type(exc).__name__,'message':msg[:1000]})
                    transient=getattr(exc,'transient',False) or getattr(exc,'status_code',None) in (408,429,500,502,503,504) or type(exc).__name__ in ('APIConnectionError','APITimeoutError','Timeout','RateLimitError')
                    if not transient or attempt==3: return False
                    time.sleep(attempt)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ok in pool.map(call,todo[:max_new_calls]):
                if ok is not None: counts['completed' if ok else 'failed']+=1
        counts['calls_started'] = attempts_started
        counts['pending']-=counts['completed']
        counts['complete'] = counts['pending'] == 0 and counts['failed'] == 0
        return counts
    if dry_run: return execute()
    with _lock(output): return execute()
