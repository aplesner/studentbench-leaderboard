"""Resumable, explicitly scheduled ranking experiments. No calls during analysis."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
import tomllib

from .comparison import ConfiguredJudge
from .data import Dataset,digest
from .judge import LiteLLMJudge,make_task
from .judge_config import load_definition
from .structured_judge import parse_judge_response

DEFAULT_STUDY=Path('results/ranking-study-20260910')
PRICES={'gpt-5.6-luna':(.2,.02,1.2),'gpt-5.6-terra':(2,.2,12),
        'gpt-5.6-sol':(4,.4,20),'gpt-6-astra':(10,1,50)}

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temp.replace(path)

def estimated_cost(model,usage):
    a,c,b=PRICES[model.split('/')[-1]]
    prompt=usage.get('prompt_tokens',0); output=usage.get('completion_tokens',0)
    cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
    return ((prompt-cached)*a+cached*c+output*b)/1e6

def catalog(manifest):
    return {p['pair_id']:{**p,'split':split} for split,items in manifest['pair_catalog'].items() for p in items}

def freeze(study):
    original=Path('experiments/ranking-study')
    for name in ['configs','prompts']:
        target=study/name
        if not target.exists():shutil.copytree(original/name,target)
    if not (study/'variants.json').exists():shutil.copy(original/'variants.json',study/'variants.json')
    variants={v['variant']:v for v in json.loads((study/'variants.json').read_text())}
    for name,v in variants.items():
        definition=load_definition(v['judge_id'],study/'configs'/name)
        if (len(definition.members)!=1 or definition.members[0].model!=v['model'] or
                definition.members[0].reasoning_effort!=v['reasoning_effort'] or
                definition.response_style!=v['response_style']):
            raise ValueError(f'Variant metadata differs from its frozen configuration: {name}')
        if context:=v.get('pretest_context'):
            declared=tomllib.loads((study/'configs'/name/'context_input.toml').read_text())
            if (declared['source_bank']!=context['file'] or declared['source_sha256']!=context['sha256'] or
                    declared['projection']!=definition.plan_projection):
                raise ValueError(f'Pretest context differs from frozen input configuration: {name}')
    inventory={str(p.relative_to(study)):digest(p.read_text()) for p in (study/'configs').rglob('*') if p.is_file()}
    inventory_path=study/'config_inventory.json'
    if inventory_path.exists():
        previous=json.loads(inventory_path.read_text())
        if any(inventory.get(k)!=v for k,v in previous.items()):
            raise ValueError('An existing frozen judge configuration changed')
    write_json(inventory_path,inventory)
    return variants

def call_path(study,job):
    return study/'calls'/f"{job['variant']}--{job['pair_id']}--{job.get('draw',0)}.json"

def run_jobs(study,jobs,workers=20):
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP','True')
    import litellm
    litellm.suppress_debug_info=True
    logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    manifest=json.loads((study/'manifest.json').read_text())
    data=Dataset(Path('studentbench_data_ssot'),manifest['instrument'])
    if data.fingerprint!=manifest['dataset_sha256']:raise ValueError('SSOT fingerprint changed')
    variants=freeze(study); pairs=catalog(manifest)
    judges={name:ConfiguredJudge(load_definition(v['judge_id'],study/'configs'/name)) for name,v in variants.items()}
    context_banks={}
    needed_variants={job['variant'] for job in jobs}
    for name,v in variants.items():
        if name in needed_variants and (context:=v.get('pretest_context')):
            from .pretest_context import PretestContextBank
            source=(study/'configs'/name/context['file']).resolve()
            if not source.is_relative_to((study/'configs'/name).resolve()):
                raise ValueError('Pretest source must be inside its frozen configuration')
            context_banks[name]=PretestContextBank.load(data.root,source_bank=source,
                expected_source_sha256=context['sha256'])
    source_hash=digest({p.name:p.read_text() for p in [Path(__file__),Path(__file__).with_name('structured_judge.py'),Path(__file__).with_name('comparison.py'),Path(__file__).with_name('judge.py')]})
    lock=threading.Lock();counts={'completed':0,'reused':0,'failed':0,'estimated_usd':0.0};started=time.monotonic()
    limits={model:threading.Semaphore(min(workers,16 if 'astra' in model else 32)) for model in PRICES}
    def perform(job):
        path=call_path(study,job)
        entry=pairs[job['pair_id']];variant=variants[job['variant']];judge=judges[job['variant']]
        task=make_task(data,entry['grading_id'],entry['plan_ids'])
        if job['variant'] in context_banks:
            task=context_banks[job['variant']].enrich_task(task)
        # Same randomized order for all variants, independent of answers/authors.
        swapped=bool(int(digest([manifest['seed'],entry['pair_id']]),16)%2) ^ bool(job.get('draw',0)%2)
        ct=judge.call_task(task,swapped)
        request=digest({'messages':ct.messages(),'schema':ct.schema(),'judge':judge.configuration(),'draw':job.get('draw',0)})
        if path.exists():
            old=json.loads(path.read_text())
            if old['request_id']!=request:raise ValueError(f'Cached request changed: {path}')
            reparsed=parse_judge_response(old['raw_response'],ct)
            normalized={k:{**v,'score':-v['score'] if swapped and v['score'] is not None else v['score']} for k,v in reparsed.items()}
            if normalized!=old['normalized_ratings']:
                raise ValueError(f'Cached result disagrees with current response parser: {path}')
            if old['judge']!=judge.configuration():raise ValueError('Cached judge version mismatch')
            with lock:counts['reused']+=1
            return
        for attempt in range(1,4):
            t=time.monotonic()
            try:
                with limits[variant['model'].split('/')[-1]]:
                    raw,metadata=LiteLLMJudge(variant['model'],variant['reasoning_effort'],timeout=360).complete(ct)
                ratings=parse_judge_response(raw,ct)
                normalized={k:{**v,'score':-v['score'] if swapped and v['score'] is not None else v['score']} for k,v in ratings.items()}
                cost=estimated_cost(variant['model'],metadata.get('usage',{}))
                record={**job,'subject':task.subject,'plan_ids':task.plan_ids,'presented_plan_ids':ct.plan_ids,
                    'order':'BA' if swapped else 'AB','split':entry['split'],'human_reviewed':entry['human_reviewed'],
                    'model_a':entry['model_a'],'model_b':entry['model_b'],'grading_id':entry['grading_id'],
                    'criteria':task.metadata()['criteria'],'judge':judge.configuration(),'request_id':request,
                    'messages':ct.messages(),'response_schema':ct.schema(),'raw_response':raw,'ratings':ratings,
                    'normalized_ratings':normalized,'dataset_sha256':data.fingerprint,'runner_sha256':source_hash,
                    'created_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-t,
                    'attempts':attempt,'estimated_usd':cost,**metadata}
                if job['variant'] in context_banks:
                    record['pretest_context_provenance']=context_banks[job['variant']].provenance
                write_json(path,record)
                with lock:
                    counts['completed']+=1;counts['estimated_usd']+=cost
                    if counts['completed']%10==0:print(json.dumps({'event':'progress',**counts,'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
                return
            except Exception as exc:
                message=str(exc)
                for key,value in os.environ.items():
                    if any(x in key for x in ['KEY','TOKEN','SECRET','PASSWORD']) and len(value)>8:message=message.replace(value,'[REDACTED]')
                error={**job,'attempt':attempt,'type':type(exc).__name__,'status_code':getattr(exc,'status_code',None),'message':message[:1000]}
                write_json(study/'errors'/f"{path.stem}-{attempt}.json",error)
                transient=error['status_code'] in [408,429,500,502,503,504] or error['type'] in ['APIConnectionError','APITimeoutError','Timeout','RateLimitError']
                if transient and attempt<3:time.sleep(attempt*3);continue
                with lock:counts['failed']+=1;print(json.dumps({'event':'failed',**error}),flush=True)
                return
    print(json.dumps({'event':'start','jobs':len(jobs),'workers':workers}),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(perform,job) for job in jobs]
        for f in as_completed(futures):f.result()
    print(json.dumps({'event':'finished',**counts,'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
    return counts

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('jobs',type=Path);run.add_argument('--workers',type=int,default=20)
    run.add_argument('--max-new-calls',type=int,default=1000)
    args=parser.parse_args()
    jobs=json.loads(args.jobs.read_text())
    if not isinstance(jobs,list) or len({(j['variant'],j['pair_id'],j.get('draw',0)) for j in jobs})!=len(jobs):raise ValueError('Jobs must be a unique list')
    new=sum(not call_path(args.study,j).exists() for j in jobs)
    if new>args.max_new_calls:raise ValueError(f'{new} new calls exceeds configured limit')
    counts=run_jobs(args.study,jobs,args.workers)
    if counts['failed']:raise SystemExit(1)

if __name__=='__main__':main()
