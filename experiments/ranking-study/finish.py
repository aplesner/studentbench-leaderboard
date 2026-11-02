"""Complete the frozen validation-to-test protocol; resumable and bounded."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from studentbench_leaderboard.data import Dataset,digest
from studentbench_leaderboard.study import call_path,run_jobs,write_json,freeze
from studentbench_leaderboard.study_analysis import analyze_round
from studentbench_leaderboard.human_analysis import run_analysis
from studentbench_leaderboard.study_report import render_report

STUDY=Path('results/ranking-study-20260910')

def load(name):return json.loads((STUDY/name).read_text())
def event(name,**fields):
    print(json.dumps({'event':name,**fields}),flush=True)
    write_json(STUDY/'progress.json',{'event':name,'at':datetime.now(timezone.utc).isoformat(),**fields})

def wait_for(jobs,label):
    while True:
        missing=[j for j in jobs if not call_path(STUDY,j).exists()]
        if not missing:return
        event('waiting',phase=label,complete=len(jobs)-len(missing),total=len(jobs))
        time.sleep(30)

def unique(jobs):return list({(j['variant'],j['pair_id'],j.get('draw',0)):j for j in jobs}.values())
def choose(analysis,eligible):
    candidates=[j for j in analysis['judges'] if j['variant'] in eligible and
        j['metrics']['kendall_tau_b'] is not None and j['coverage']['criterion_coverage']>=.90 and
        j['coverage']['missing_calls']==0]
    if not candidates:raise ValueError('No fully executed eligible ranking meets the fixed coverage threshold')
    return min(candidates,key=lambda j:(-j['metrics']['kendall_tau_b'],j['metrics']['probability_mae'],j['estimated_usd'],j['variant']))

def main():
    manifest=load('manifest.json')
    # Explicit selection rule is written before any new validation analysis.
    rule={'primary':'maximum Kendall tau-b','tie_breakers':['minimum fitted probability MAE','minimum observed token cost','variant name'],
          'minimum_criterion_coverage':.90,'round1_eligible':'ten original single-call variants',
          'round2_eligible':['round1 selected single','enhanced_luna_high','enhanced_terra_high','enhanced_luna_high_balanced'],
          'test':'one frozen selection;155reviewedpairs with legacy_luna_low control;selected judge also all293availablepairs',
          'test_tuning':False}
    if (STUDY/'selection_rule.json').exists():
        if load('selection_rule.json')!=rule:raise ValueError('Selection rule changed')
    else:write_json(STUDY/'selection_rule.json',rule)
    jobs1=load('round1_jobs.json');wait_for(jobs1,'round1')
    event('analyzing',phase='round1')
    a1=analyze_round(STUDY,'round1',STUDY/'round1_jobs.json',bootstrap=300,density=True)
    winner1=choose(a1,{j['variant'] for j in jobs1})
    write_json(STUDY/'round1_selection.json',{'variant':winner1['variant'],'metrics':winner1['metrics'],'rule':rule})
    event('round1_selected',variant=winner1['variant'],metrics=winner1['metrics'])
    render_report(STUDY)
    controls=[{'variant':winner1['variant'],'pair_id':p,'draw':0} for p in manifest['round2_validation_pairs']]
    write_json(STUDY/'round2_control_jobs.json',controls)
    if run_jobs(STUDY,controls,workers=32)['failed']:raise RuntimeError('Round2 control calls failed; rerun to resume')
    jobs2=unique(controls+load('round2_enhanced_jobs.json')+load('round2_balanced_jobs.json'))
    write_json(STUDY/'round2_jobs.json',jobs2);wait_for(jobs2,'round2')
    event('analyzing',phase='round2')
    analyze_round(STUDY,'input_ablation',STUDY/'input_ablation_jobs.json',bootstrap=300)
    a2=analyze_round(STUDY,'round2',STUDY/'round2_jobs.json',bootstrap=500,density=True)
    eligible={winner1['variant'],'enhanced_luna_high','enhanced_terra_high','enhanced_luna_high_balanced'}
    winner2=choose(a2,eligible)
    selection={'variant':winner2['variant'],'members':[list(member) for member in winner2['members']],'aggregation':winner2['aggregation'],
               'validation_metrics':winner2['metrics'],'validation_jobs_sha256':a2['jobs_sha256'],
               'rule':rule,'config_inventory_sha256':digest(load('config_inventory.json'))}
    if (STUDY/'selection.json').exists():
        if load('selection.json')!=selection:raise ValueError('Frozen test selection changed')
    else:write_json(STUDY/'selection.json',selection)
    manifest=load('manifest.json')
    if manifest.get('selection_locked') and manifest.get('selection_sha256')!=digest(selection):raise ValueError('Selection lock mismatch')
    manifest.update(selection_locked=True,selection_sha256=digest(selection),selection=selection)
    write_json(STUDY/'manifest.json',manifest)
    event('selection_locked',variant=winner2['variant'],metrics=winner2['metrics'])
    render_report(STUDY)
    # No test verdicts contribute to tuning, metrics, or selection above this line.
    reviewed=[p['pair_id'] for p in manifest['pair_catalog']['test'] if p['human_reviewed']]
    all_test=[p['pair_id'] for p in manifest['pair_catalog']['test']]
    selected_jobs=lambda ids:[{'variant':v,'pair_id':p,'draw':d} for p in ids for v,d in selection['members']]
    test_jobs=unique(selected_jobs(reviewed)+[{'variant':'legacy_luna_low','pair_id':p,'draw':0} for p in reviewed])
    dense_jobs=selected_jobs(all_test)
    write_json(STUDY/'test_jobs.json',test_jobs);write_json(STUDY/'dense_jobs.json',dense_jobs)
    all_jobs=unique(test_jobs+dense_jobs);write_json(STUDY/'final_execution_jobs.json',all_jobs)
    if run_jobs(STUDY,all_jobs,workers=32)['failed']:raise RuntimeError('Test calls failed; rerun to resume frozen selection')
    event('analyzing',phase='test_and_density')
    test=analyze_round(STUDY,'test',STUDY/'test_jobs.json',bootstrap=500,density=True)
    dense=analyze_round(STUDY,'dense',STUDY/'dense_jobs.json',bootstrap=500,density=True)
    data=Dataset(Path('studentbench_data_ssot'),manifest['instrument'])
    human=run_analysis(data,manifest['splits']['test'],STUDY/'human',seed=manifest['seed'],repeats=200,bootstrap=500,split_name='test')
    calls=[json.loads(p.read_text()) for p in (STUDY/'calls').glob('*.json')]
    summary={'status':'complete','selected_judge':selection,'new_completed_calls':len(calls),
             'estimated_usd':sum(c.get('estimated_usd',0) for c in calls),
             'tokens':{key:sum(c.get('usage',{}).get(key,0) or 0 for c in calls) for key in ['prompt_tokens','completion_tokens','total_tokens']},
             'test_metrics':next(j['metrics'] for j in test['judges'] if j['variant']==selection['variant']),
             'dense_metrics':next(j['metrics'] for j in dense['judges'] if j['variant']==selection['variant']),
             'test_human_consistency':human['reviewer_split_half']['summary'],
             'limitations':['GRE Quant only;12authoringpresets','83heldoutsubjects; sourcefigures unrendered','RidgeBT andequalrubricweights areexplicit policychoices','Subject bootstrap does not fully capturecrossedtutor dependence','Density experiments condition on existingplanavailability;nohumanlabels fabricated fornewedges']}
    write_json(STUDY/'analysis.json',summary)
    render_report(STUDY)
    event('complete',calls=len(calls),estimated_usd=summary['estimated_usd'],test_metrics=summary['test_metrics'],dense_metrics=summary['dense_metrics'])

if __name__=='__main__':
    import fcntl
    with (STUDY/'controller.lock').open('a') as controller_lock:
        try:fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A study controller is already running')
        main()
