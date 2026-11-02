from studentbench_leaderboard.data import Dataset,digest
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations
import random,json
D=Dataset(Path('studentbench_data_ssot'),'gre_quant')
old=json.loads(Path('results/quant-100-20260910/manifest.json').read_text())
dev=sorted({D.sessions[g]['review_subject_id'] for g in old['grading_ids']})
remaining=sorted(set(D.pretests)-set(dev),key=lambda s:digest(['ranking-study-v1',s]))
splits={'development':dev,'validation':remaining[:82],'test':remaining[82:]}
assert not set(splits['validation'])&set(splits['test'])
pairs=defaultdict(list); by_sub=defaultdict(set)
for gid,s in sorted(D.sessions.items()):
 pairs[(s['review_subject_id'],*sorted([s['lesson_plan_a_id'],s['lesson_plan_b_id']]))].append(gid)
for pid,p in D.plans.items():
 for s in p['review_subject_ids']: by_sub[s].add(pid)
by_subject=defaultdict(list)
for key,gids in pairs.items():
 s,a,b=key
 by_subject[s].append({'pair_id':digest(key)[:20],'subject':s,'plan_ids':[a,b],'grading_id':gids[0], 'human_grading_ids':gids,'model_a':D.plans[a]['ai_model_preset_id'],'model_b':D.plans[b]['ai_model_preset_id'],'human_reviewed':True})
all_pairs={}
for split,subjects in splits.items():
 entries=[]
 for s in subjects:
  gid=sorted(g for g,x in D.sessions.items() if x['review_subject_id']==s)[0]
  for a,b in combinations(sorted(by_sub[s]),2):
   key=(s,a,b)
   if key in pairs: entry=next(x for x in by_subject[s] if x['plan_ids']==[a,b])
   else: entry={'pair_id':digest(key)[:20],'subject':s,'plan_ids':[a,b],'grading_id':gid,'human_grading_ids':[],'model_a':D.plans[a]['ai_model_preset_id'],'model_b':D.plans[b]['ai_model_preset_id'],'human_reviewed':False}
   if entry['model_a']!=entry['model_b']: entries.append(entry)
 all_pairs[split]=entries
# Interleave subjects so sparse samples cover subjects; choose edges using only model exposure.
def sample(entries,n):
 chosen=[]; pool=list(entries); model_counts=Counter(); subject_counts=Counter()
 while pool and len(chosen)<n:
  x=min(pool,key=lambda e:(subject_counts[e['subject']],model_counts[e['model_a']]+model_counts[e['model_b']],digest([20260910,e['pair_id']])))
  chosen.append(x); pool.remove(x); subject_counts[x['subject']]+=1
  model_counts[x['model_a']]+=1; model_counts[x['model_b']]+=1
 return chosen
val=[e for e in all_pairs['validation'] if e['human_reviewed']]
test=[e for e in all_pairs['test'] if e['human_reviewed']]
manifest={'schema_version':1,'seed':20260910,'dataset_sha256':D.fingerprint,'instrument':'gre_quant','splits':splits,'pair_catalog':all_pairs,'round1_validation_pairs':[x['pair_id'] for x in sample(val,60)],'round2_validation_pairs':[x['pair_id'] for x in sample(val,120)],'test_pairs':[x['pair_id'] for x in sample(test,len(test))], 'protocol':{'primary_metric':'Kendall tau-b across 12 authoring model presets, fractional-tie ridge Bradley-Terry, equal pair weight and equal rubric-criterion weight','secondary':['Spearman','pairwise model-order agreement','top model overlap','rank MAE','probability MAE','subject bootstrap confidence intervals'],'selection':'Round1 screen on 60 validation pairs. Advance up to 3 single judge variants and fixed derived strategies to 120 validation pairs. Freeze selection before test evaluation. Test all available human-reviewed pairs with selected judge plus denser unreviewed comparisons.','holdout':'All 81 previously exposed subjects are development. Remaining 165 subjects hashed into 82 validation and 83 test; no shared plans or duplicate complete pretests.','composite':'Primary equal weight over q01-q08 criterion families, ignoring magnitude (win=1,tie=.5,loss=0), average repeated human labels before equal criterion averaging. Per-variant sensitivity retained; subjective composite is explicit policy choice.','density':'Nested seeded subsamples with 100 repeats; human reviewed pairs vs unreviewed augmentation reported separately; subject sample coverage also reported.','scope':'GRE Quant only; model presets include generation reasoning settings, kept separate.'}}
out=Path('results/ranking-study-20260910/manifest.json')
if out.exists():
 existing=json.loads(out.read_text())
 assert all(existing.get(k)==v for k,v in manifest.items()), 'Frozen protocol or split changed'
else: out.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'splits':{k:{'subjects':len(v),'human_pairs':sum(e['human_reviewed'] for e in all_pairs[k]),'all_pairs':len(all_pairs[k]),'models':len({m for e in all_pairs[k] for m in [e['model_a'],e['model_b']]})} for k,v in splits.items()}},indent=2))
variants=json.loads(Path('experiments/ranking-study/variants.json').read_text())
initial=[v['variant'] for v in variants if not v['variant'].startswith('enhanced_')]
jobs=[{'variant':v,'pair_id':p,'draw':0} for p in manifest['round1_validation_pairs'] for v in initial]
job_path=out.parent/'round1_jobs.json'
if job_path.exists(): assert json.loads(job_path.read_text())==jobs
else: job_path.write_text(json.dumps(jobs,indent=2)+'\n')
