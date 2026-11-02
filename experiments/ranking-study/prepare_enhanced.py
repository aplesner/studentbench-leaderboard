"""Freeze two audit-driven input improvements before any test evaluation."""
import json,shutil,hashlib
from pathlib import Path
from studentbench_leaderboard.study import write_json,freeze
from studentbench_leaderboard.pretest_context import PretestContextBank
study=Path('results/ranking-study-20260910');source=Path('experiments/ranking-study')
sha='70bc1f1d4d85b93d6f6b23f035a49ac912d4aa5f29787a9510f68072d0a7f336'
bank=study/'context/quant_application_bank.json'
if not bank.exists():
 bank.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy('../studentbench/data/questions_gre_quant.json',bank)
PretestContextBank.load(Path('studentbench_data_ssot'),source_bank=bank,expected_source_sha256=sha)
manifest=json.loads((study/'manifest.json').read_text())
variants=json.loads((study/'variants.json').read_text())
for base in ['luna_high','terra_high']:
 name='enhanced_'+base
 existing=next(v for v in variants if v['variant']==base)
 info={**existing,'variant':name,'label':f'{base}: verified pretest context + human display',
       'pretest_context':{'file':'question_bank.json','sha256':sha}}
 for parent in [source,study]:
  target=parent/'configs'/name
  if not target.exists():shutil.copytree(parent/'configs'/base,target)
  config=target/'judge_5.toml'
  text=config.read_text()
  if 'plan_projection' not in text:
   text=text.replace('response_style = "evidence_winner"','response_style = "evidence_winner"\nplan_projection = "human_display"')
   config.write_text(text)
  user=target/'prompts/user.txt'
  user.write_text('Use the verified pretest quantities and shared data below; any unavailable figure is explicitly marked.\n$payload\n')
  (target/'context_input.toml').write_text(f'source_bank = "question_bank.json"\nsource_sha256 = "{sha}"\nprojection = "human_display"\n')
  if parent==study:shutil.copy(bank,target/'question_bank.json')
 if not any(v['variant']==name for v in variants):variants.append(info)
write_json(study/'variants.json',variants);write_json(source/'variants.json',variants)
freeze(study)
protocol={'status':'frozen_before_test','source':'development audit, fixed prototypes independent of round1 validation scores',
 'variants':['enhanced_luna_high','enhanced_terra_high'],
 'paired_control':'Same model/effort as round1; pretest context restoration and hidden archival answer-field removal form one input-improvement bundle.',
 'round2':'Both fixed prototypes run all120validationpairs; top single from completed round1 advances to the same120pair cohort. Winner selected on Kendall tau-b, then probability MAE, then measured token cost.',
 'order_ablation':'Evaluate original and reversed order for fixed enhanced_luna_high on validation before locking test.',
 'bank_sha256':sha,'test_labels_used':False}
write_json(study/'round2_protocol.json',protocol)
jobs=[{'variant':v,'pair_id':p,'draw':0} for p in manifest['round2_validation_pairs'] for v in protocol['variants']]
write_json(study/'round2_enhanced_jobs.json',jobs)
print('Frozen enhanced prototype jobs',len(jobs))
balanced=[{'variant':'enhanced_luna_high','pair_id':p,'draw':1} for p in manifest['round2_validation_pairs']]
write_json(study/'round2_balanced_jobs.json',balanced)
shutil.copy(source/'derived_judges.toml',study/'derived_judges.toml')
protocol['balanced_prototype']='Fixed enhanced_luna_high exactly oneAB+oneBA, soft win-share average; scheduled before reading new validation metrics.'
write_json(study/'round2_protocol.json',protocol)
input_jobs=[{'variant':v,'pair_id':p,'draw':0} for p in manifest['round1_validation_pairs']
            for v in ['luna_high','terra_high','enhanced_luna_high','enhanced_terra_high']]
write_json(study/'input_ablation_jobs.json',input_jobs)
