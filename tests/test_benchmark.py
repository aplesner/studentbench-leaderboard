import json
import shutil
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from studentbench_leaderboard.benchmark import Benchmark, _hash, _write
from studentbench_leaderboard.data import digest


def make_benchmark(root: Path):
    """Build a portable synthetic bundle with two subjects and matching plans."""
    root.mkdir(parents=True, exist_ok=True)
    judge = Path(__file__).parents[1] / 'src/studentbench_leaderboard/judge_configs'
    shutil.copytree(judge, root / 'judge')
    rubric = [{'id':'q01','question':'Is the plan useful?', 'scale_id':'lesson_plan_quality_default', 'rating_labels':{str(i):str(i) for i in range(1,6)}}]
    contexts={}; plans={}; sessions={}
    for i in (1,2):
        subject=f's{i}'
        pretest=[{'presented_position':'1','question_id':'Q1','question_text':'1+1?', 'options_json':'{}', 'correct_answer':'2','student_answer':'1','is_correct':'false','skill':'secret label','subskill':'secret', 'difficulty':'secret','format':'SPR','time_spent_seconds':'20','answer_changes':'0'}]
        contexts[subject]={'pretest':pretest,'rubric':rubric}
        for m in ('baseline_a','baseline_b'):
            pid=f'{subject}_{m}'
            plans[pid]={'lesson_plan_id':pid,'ai_model_preset_id':m,'review_subject_ids':[subject], 'plan_as_stored':{'summary':'baseline','concepts':[]}}
        sessions[f'g{i}']={'review_subject_id':subject,'lesson_plan_a_id':f'{subject}_baseline_a','lesson_plan_b_id':f'{subject}_baseline_b'}
    payload={'instrument':'gre_quant','source_dataset_sha256':'fixture','contexts':contexts,'plans':plans,'models':['baseline_a','baseline_b'], 'sessions':sessions,'questions':{},'scales':{'lesson_plan_quality_default':{str(i):str(i) for i in range(1,6)}},'baseline_records':[], 'generation_prompt':'Prepare a {treatment_minutes}-minute plan.', 'provenance':{}}
    _write(root / 'benchmark.json',payload)
    files={str(p.relative_to(root)):_hash(p) for p in root.rglob('*') if p.is_file()}
    _write(root / 'manifest.json',{'schema_version':1,'files':files,'fingerprint':digest(files)})
    return Benchmark.load(root)

class BenchmarkTests(unittest.TestCase):
    def test_bundle_integrity_and_catalog(self):
        from studentbench_leaderboard.plan_catalog import PlanCatalog
        with TemporaryDirectory() as temp:
            b=make_benchmark(Path(temp)/'bundle')
            self.assertEqual(len(b.contexts), 2)
            self.assertEqual(len(PlanCatalog(b).plans), 4)
            (b.path/'benchmark.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'): Benchmark.load(b.path)

    def test_bundle_rejects_symlink(self):
        with TemporaryDirectory() as temp:
            b=make_benchmark(Path(temp)/'bundle')
            (b.path/'escape').symlink_to(temp)
            with self.assertRaisesRegex(ValueError, 'Symlinks'): Benchmark.load(b.path)

    def test_prepare_rejects_missing_frozen_calls(self):
        from studentbench_leaderboard.benchmark import prepare_benchmark
        repo=Path(__file__).parents[1]
        source=repo/'results/ranking-study-20260910'
        ssot=repo/'studentbench_data_ssot'
        if not (source/'verification.json').exists() or not (ssot/'release_manifest.json').exists():
            self.skipTest('Frozen local study inputs are not installed')
        with TemporaryDirectory() as temp:
            root=Path(temp); study=root/'study'; study.mkdir(); (study/'calls').mkdir()
            for name in ('manifest.json','selection.json','verification.json','config_inventory.json'):
                shutil.copy(source/name,study/name)
            (study/'configs').mkdir()
            shutil.copytree(source/'configs/enhanced_terra_high',study/'configs/enhanced_terra_high')
            with self.assertRaisesRegex(ValueError,'call inventory'):
                prepare_benchmark(ssot,root/'bundle',study=study)

    def test_bundle_root_symlink_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); make_benchmark(root/'bundle')
            (root/'alias').symlink_to(root/'bundle',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Symlinks'): Benchmark.load(root/'alias')
