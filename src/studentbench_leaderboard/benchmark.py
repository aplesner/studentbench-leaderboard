"""Prepare and verify a self-contained, immutable benchmark bundle."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from .data import Dataset, digest
from .pretest_context import PretestContextBank

DEFAULT_STUDY = Path('results/ranking-study-20260910')
BANK_SHA = '70bc1f1d4d85b93d6f6b23f035a49ac912d4aa5f29787a9510f68072d0a7f336'

def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def _safe_tree(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError(f'Symlinks are not permitted in benchmark inputs or bundles: {root}')

def prepare_benchmark(ssot: Path, output: Path, study: Path | None = None, source_bank: Path | None = None) -> dict:
    ssot, output = Path(ssot).resolve(), Path(output).absolute()
    study = Path(study or DEFAULT_STUDY).absolute()
    if study.is_symlink(): raise ValueError('Study directory must not be a symlink')
    study = study.resolve()
    for name in ('manifest.json', 'selection.json', 'verification.json', 'config_inventory.json', 'calls', 'configs'):
        if (study / name).is_symlink(): raise ValueError(f'Symlink in frozen study: {name}')
    if output.resolve().is_relative_to(ssot):
        raise ValueError('Benchmark output must be outside the SSOT')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Benchmark output must be empty: {output}')
    data = Dataset(ssot, 'gre_quant'); data.verify_manifest()
    manifest = json.loads((study / 'manifest.json').read_text())
    selection = json.loads((study / 'selection.json').read_text())
    verification = json.loads((study / 'verification.json').read_text())
    if (manifest['dataset_sha256'] != data.fingerprint or selection['variant'] != 'enhanced_terra_high'
        or not manifest.get('selection_locked') or manifest.get('selection_sha256') != digest(selection)
        or verification.get('status') != 'passed' or verification.get('selection_sha256') != digest(selection)):
        raise ValueError('Frozen study selection or dataset verification does not match')
    subjects = manifest['splits']['test']
    if len(subjects) != 83 or len(set(subjects)) != 83:
        raise ValueError('Frozen test split must contain 83 unique subjects')
    config = study / 'configs' / 'enhanced_terra_high'
    _safe_tree(config)
    inventory = json.loads((study / 'config_inventory.json').read_text())
    if selection.get('config_inventory_sha256') != digest(inventory):
        raise ValueError('Frozen judge inventory differs from selection lock')
    expected_config = {k:v for k,v in inventory.items() if k.startswith('configs/enhanced_terra_high/')}
    actual_config = {str(path.relative_to(study)):digest(path.read_text())
                     for path in config.rglob('*') if path.is_file()}
    if not expected_config or actual_config != expected_config:
        raise ValueError('Frozen selected judge configuration inventory is incomplete or changed')
    bank_path = Path(source_bank) if source_bank else config / 'question_bank.json'
    if bank_path.is_symlink(): raise ValueError('Source bank must not be a symlink')
    bank = PretestContextBank.load(ssot, source_bank=bank_path, expected_source_sha256=BANK_SHA)
    sessions = {k:v for k,v in data.sessions.items() if v['review_subject_id'] in subjects}
    contexts = {}
    for subject in subjects:
        gid = min(k for k,v in sessions.items() if v['review_subject_id'] == subject)
        contexts[subject] = {'pretest': bank.enrich_pretest(data.pretests[subject]), 'rubric': data.rubric(gid)}
    plans = {k:v for k,v in data.plans.items() if set(v['review_subject_ids']) & set(subjects)}
    pairs = {p['pair_id']:p for p in manifest['pair_catalog']['test']}
    expected_calls = {name for name in verification.get('call_sha256', {})
                      if name.startswith('calls/enhanced_terra_high--')
                      and name.rsplit('/', 1)[-1].split('--')[1] in pairs}
    actual_calls = {str(path.relative_to(study)) for path in (study / 'calls').glob('enhanced_terra_high--*.json')
                    if path.name.split('--')[1] in pairs}
    if len(expected_calls) != 293 or actual_calls != expected_calls:
        raise ValueError('Frozen selected test call inventory is incomplete or changed')
    records = []
    for path in sorted((study / 'calls').glob('enhanced_terra_high--*.json')):
        if path.is_symlink(): raise ValueError(f'Symlink in study calls: {path}')
        record = json.loads(path.read_text())
        if record.get('split') != 'test': continue
        expected = verification.get('call_sha256', {}).get(str(path.relative_to(study)))
        if expected != _hash(path): raise ValueError(f'Frozen call checksum mismatch: {path}')
        pair = pairs.get(record.get('pair_id'))
        if not pair or record.get('subject') != pair['subject'] or record.get('plan_ids') != pair['plan_ids']:
            raise ValueError(f'Frozen call does not match pair catalog: {path}')
        records.append({**record, 'pair': pair, 'source_sha256': expected})
    prompt_path = ssot / '5_prompts' / 'gre.quant.plan.lean.v3.txt'
    release = json.loads((ssot / 'release_manifest.json').read_text())
    if next((x['sha256'] for x in release['files'] if x['path'] == '5_prompts/gre.quant.plan.lean.v3.txt'), None) != _hash(prompt_path):
        raise ValueError('Generation prompt differs from SSOT release manifest')
    payload = {'instrument':'gre_quant', 'source_dataset_sha256':data.fingerprint,
      'contexts':contexts, 'plans':plans, 'models':sorted({p['ai_model_preset_id'] for p in plans.values()}),
      'sessions':sessions, 'questions':data.questions, 'scales':data.scales,
      'rubrics':{gid:data.rubric(gid) for gid in sessions},
      'baseline_records':records, 'generation_prompt':prompt_path.read_text(),
      'provenance':{'study_selection_sha256':digest(selection), 'study_verification':'passed: read-only request reconstruction and response reparse', 'historical_transport':'native LiteLLM OpenAI adapter', 'bank':bank.provenance,
          'generation_prompt_sha256':_hash(prompt_path), 'renderer':'studentbench.raw-pretest.v1',
          'postprocessing':'studentbench.options-normalization.v1', 'historical_pipeline_equivalence':False,
          'treatment_minutes':60, 'duration_evidence':'All 795 released baseline plans allocate 60 minutes.'}}
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config, output / 'judge', symlinks=False)
    _write(output / 'benchmark.json', payload)
    files = {str(p.relative_to(output)): _hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    bundle_manifest = {'schema_version':1, 'files':files, 'fingerprint':digest(files)}
    _write(output / 'manifest.json', bundle_manifest)
    return {'output':str(output), 'fingerprint':bundle_manifest['fingerprint'], 'subjects':len(subjects),
            'plans':len(plans), 'baseline_records':len(records)}

class Benchmark:
    @classmethod
    def load(cls, path: Path):
        path = Path(path).absolute(); _safe_tree(path); path = path.resolve()
        manifest = json.loads((path / 'manifest.json').read_text())
        files = manifest.get('files', {})
        if manifest.get('schema_version') != 1 or digest(files) != manifest.get('fingerprint'):
            raise ValueError('Invalid benchmark manifest')
        actual = {str(p.relative_to(path)): _hash(p) for p in path.rglob('*') if p.is_file() and p != path / 'manifest.json'}
        if actual != files: raise ValueError('Benchmark file checksum mismatch')
        data = json.loads((path / 'benchmark.json').read_text())
        obj = cls(); obj.path = path; obj.fingerprint = manifest['fingerprint']
        for name in ('instrument','contexts','plans','models','sessions','questions','scales','baseline_records','generation_prompt'):
            setattr(obj, name, data[name])
        obj.rubrics = data.get('rubrics', {gid:obj.contexts[s['review_subject_id']]['rubric'] for gid,s in obj.sessions.items()})
        if not isinstance(obj.contexts, dict) or not obj.contexts or not isinstance(obj.plans, dict):
            raise ValueError('Invalid benchmark contexts or plans')
        for subject, context in obj.contexts.items():
            if not subject or not isinstance(context.get('pretest'), list) or not context['pretest'] or not isinstance(context.get('rubric'), list) or not context['rubric']:
                raise ValueError(f'Invalid benchmark context: {subject}')
            for criterion in context['rubric']:
                if criterion.get('scale_id') not in obj.scales or not criterion.get('id'):
                    raise ValueError(f'Invalid benchmark rubric: {subject}')
        for pid, plan in obj.plans.items():
            if (plan.get('lesson_plan_id') != pid or not plan.get('ai_model_preset_id') or
                not isinstance(plan.get('plan_as_stored'), dict) or not plan.get('review_subject_ids') or
                not set(plan['review_subject_ids']) & set(obj.contexts)):
                raise ValueError(f'Invalid benchmark plan: {pid}')
        if sorted(set(obj.models)) != sorted({p['ai_model_preset_id'] for p in obj.plans.values()}):
            raise ValueError('Benchmark model universe does not match plans')
        for gid, session in obj.sessions.items():
            subject = session.get('review_subject_id')
            if subject not in obj.contexts or gid not in obj.rubrics:
                raise ValueError(f'Invalid benchmark session: {gid}')
            for slot in ('a', 'b'):
                pid=session.get(f'lesson_plan_{slot}_id')
                if pid not in obj.plans or subject not in obj.plans[pid]['review_subject_ids']:
                    raise ValueError(f'Invalid benchmark session plan: {gid}')
        obj.source_dataset_sha256 = data.get('source_dataset_sha256', obj.fingerprint)
        obj.provenance = data.get('provenance', {})
        obj.judge_dir = path / 'judge'; obj.judge_id = 'judge_5'
        return obj
