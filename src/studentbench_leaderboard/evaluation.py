"""Resumable, student-matched candidate evaluation against a frozen benchmark."""
from __future__ import annotations

import csv
import fcntl
import html
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .comparison import ConfiguredJudge, aggregate
from .data import digest
from .generation import generation_coverage, load_generation
from .judge import Task
from .judge_config import SOFT_STRATEGIES, load_definition
from .ranking import aggregate_pairs, fit_bradley_terry
from .structured_judge import parse_judge_response


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def _redact(error):
    message = str(error)
    for key, value in os.environ.items():
        if any(x in key.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) and len(value) >= 8:
            message = message.replace(value, '[REDACTED]')
    return {'type': type(error).__name__, 'message': message[:1000]}


def _family(key):
    match = re.search(r'_q(\d{2})(?:_|$)', key)
    return 'q' + match.group(1) if match else key


def _rank(rows, models, bootstrap, seed):
    pairs = aggregate_pairs(rows)
    fit = fit_bradley_terry(pairs, models=models)
    by_subject = Counter(row['subject'] for row in pairs)
    scale = len(pairs) / len(by_subject) if by_subject else 1
    weighted = [{**row, 'weight': scale / by_subject[row['subject']]} for row in pairs]
    sensitivity = fit_bradley_terry(weighted, models=models)
    subjects = sorted(by_subject)
    values = defaultdict(list)
    valid = 0
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap if subjects else 0):
        counts = Counter(rng.choice(subjects, size=len(subjects), replace=True).tolist())
        sample = [{**row, 'weight': counts[row['subject']]} for row in pairs if counts[row['subject']]]
        result = fit_bradley_terry(sample, models=models)
        if result['connected'] and result['converged']:
            valid += 1
            for row in result['leaderboard']:
                values[row['model']].append(row['rank'])
    intervals = {}
    for model in models:
        ranks = values[model]
        intervals[model] = {'low': float(np.quantile(ranks, .025)) if ranks else None,
                            'high': float(np.quantile(ranks, .975)) if ranks else None,
                            'n_valid': len(ranks)}
    return {'ranking': fit, 'equal_student_sensitivity': sensitivity,
            'rank_intervals': intervals, 'bootstrap': {'requested': bootstrap, 'valid': valid,
            'seed': seed, 'method': 'subject_cluster_percentile',
            'warning': 'Intervals use connected, converged resamples only.'}}


def _report(output, result):
    _write(output / 'evaluation.json', result)
    with (output / 'leaderboard.csv').open('w', newline='', encoding='utf-8') as f:
        columns = ['model', 'rank', 'score', 'rank_low', 'rank_high', 'n_comparisons', 'n_subjects']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in result['ranking']['leaderboard']:
            interval = result['rank_intervals'][row['model']]
            writer.writerow({**{k: row[k] for k in columns if k in row},
                             'rank_low': interval['low'], 'rank_high': interval['high']})
    rows = ''.join('<tr>' + ''.join('<td>' + html.escape(str(value if value is not None else '—')) + '</td>'
                      for value in (row['model'], row['rank'], round(row['score'], 4),
                                    result['rank_intervals'][row['model']]['low'],
                                    result['rank_intervals'][row['model']]['high'], row['n_comparisons'])) + '</tr>'
                   for row in result['ranking']['leaderboard'])
    details = html.escape(json.dumps({'coverage': result['coverage'], 'per_rubric': result['per_rubric'],
                                      'rubric_coverage': result['rubric_coverage'],
                                      'equal_student_sensitivity': result['equal_student_sensitivity'],
                                      'pairwise_probabilities': result['ranking']['pairwise_probabilities'],
                                      'baseline_import': result['baseline_import']}, indent=2))
    (output / 'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>StudentBench evaluation</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:3em auto;padding:0 1em}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #bbb}pre{white-space:pre-wrap}</style></head><body>'
        '<h1>AI model leaderboard</h1><p>Historical human reference judgments are separate; candidate plans have no human grades. '
        'Missing and abstained judgments are excluded, not ties. Rank intervals condition on connected bootstrap samples.</p>'
        '<table><thead><tr><th>Model</th><th>Rank</th><th>Score</th><th>Rank low</th><th>Rank high</th><th>Comparisons</th></tr></thead><tbody>'
        + rows + '</tbody></table><h2>Coverage and sensitivity</h2><pre>' + details + '</pre></body></html>', encoding='utf-8')


def _baseline(record, benchmark, plans, judge, definition, routes):
    pair = record['pair']
    subject, ids = pair['subject'], pair['plan_ids']
    if len(ids) != 2 or subject not in benchmark.contexts or any(
            p not in benchmark.plans or subject not in plans[p]['review_subject_ids'] for p in ids):
        raise ValueError('baseline pair/context mismatch')
    ctx = benchmark.contexts[subject]
    rubric = getattr(benchmark, 'rubrics', {}).get(record.get('grading_id'), ctx['rubric'])
    task = Task(benchmark.instrument, subject, ids, [plans[p]['plan_as_stored'] for p in ids],
                ctx['pretest'], rubric, benchmark.fingerprint)
    source_fingerprint = getattr(benchmark, 'source_dataset_sha256', benchmark.fingerprint)
    if record['dataset_sha256'] != source_fingerprint or record['judge'] != judge.configuration():
        raise ValueError('baseline benchmark or judge mismatch')
    if len(definition.members) != 1 or definition.call_count != 1:
        raise ValueError('baseline import requires one-call judge')
    route = routes[0]
    # Historical study calls are checksum-verified by the bundle preparation step,
    # which records the known native LiteLLM OpenAI adapter in its source ledger.
    import hashlib
    native_base = hashlib.sha256(b'https://api.openai.com/v1').hexdigest()
    model = definition.members[0].model
    native = (record.get('source_sha256') and route.get('provider') == 'openai'
              and route.get('transport') == 'direct' and route.get('effective_model') == model
              and route.get('api_base_sha256') == native_base
              and record.get('response_model') == model.removeprefix('openai/'))
    if record.get('route_provenance') != route and not native:
        raise ValueError('baseline route provenance missing or differs')
    swapped = record['presented_plan_ids'] == list(reversed(ids))
    ct = judge.call_task(task, swapped)
    if record['presented_plan_ids'] != ct.plan_ids or record['messages'] != ct.messages() or record['response_schema'] != ct.schema():
        raise ValueError('baseline request/context mismatch')
    expected = digest({'messages': ct.messages(), 'schema': ct.schema(), 'judge': judge.configuration(), 'draw': record.get('draw', 0)})
    if record['request_id'] != expected:
        raise ValueError('baseline request hash mismatch')
    ratings = parse_judge_response(record['raw_response'], ct)
    normalized = {k: {**v, 'score': -v['score'] if swapped and v['score'] is not None else v['score']} for k, v in ratings.items()}
    if normalized != record['normalized_ratings']:
        raise ValueError('baseline parser mismatch')
    rows = []
    for criterion in task.rubric:
        score = normalized[criterion['id']]['score']
        rows.append({'subject': subject, 'pair_id': digest([subject, *sorted(ids)]),
                     'model_a': plans[ids[0]]['ai_model_preset_id'], 'model_b': plans[ids[1]]['ai_model_preset_id'],
                     'criterion': criterion['id'] + '|' + criterion['scale_id'], 'criterion_family': _family(criterion['id']),
                     'value': None if score is None else 1.0 if score > 0 else 0.0 if score < 0 else .5})
    return rows


def _read_call(path, call):
    saved = json.loads(path.read_text())
    if saved.get('request') != call['request'] or saved.get('request_id') != call['id']:
        raise ValueError(f'Cached call request mismatch: {path}')
    if parse_judge_response(saved['raw_response'], call['task']) != saved.get('ratings'):
        raise ValueError(f'Cached call result mismatch: {path}')
    metadata = saved.get('metadata', {})
    if metadata.get('routing', call['request']['route']) != call['request']['route']:
        raise ValueError(f'Cached call route mismatch: {path}')
    return saved


def run_evaluation(benchmark, generation_dir: Path, client, *, output: Path | None = None,
                   dry_run=False, max_new_calls=1000, workers=1, timeout=180,
                   comparisons_per_student=None, seed=20261102, bootstrap=500, cache_dir: Path | None = None):
    """Freeze and execute comparisons; each validated provider call is independently resumable."""
    if (type(max_new_calls) is not int or max_new_calls < 0 or type(workers) is not int or workers < 1
            or type(bootstrap) is not int or bootstrap < 0 or timeout <= 0
            or (comparisons_per_student is not None and
                (type(comparisons_per_student) is not int or comparisons_per_student < 1))):
        raise ValueError('Invalid evaluation budget, workers, timeout, bootstrap or comparison limit')
    generation_dir = Path(generation_dir)
    output = Path(output) if output is not None else generation_dir / 'evaluation'
    # Never write into the SSOT even if it is supplied as an output directory.
    ssot = Path('studentbench_data_ssot').resolve()
    # Do not follow links in an existing output tree, including report and lock targets.
    # Check this before reading a saved schedule or opening any artifact.
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError('Evaluation output must be a directory, not a symlink or file')
    if output.exists() and any(path.is_symlink() for path in output.rglob('*')):
        raise ValueError('Evaluation output must not contain symlinks')
    target = output.resolve()
    bundle_path = getattr(benchmark, 'path', None)
    protected = [generation_dir.resolve()] + ([Path(bundle_path).resolve()] if bundle_path else [])
    if any(path == target or path.is_relative_to(target) for path in protected):
        raise ValueError('Evaluation output cannot replace the generation or benchmark directory')
    if bundle_path and target.is_relative_to(Path(bundle_path).resolve()):
        raise ValueError('Evaluation output cannot be inside the benchmark bundle')
    for name in ('calls', 'generations', 'generation_errors', 'generation_manifest.json', '.generation.lock'):
        artifact = (generation_dir / name).resolve()
        if target == artifact or target.is_relative_to(artifact):
            raise ValueError('Evaluation output cannot overlap generation artifacts')
    if target.is_relative_to(ssot):
        raise ValueError('Evaluation output cannot be inside the SSOT')
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
            raise ValueError('Shared cache must be a directory, not a symlink or file')
        if cache.exists() and any(path.is_symlink() for path in cache.rglob('*')):
            raise ValueError('Shared cache must not contain symlinks')
        cache_target = cache.resolve()
        if (cache_target == target or cache_target.is_relative_to(target) or target.is_relative_to(cache_target)
                or cache_target.is_relative_to(ssot) or ssot.is_relative_to(cache_target)):
            raise ValueError('Shared cache overlaps evaluation output or SSOT')
        if bundle_path and (cache_target.is_relative_to(Path(bundle_path).resolve()) or
                            Path(bundle_path).resolve().is_relative_to(cache_target)):
            raise ValueError('Shared cache overlaps benchmark bundle')
        gen_target = generation_dir.resolve()
        if gen_target == cache_target or gen_target.is_relative_to(cache_target):
            raise ValueError('Shared cache overlaps generation output')
        for name in ('calls', 'generations', 'generation_errors', 'generation_manifest.json', '.generation.lock'):
            artifact = (generation_dir / name).resolve()
            if cache_target == artifact or cache_target.is_relative_to(artifact):
                raise ValueError('Shared cache overlaps generation artifacts')
    generation_status = generation_coverage(generation_dir, benchmark)
    if generation_status['pending_subjects']:
        return {'complete': False, 'dry_run': bool(dry_run), 'status': 'generation_incomplete',
                'generation_coverage': generation_status,
                'coverage': {'scheduled_comparisons': 0, 'scheduled_calls': 0},
                'message': 'Complete generation for the selected cohort before starting evaluation.'}
    generated = load_generation(generation_dir, benchmark)
    if not generated:
        raise ValueError('No completed candidate plans are available')
    plans = dict(benchmark.plans)
    for plan in generated:
        if plan['plan_id'] in plans:
            raise ValueError('Candidate plan ID collides with benchmark')
        plans[plan['plan_id']] = plan
    definition = load_definition(benchmark.judge_id, benchmark.judge_dir)
    judge = ConfiguredJudge(definition)
    routes = [client.configuration(m.model, reasoning_effort=m.reasoning_effort) for m in definition.members]
    models = sorted({p['ai_model_preset_id'] for p in plans.values()})
    schedule = []
    for candidate in sorted(generated, key=lambda p: p['plan_id']):
        for subject in sorted(candidate['review_subject_ids']):
            if subject not in benchmark.contexts:
                raise ValueError('Candidate references an unknown student context')
            candidates = sorted(pid for pid, plan in benchmark.plans.items()
                                if subject in plan['review_subject_ids'] and
                                plan['ai_model_preset_id'] != candidate['ai_model_preset_id'])
            rng = random.Random(int(digest([seed, subject, candidate['plan_id']]), 16))
            rng.shuffle(candidates)
            for baseline in candidates[:comparisons_per_student]:
                ids = [candidate['plan_id'], baseline]
                if rng.getrandbits(1):
                    ids.reverse()
                schedule.append({'subject': subject, 'plan_ids': ids})
    if not schedule:
        raise ValueError('No candidate plans have an available baseline for the same student')
    imported_rows, rejected = [], []
    imported = 0
    scheduled_pairs = {(entry['subject'], tuple(sorted(entry['plan_ids']))) for entry in schedule}
    for record in benchmark.baseline_records:
        try:
            imported_rows.extend(_baseline(record, benchmark, plans, judge, definition, routes))
            imported += 1
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({'request_id': record.get('request_id'), 'reason': str(exc)})
            pair = record.get('pair', {})
            subject, ids = pair.get('subject'), pair.get('plan_ids', [])
            if (subject not in benchmark.contexts or len(ids) != 2 or ids[0] == ids[1] or
                    any(p not in benchmark.plans or subject not in plans[p]['review_subject_ids'] for p in ids)):
                raise ValueError('Cannot schedule invalid historical baseline pair') from exc
            key = (subject, tuple(sorted(ids)))
            if key not in scheduled_pairs:
                ordered = list(ids)
                if int(digest([seed, subject, sorted(ids)]), 16) % 2:
                    ordered.reverse()
                schedule.append({'subject': subject, 'plan_ids': ordered, 'grading_id': record.get('grading_id')})
                scheduled_pairs.add(key)
    schedule.sort(key=lambda x: (x['subject'], x['plan_ids']))
    header = {'version': 1, 'benchmark': benchmark.fingerprint, 'judge': judge.configuration(),
              'routes': routes, 'seed': seed, 'comparisons_per_student': comparisons_per_student,
              'schedule': schedule}
    tasks = []
    for entry in schedule:
        ctx = benchmark.contexts[entry['subject']]
        task = Task(benchmark.instrument, entry['subject'], entry['plan_ids'],
                    [plans[p]['plan_as_stored'] for p in entry['plan_ids']], ctx['pretest'],
                    getattr(benchmark, 'rubrics', {}).get(entry.get('grading_id'), ctx['rubric']), benchmark.fingerprint)
        calls = []
        for index, (member_index, swapped) in enumerate(judge.schedule(task)):
            ct = judge.call_task(task, swapped)
            member = definition.members[member_index]
            request = {'messages': ct.messages(), 'schema': ct.schema(), 'judge': judge.configuration(),
                       'route': routes[member_index], 'member': member.configuration(), 'timeout': timeout,
                       'subject': task.subject, 'plan_ids': task.plan_ids, 'benchmark_sha256': benchmark.fingerprint, 'call_index': index}
            calls.append({'id': digest(request), 'task': ct, 'swapped': swapped,
                          'member': member, 'request': request, 'index': index})
        tasks.append((task, calls))
    total_calls = sum(len(calls) for _, calls in tasks)
    header['requests'] = [call['id'] for _, calls in tasks for call in calls]
    header['baseline_ledger'] = {'imported': imported, 'rejected': rejected}
    header['identity'] = digest(header)
    base_result = {'complete': False, 'dry_run': bool(dry_run), 'generation_coverage': generation_status,
                   'schedule_identity': header['identity'],
                   'coverage': {'scheduled_comparisons': len(tasks), 'scheduled_calls': total_calls},
                   'baseline_import': {'imported': imported, 'rejected': rejected},
                   'estimated_cost_usd': None, 'cost_note': 'Cost is unknown until provider usage is available.'}
    if dry_run:
        schedule_path = output / 'schedule.json'
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != header:
            raise ValueError('Evaluation schedule, judge, route, or inputs changed; use a new output directory')
        cached = 0
        shared_hits = 0
        for _, calls in tasks:
            for call in calls:
                path = output / 'calls' / (call['id'] + '.json')
                shared_path = cache / (call['id'] + '.json') if cache else None
                if path.exists():
                    _read_call(path, call)
                    cached += 1
                elif shared_path is not None and shared_path.exists():
                    _read_call(shared_path, call)
                    cached += 1
                    shared_hits += 1
        base_result['request_previews'] = [call['request'] for _, calls in tasks for call in calls]
        base_result['coverage'].update(completed_calls=cached, shared_cache_reused=shared_hits, missing_calls=total_calls-cached,
                                       new_calls_planned=min(max_new_calls, total_calls-cached))
        return base_result
    output.mkdir(parents=True, exist_ok=True)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for directory in [output] + ([cache] if cache is not None else []):
            lock_path = directory / '.lock'
            fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            lock = stack.enter_context(os.fdopen(fd, 'a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f'Evaluation output or shared cache is locked: {directory}') from exc
        schedule_path = output / 'schedule.json'
        if schedule_path.exists():
            if json.loads(schedule_path.read_text()) != header:
                raise ValueError('Evaluation schedule, judge, route, or inputs changed; use a new output directory')
        else:
            _write(schedule_path, header)
        completed = {}
        pending = []
        shared_hits = 0
        for task, calls in tasks:
            for call in calls:
                path = output / 'calls' / (call['id'] + '.json')
                shared_path = cache / (call['id'] + '.json') if cache else None
                if path.exists():
                    completed[call['id']] = _read_call(path, call)
                elif shared_path is not None and shared_path.exists():
                    saved = _read_call(shared_path, call)
                    _write(path, saved)
                    completed[call['id']] = saved
                    shared_hits += 1
                else:
                    pending.append(call)
        # Repeated identical call identities must still be separate draws.
        if len({call['id'] for _, calls in tasks for call in calls}) != total_calls:
            raise ValueError('Evaluation contains duplicate call identities')
        selected = pending[:max_new_calls]
        def perform(item):
            call, attempt = item
            try:
                member, ct = call['member'], call['task']
                raw, metadata = client.complete(member.model, ct.messages(), ct.schema(),
                                                reasoning_effort=member.reasoning_effort, timeout=timeout)
                if metadata.get('routing', call['request']['route']) != call['request']['route']:
                    raise ValueError('Provider response route differs from the requested route')
                ratings = parse_judge_response(raw, ct)
                record = {'request_id': call['id'], 'request': call['request'], 'raw_response': raw,
                          'ratings': ratings, 'metadata': metadata}
                if cache is not None:
                    shared_path = cache / (call['id'] + '.json')
                    if shared_path.exists():
                        # The cache lock prevents another writer; never replace an existing response.
                        record = _read_call(shared_path, call)
                    else:
                        _write(shared_path, record)
                _write(output / 'calls' / (call['id'] + '.json'), record)
                return call, record, False
            except Exception as exc:
                _write(output / 'failures' / (call['id'] + f'-{attempt}.json'),
                       {'request_id': call['id'], 'attempt': attempt, **_redact(exc)})
                transient = (getattr(exc, 'transient', False) or getattr(exc, 'status_code', None) in (408, 429, 500, 502, 503, 504) or
                             type(exc).__name__ in ('APIConnectionError', 'APITimeoutError', 'Timeout', 'TimeoutError', 'RateLimitError'))
                return call, None, transient
        attempts = 0
        round_items = [(call, 1) for call in selected]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while round_items and attempts < max_new_calls:
                round_items = round_items[:max_new_calls - attempts]
                attempts += len(round_items)
                results = list(pool.map(perform, round_items))
                next_round = []
                for (_, attempt), (call, record, transient) in zip(round_items, results):
                    if record is not None:
                        completed[call['id']] = record
                    elif transient and attempt < 3:
                        next_round.append((call, attempt + 1))
                round_items = next_round
        rows = []
        complete_comparisons = 0
        for task, calls in tasks:
            if any(call['id'] not in completed for call in calls):
                continue
            complete_comparisons += 1
            normalized_calls = []
            for call in calls:
                ratings = completed[call['id']]['ratings']
                normalized = {k: {**v, 'score': -v['score'] if call['swapped'] and v['score'] is not None else v['score']}
                              for k, v in ratings.items()}
                normalized_calls.append({'normalized_ratings': normalized})
            ratings, diagnostics = aggregate(task, normalized_calls, definition.strategy)
            for criterion in task.rubric:
                key = criterion['id']
                score = ratings[key]['score']
                value = (diagnostics[key]['preference_probability'] if definition.strategy in SOFT_STRATEGIES
                         else None if score is None else 1.0 if score > 0 else 0.0 if score < 0 else .5)
                rows.append({'subject': task.subject, 'pair_id': digest([task.subject, *sorted(task.plan_ids)]),
                             'model_a': plans[task.plan_ids[0]]['ai_model_preset_id'],
                             'model_b': plans[task.plan_ids[1]]['ai_model_preset_id'],
                             'criterion': key + '|' + criterion['scale_id'], 'criterion_family': _family(key),
                             'value': value})
        rows.extend(imported_rows)
        result = {**base_result, **_rank(rows, models, bootstrap, seed),
                  'complete': complete_comparisons == len(tasks),
                  'coverage': {'scheduled_comparisons': len(tasks), 'completed_comparisons': complete_comparisons,
                               'missing_comparisons': len(tasks) - complete_comparisons, 'scheduled_calls': total_calls,
                               'completed_calls': len(completed), 'missing_calls': total_calls - len(completed),
                               'new_calls_attempted': attempts, 'shared_cache_reused': shared_hits, 'failed_calls': sum(call['id'] not in completed for call in selected),
                               'requested_criteria': sum(len(task.rubric) for task, _ in tasks),
                               'scored_criteria': sum(r['value'] is not None for r in rows) - sum(r['value'] is not None for r in imported_rows),
                               'missing_criteria': sum(len(task.rubric) for task, calls in tasks if any(call['id'] not in completed for call in calls)),
                               'abstained_criteria': sum(r['value'] is None for r in rows) - sum(r['value'] is None for r in imported_rows)},
                  'per_rubric': {key: _rank([r for r in rows if r['criterion_family'] == key], models, 0, seed)['ranking']
                                 for key in sorted({r['criterion_family'] for r in rows})},
                  'rubric_coverage': {family: {
                      'scheduled': sum(_family(q['id']) == family for task, _ in tasks for q in task.rubric),
                      'scored': sum(r['criterion_family'] == family and r['value'] is not None for r in rows) - sum(r['criterion_family'] == family and r['value'] is not None for r in imported_rows),
                      'abstained': sum(r['criterion_family'] == family and r['value'] is None for r in rows) - sum(r['criterion_family'] == family and r['value'] is None for r in imported_rows),
                      'missing': sum(_family(q['id']) == family for task, calls in tasks if any(call['id'] not in completed for call in calls) for q in task.rubric)}
                      for family in sorted({_family(q['id']) for task, _ in tasks for q in task.rubric})},
                  'baseline_import': {'imported': imported, 'rejected': rejected},
                  'human_reference': {'status': 'separate', 'candidate_human_scores': None},
                  'report_paths': {name: str(output / file) for name, file in [('json', 'evaluation.json'), ('csv', 'leaderboard.csv'), ('html', 'report.html')]}}
        _report(output, result)
        return result
