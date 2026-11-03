#!/usr/bin/env python3
"""Plan or run a separate extension of the frozen first public release.

No model identifiers or prices are inferred. Supply a local JSON roster, a
verified catalog and proxy configuration. By default commands only estimate;
--execute is required for any inference. The extension has its own $1,000 ledger.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

from studentbench_leaderboard.benchmark import Benchmark
from studentbench_leaderboard.comparison import ConfiguredJudge
from studentbench_leaderboard.data import digest
from studentbench_leaderboard.evaluation import run_multi_evaluation
from studentbench_leaderboard.generation import run_generation, generation_coverage
from studentbench_leaderboard.judge_config import load_definition
from studentbench_leaderboard.llm_client import LLMClient

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('studentbench_release_driver', _HERE / 'run.py')
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_roster(config, existing_models):
    if not isinstance(config, dict) or set(config) != {'models'}:
        raise ValueError('Extension config must contain only models')
    models = config.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('Extension config requires a nonempty models list')
    ids, model_ids = set(), set(existing_models)
    active = []
    for entry in models:
        if not isinstance(entry, dict) or set(entry) - {'id', 'model', 'price_id', 'label', 'enabled', 'generation_settings', 'reasoning_effort'}:
            raise ValueError('Invalid extension model entry')
        for key in ('id', 'model', 'price_id'):
            if not isinstance(entry.get(key), str) or not entry[key].strip() or entry[key] != entry[key].strip():
                raise ValueError(f'Model {key} must be a nonempty string')
        if not all(c.isalnum() or c in '_-' for c in entry['id']) or entry['id'] in ids:
            raise ValueError('Model IDs must be unique safe directory names')
        if entry['model'] in model_ids:
            raise ValueError('Extension models must be distinct from existing model identities')
        if 'label' in entry and (not isinstance(entry['label'], str) or not entry['label'].strip()):
            raise ValueError('label must be a nonempty string')
        if type(entry.get('enabled', True)) is not bool:
            raise ValueError('enabled must be boolean')
        if not isinstance(entry.get('generation_settings', {}), dict):
            raise ValueError('generation_settings must be an object')
        ids.add(entry['id']); model_ids.add(entry['model'])
        if entry.get('enabled', True): active.append(entry)
    if not active: raise ValueError('At least one extension model must be enabled')
    return active


def price_map(catalog, entries, judge_model):
    available = release.prices_from_catalog(catalog)
    result = {}
    for entry in entries:
        if entry['price_id'] not in available:
            raise ValueError(f"Missing frozen catalog price for {entry['price_id']}")
        result[entry['model']] = available[entry['price_id']]
    if judge_model not in available: raise ValueError(f'Missing frozen catalog judge price for {judge_model}')
    result[judge_model] = available[judge_model]
    return result


def estimate_requests(requests, prices):
    total = Decimal(0)
    for req in requests:
        config = req.get('configuration', req.get('route'))
        model = config['requested_model']
        settings = config['settings']
        max_tokens = settings.get('max_tokens', settings.get('max_completion_tokens'))
        if model not in prices: raise ValueError(f'Missing frozen catalog price for {model}')
        total += release.maximum_cost(req['messages'], req['schema'], max_tokens, prices[model])
    return total


@contextmanager
def extension_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    fd = os.open(path / '.extension.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(file, fcntl.LOCK_UN)


def prepare_cache(source, target, expected_ids, inventory):
    """Copy the original cache once; never modify the source or replace an existing copy."""
    source, target = Path(source), Path(target)
    if target.exists():
        if target.is_symlink() or not target.is_dir(): raise ValueError('Invalid extension cache')
        for call_id in expected_ids:
            path = target / (call_id + '.json')
            if path.is_symlink() or not path.is_file() or sha(path) != inventory[call_id]:
                raise ValueError('Extension cache original judgment is missing or changed; restore it before continuing')
        return
    if source.is_symlink() or not source.is_dir(): raise ValueError('Original judgment cache unavailable')
    files = sorted(source.glob('*.json'))
    if any(p.is_symlink() for p in files): raise ValueError('Symlink in original judgment cache')
    if not set(expected_ids).issubset({p.stem for p in files}):
        raise ValueError('Original cache is incomplete; restore it before extending')
    tmp = Path(tempfile.mkdtemp(prefix='.cache-copy-', dir=target.parent))
    try:
        for path in files: shutil.copyfile(path, tmp / path.name)
        for call_id in expected_ids:
            if sha(tmp / (call_id + '.json')) != inventory[call_id]:
                raise ValueError('Original cache changed during copy')
        os.rename(tmp, target)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'evaluate', 'summary'])
    parser.add_argument('--config', type=Path, required=True, help='Local extension roster JSON')
    parser.add_argument('--extension-dir', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True, help='Frozen local pricing catalog JSON')
    parser.add_argument('--connection', type=Path, default=release.RUN / 'connection.toml')
    parser.add_argument('--models', nargs='+', help='Extension IDs to generate (default: all enabled)')
    parser.add_argument('--workers', type=release.worker_count, default=1)
    parser.add_argument('--max-new-calls', type=int, default=None)
    parser.add_argument('--execute', action='store_true', help='Allow inference after the estimate passes')
    args = parser.parse_args(argv)
    if args.max_new_calls is None:
        if args.execute: parser.error('--execute requires an explicit positive --max-new-calls')
        args.max_new_calls = 10000
    if args.max_new_calls < 0 or args.execute and args.max_new_calls < 1:
        parser.error('--execute requires a positive --max-new-calls')
    if args.command == 'evaluate' and args.models:
        parser.error('Joint evaluation requires all enabled extension models')
    v1 = release.RUN.resolve()
    extension = args.extension_dir.absolute()
    if extension.is_symlink(): raise ValueError('Extension output must not be a symlink')
    extension = extension.resolve()
    if extension == v1 or extension.is_relative_to(v1) or v1.is_relative_to(extension):
        raise ValueError('Extension must be separate from the original release')
    config = json.loads(args.config.read_text())
    original = json.loads((v1 / 'protocol.json').read_text())
    roster = [m for m in original['models'] if m.get('enabled', True)]
    active = validate_roster(config, [m['model'] for m in roster])
    selected_ids = set(args.models or [m['id'] for m in active])
    if selected_ids - {m['id'] for m in active}: raise ValueError('Unknown or disabled extension model ID')
    selected = [m for m in active if m['id'] in selected_ids]
    if len({m['id'] for m in roster} & {m['id'] for m in active}): raise ValueError('Extension IDs collide with original model IDs')
    schedule = json.loads((v1 / 'evaluation' / 'schedule.json').read_text())
    identity = schedule.get('identity')
    if identity != digest({k:v for k,v in schedule.items() if k != 'identity'}):
        raise ValueError('Original schedule integrity failure')
    if schedule.get('generation_models') != sorted(m['model'] for m in roster):
        raise ValueError('Original schedule differs from original active roster')
    if not schedule.get('include_new_pairs') or schedule.get('comparisons_per_student') is not None:
        raise ValueError('Original comparison policy is incompatible')
    timeout = release.judge_timeout(original)
    benchmark = Benchmark.load(release.ROOT / original['benchmark'])
    if benchmark.fingerprint != schedule['benchmark']: raise ValueError('Original benchmark fingerprint changed')
    baseline_models = {plan['ai_model_preset_id'] for plan in benchmark.plans.values()}
    if any(m['model'] in baseline_models for m in active): raise ValueError('Extension model duplicates a baseline')
    ssot = Path(os.environ.get('STUDENTBENCH_SSOT', str(release.ROOT / 'studentbench_data_ssot'))).resolve()
    if (extension.is_relative_to(benchmark.path) or benchmark.path.is_relative_to(extension)
            or extension.is_relative_to(ssot) or ssot.is_relative_to(extension)
            or any((parent / 'release_manifest.json').exists() for parent in (extension, *extension.parents))):
        raise ValueError('Extension output cannot overlap benchmark or SSOT')
    original_result = json.loads((v1 / 'evaluation' / 'evaluation.json').read_text())
    if not original_result.get('complete') or original_result.get('schedule_identity') != identity:
        raise ValueError('Original evaluation is incomplete or its schedule differs')
    for m in roster:
        coverage = generation_coverage(v1 / 'generations' / m['id'], benchmark)
        if coverage['selected_subjects'] != sorted(benchmark.contexts) or coverage['pending_subjects']:
            raise ValueError(f"Original generation is incomplete for {m['id']}")
        old_manifest = json.loads((v1 / 'generations' / m['id'] / 'generation_manifest.json').read_text())
        if old_manifest.get('model') != m['model']:
            raise ValueError(f"Original model identity changed for {m['id']}")
    definition = load_definition(benchmark.judge_id, benchmark.judge_dir)
    if len(definition.members) != 1: raise ValueError('Extension requires the original single-member judge')
    if ConfiguredJudge(definition).configuration() != schedule['judge']:
        raise ValueError('Original judge configuration changed')
    judge_model = definition.members[0].model
    prices = price_map(args.catalog, active, judge_model)
    catalog_hash = sha(args.catalog)
    ledger_path = v1 / 'spend-ledger.jsonl'
    if ledger_path.is_symlink() or not ledger_path.is_file(): raise ValueError('Original spend ledger is missing')
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    seeds = [e for e in events if e.get('event') == 'seed']
    if len(seeds) != 1: raise ValueError('Original ledger must have one seed')
    if seeds[0].get('catalog_sha256') != sha(release.CATALOG): raise ValueError('Original ledger catalog mismatch')
    release.Ledger.balance(events)
    base_client = LLMClient('tailscale', args.connection)
    if (base_client.profile_config.get('api_base_env') != 'LITELLM_BASE_URL' or
            base_client.profile_config.get('api_key_env') != 'LITELLM_API_KEY'):
        raise ValueError('Extension requires the team proxy environment variables LITELLM_BASE_URL and LITELLM_API_KEY')
    # No ledger writes occur when resolving request configuration.
    client = release.BudgetedClient(base_client, None, prices,
                                    max_calls=args.max_new_calls, catalog_sha256=catalog_hash)
    judge_routes = [client.configuration(m.model, reasoning_effort=m.reasoning_effort) for m in definition.members]
    if judge_routes != schedule['routes']: raise ValueError('Judge route differs from original; cache reuse is unsafe')
    cache_inventory = {}
    for call_id in schedule['requests']:
        path = v1 / 'judge-cache' / (call_id + '.json')
        if path.is_symlink() or not path.is_file(): raise ValueError('Original judgment cache is incomplete')
        call = json.loads(path.read_text())
        if (call.get('request_id') != call_id or digest(call.get('request')) != call_id
                or call['request'].get('timeout') != timeout):
            raise ValueError('Original judgment cache identity or timeout differs')
        cache_inventory[call_id] = sha(path)
    routes = {m['id']:client.configuration(m['model'], reasoning_effort=m.get('reasoning_effort'),
              **release.generation_settings(original, m)) for m in active}
    manifest = {'version':1, 'roster':config, 'v1_protocol_sha256':sha(v1 / 'protocol.json'),
                'v1_schedule_identity':identity, 'v1_ledger_sha256':sha(ledger_path), 'benchmark_sha256':benchmark.fingerprint,
                'catalog_sha256':catalog_hash, 'judge_timeout':timeout, 'judge_routes':judge_routes,
                'v1_cache_sha256':cache_inventory,
                'generation_routes':routes}
    with extension_lock(extension):
        manifest_path = extension / 'extension_manifest.json'
        new_ledger_path = extension / 'spend-ledger.jsonl'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest: raise ValueError('Extension inputs changed; use a new extension directory')
            if new_ledger_path.is_symlink() or not new_ledger_path.is_file():
                raise ValueError('Extension ledger missing alongside existing history; restore it before proceeding')
            ledger = release.Ledger(new_ledger_path)
            with ledger.locked():
                new_events = ledger._read()
                new_seeds = [e for e in new_events if e.get('event') == 'seed']
                if len(new_seeds) != 1 or new_seeds[0].get('catalog_sha256') != catalog_hash:
                    raise ValueError('Extension ledger seed or catalog mismatch')
                ledger.balance(new_events)
        else:
            # Only explicitly supplied input files may exist before the first ledger.
            permitted = {'.extension.lock', 'protocol.json', 'connection.toml',
                         'openrouter-catalog.json', 'proxy-catalog.json'}
            permitted_paths = {p.resolve() for p in (args.config, args.catalog, args.connection)}
            if any(p.is_symlink() or not p.is_file() or (p.name not in permitted and p.resolve() not in permitted_paths)
                   for p in extension.iterdir()):
                raise ValueError('Extension contains existing artifacts without a ledger; restore it before proceeding')
            ledger = release.Ledger(new_ledger_path)
            with ledger.locked():
                ledger._append({'event':'seed', 'catalog_sha256':catalog_hash, 'attempts':[]})
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        client.ledger = ledger
        cache = extension / 'judge-cache'
        prepare_cache(v1 / 'judge-cache', cache, schedule['requests'], cache_inventory)
    balance = Decimal(ledger.summary()['estimated_committed_usd'])
    remaining = ledger.cap - balance
    if args.command == 'summary':
        print(json.dumps({**ledger.summary(), 'extension_models':len(active), 'remaining_estimate_usd':str(remaining)})); return
    if args.command == 'generate':
        jobs = []
        for m in selected:
            options = release.generation_settings(original, m)
            preview = run_generation(benchmark, extension / 'generations' / m['id'], m['model'], client,
                dry_run=True, max_new_calls=args.max_new_calls, workers=args.workers,
                reasoning_effort=m.get('reasoning_effort'), timeout=original['generation_timeout'], **options)
            jobs.append((m, preview))
        requests = [r for _, p in jobs for r in p['requests']][:args.max_new_calls]
        estimate = estimate_requests(requests, prices)
        print(json.dumps({'planned_calls':len(requests), 'maximum_reservation_usd':str(estimate),
                          'remaining_estimate_usd':str(remaining), 'execute':args.execute}), flush=True)
        if args.execute and estimate > remaining: raise release.BudgetExceeded('Planned reservations exceed remaining estimated cap')
        if args.execute:
            for m, _ in jobs:
                remaining_calls = args.max_new_calls - client.started
                if remaining_calls <= 0: break
                out = run_generation(benchmark, extension / 'generations' / m['id'], m['model'], client,
                    max_new_calls=remaining_calls, workers=args.workers, reasoning_effort=m.get('reasoning_effort'),
                    timeout=original['generation_timeout'], **release.generation_settings(original, m))
                print(json.dumps({'model':m['id'], **out}), flush=True)
    else:
        dirs = [v1 / 'generations' / m['id'] for m in roster] + [extension / 'generations' / m['id'] for m in active]
        result = run_multi_evaluation(benchmark, dirs, client, output=extension / 'evaluation', cache_dir=cache,
            dry_run=True, max_new_calls=args.max_new_calls, workers=args.workers, timeout=timeout, seed=schedule['seed'])
        if result.get('status') == 'generation_incomplete':
            print(json.dumps({'status':'generation_incomplete', 'coverage':result['generation_coverage']})); return
        if not set(schedule['requests']).issubset({digest(r) for r in result['request_previews']}):
            raise ValueError('Original judgment requests changed; refusing to rejudge the v1 snapshot')
        pending = [r for r in result['request_previews'] if not (cache / (digest(r) + '.json')).exists()
                   and not (extension / 'evaluation' / 'calls' / (digest(r) + '.json')).exists()][:args.max_new_calls]
        estimate = estimate_requests(pending, prices)
        print(json.dumps({'planned_pairs':result['planned_pairs'], 'coverage':result['coverage'],
                          'maximum_reservation_usd':str(estimate), 'remaining_estimate_usd':str(remaining),
                          'execute':args.execute}), flush=True)
        if args.execute and estimate > remaining: raise release.BudgetExceeded('Planned reservations exceed remaining estimated cap')
        if args.execute:
            out = run_multi_evaluation(benchmark, dirs, client, output=extension / 'evaluation', cache_dir=cache,
                max_new_calls=args.max_new_calls, workers=args.workers, timeout=timeout, seed=schedule['seed'])
            print(json.dumps({'complete':out.get('complete'), 'coverage':out.get('coverage')}), flush=True)


if __name__ == '__main__': main()
