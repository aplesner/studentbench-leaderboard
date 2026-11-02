#!/usr/bin/env python3
"""Run the public release through one proxy and a shared estimated-cost ledger.

Catalog prices are estimates, not proxy invoices. Outstanding/failed calls retain
full reservations. No inference is made by summary or by importing this module.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from studentbench_leaderboard.benchmark import Benchmark
from studentbench_leaderboard.generation import run_generation, _request
from studentbench_leaderboard.evaluation import run_multi_evaluation
from studentbench_leaderboard.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/public-release-20261102'
CATALOG = ROOT / 'results/live-model-smoke-20261102/openrouter-catalog.json'
CAP = Decimal('1000')
JUDGE_MAX = 8192
PRICING_REVISION = 'catalog-max-input-cache-write-v2'


def _decimal(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('Invalid catalog price or usage')
    return result


def prices_from_catalog(path):
    data = json.loads(Path(path).read_text())
    result = {}
    for row in data['data']:
        try:
            pricing = row['pricing']
            tiers = [pricing] + pricing.get('overrides', [])
            result[row['id']] = (max(_decimal(t[k]) for t in tiers
                                     for k in ('prompt', 'input_cache_write', 'input_cache_write_1h') if k in t),
                                 max(_decimal(t['completion']) for t in tiers if 'completion' in t))
        except (KeyError, ValueError, InvalidOperation):
            continue
    return result


def maximum_cost(messages, schema, max_tokens, price):
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('A positive max_tokens is required for every call')
    # At most one token per UTF-8 byte, plus a generous protocol framing allowance.
    size = len(json.dumps({'messages': messages, 'schema': schema}, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) + 4096
    return Decimal(size) * price[0] + Decimal(max_tokens) * price[1]


def usage_cost(usage, price):
    if not isinstance(usage, dict): return None
    a, b = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if type(a) is not int or type(b) is not int or a < 0 or b < 0: return None
    return Decimal(a) * price[0] + Decimal(b) * price[1]


class BudgetExceeded(ValueError):
    pass


class Ledger:
    """Append-only records; flock serializes reservations across threads/processes."""
    def __init__(self, path, cap=CAP):
        self.path = Path(path)
        self.cap = _decimal(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix('.lock')
        self.mutex = threading.Lock()

    @contextmanager
    def locked(self):
        with self.mutex:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self):
        if not self.path.exists(): return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append(self, event):
        event = {'time': datetime.now(timezone.utc).isoformat(), **event}
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a') as f:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            f.flush(); os.fsync(f.fileno())

    @staticmethod
    def balance(events):
        calls = {}
        for e in events:
            if e['event'] == 'seed':
                for item in e['attempts']:
                    if item['id'] in calls: raise ValueError('Duplicate seed attempt')
                    calls[item['id']] = _decimal(item['charge_usd'])
            elif e['event'] == 'start':
                if e['id'] in calls: raise ValueError('Duplicate ledger attempt')
                calls[e['id']] = _decimal(e['reserve_usd'])
            elif e['event'] == 'finish':
                if e['id'] not in calls: raise ValueError('Orphan ledger finish')
                calls[e['id']] = _decimal(e['charge_usd'])
        return sum(calls.values(), Decimal(0))

    def reserve(self, model, amount, *, source='live', pricing_revision=None, catalog_sha256=None):
        amount = _decimal(amount)
        with self.locked():
            events = self._read()
            if self.balance(events) + amount > self.cap:
                raise BudgetExceeded('Estimated spend cap would be exceeded; no request dispatched')
            call_id = uuid.uuid4().hex
            self._append({'event':'start', 'id':call_id, 'model':model, 'reserve_usd':str(amount), 'source':source,
                          'pricing_revision':pricing_revision, 'catalog_sha256':catalog_sha256})
            return call_id

    def finish(self, call_id, amount):
        with self.locked():
            self._append({'event':'finish', 'id':call_id, 'charge_usd':str(_decimal(amount))})

    def summary(self):
        with self.locked(): events = self._read()
        return {'estimated_committed_usd':str(self.balance(events)), 'cap_usd':str(self.cap),
                'attempts':sum(e['event']=='start' for e in events) + sum(len(e['attempts']) for e in events if e['event']=='seed'),
                'unfinished':sum(e['event']=='start' for e in events)-sum(e['event']=='finish' for e in events),
                'note':'Catalog-based estimate; proxy charges and invoices may differ. Failed or unknown calls retain their maximum reservation.'}


class BudgetedClient:
    def __init__(self, client, ledger, prices, *, max_calls=None, catalog_sha256=None):
        self.client, self.ledger, self.prices = client, ledger, prices
        if getattr(client, 'profile', None) != 'tailscale': raise ValueError('Release inference requires the tailscale profile')
        self.max_calls, self.started = max_calls, 0
        self.catalog_sha256 = catalog_sha256
        self.lock = threading.Lock()

    @staticmethod
    def _settings(settings):
        settings = dict(settings)
        if 'max_tokens' not in settings and 'max_completion_tokens' not in settings:
            settings['max_tokens'] = JUDGE_MAX
        if 'max_tokens' in settings and 'max_completion_tokens' in settings:
            raise ValueError('Use only one output token limit')
        return settings

    def configuration(self, model, reasoning_effort=None, **settings):
        result = self.client.configuration(model, reasoning_effort=reasoning_effort, **self._settings(settings))
        if result.get('transport') != 'litellm_proxy' or result.get('profile') != 'tailscale':
            raise ValueError('Release inference requires the tailscale proxy route')
        return result

    def complete(self, model, messages, schema, *, reasoning_effort=None, timeout=180, **settings):
        settings = self._settings(settings)
        self.configuration(model, reasoning_effort=reasoning_effort, **settings)
        if model not in self.prices: raise ValueError(f'No frozen catalog price for {model}')
        price = self.prices[model]
        reservation = maximum_cost(messages, schema, settings.get('max_tokens', settings.get('max_completion_tokens')), price)
        with self.lock:
            if self.max_calls is not None and self.started >= self.max_calls:
                raise BudgetExceeded('Invocation call limit reached; no request dispatched')
            call_id = self.ledger.reserve(model, reservation, pricing_revision=PRICING_REVISION,
                                          catalog_sha256=self.catalog_sha256)
            self.started += 1
        try:
            raw, metadata = self.client.complete(model, messages, schema, reasoning_effort=reasoning_effort, timeout=timeout, **settings)
        except Exception as exc:
            response_metadata = getattr(exc, 'response_metadata', None)
            actual = usage_cost(response_metadata.get('usage'), price) if isinstance(response_metadata, dict) else None
            if actual is not None:
                self.ledger.finish(call_id, actual)
            # Unknown usage retains its full reservation.
            raise
        actual = usage_cost(metadata.get('usage'), price)
        if actual is not None:
            self.ledger.finish(call_id, actual)
        return raw, metadata


def seed_ledger(ledger, benchmark, protocol, prices, catalog_hash, run_dir=RUN):
    """Import pre-ledger smoke attempts once, atomically, before old artifacts move."""
    with ledger.locked():
        events = ledger._read()
        seeds = [e for e in events if e.get('event') == 'seed']
        if seeds:
            if len(seeds) != 1 or seeds[0]['catalog_sha256'] != catalog_hash:
                raise ValueError('Frozen catalog does not match existing ledger')
            return
        if events: raise ValueError('Unseeded ledger contains records')
        smoke_traces = list(run_dir.glob('*-smoke.json'))
        # Any unexpected artifacts can represent earlier charged attempts.
        fresh_allowed = {run_dir / 'protocol.json', run_dir / 'connection.toml', ledger.path, ledger.lock_path}
        has_history = any(p.is_symlink() or (p.is_file() and p not in fresh_allowed)
                          for p in run_dir.rglob('*'))
        if not smoke_traces:
            if has_history:
                raise ValueError('Historical run artifacts found without complete legacy smoke traces; restore the original ledger or complete cost records before proceeding')
            ledger._append({'event':'seed', 'catalog_sha256':catalog_hash, 'attempts':[]})
            return
        entries = []
        allowed = {run_dir / 'protocol.json', run_dir / 'connection.toml', ledger.path, ledger.lock_path}
        allowed.update(run_dir / f"{spec['id']}-smoke.json" for spec in protocol['models'])
        for spec in protocol['models']:
            smoke_file = run_dir / f"{spec['id']}-smoke.json"
            if not smoke_file.exists():
                raise ValueError(f"Missing initial smoke trace for {spec['id']}; restore complete legacy traces or the original ledger before proceeding")
            smoke = json.loads(smoke_file.read_text())
            count = smoke.get('calls_started')
            if type(count) is not int or count < 0: raise ValueError(f"Invalid initial smoke trace for {spec['id']}")
            manifest_path = run_dir / 'generations' / spec['id'] / 'generation_manifest.json'
            if not manifest_path.exists(): raise ValueError(f"Missing initial manifest for {spec['id']}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('model') != spec['model'] or not manifest.get('selected_subjects'):
                raise ValueError(f"Invalid initial manifest for {spec['id']}")
            req = _request(benchmark, manifest['selected_subjects'][0], manifest['configuration'], spec['model'], manifest['settings'])
            reserve = maximum_cost(req['messages'], req['schema'], manifest['settings']['max_tokens'], prices[spec['model']])
            saved = list((manifest_path.parent / 'generations').glob('*.json'))
            errors = list((manifest_path.parent / 'generation_errors').glob('*.json'))
            if len(saved) + len(errors) > count:
                raise ValueError(f"Unaccounted history for {spec['id']}; restore the original ledger")
            allowed.update([manifest_path, manifest_path.parent / '.generation.lock', *saved, *errors])
            charges = []
            for path in saved:
                result = json.loads(path.read_text())
                actual = usage_cost(result.get('metadata', {}).get('usage'), prices[spec['model']])
                charges.append(actual if actual is not None else reserve)
            charges.extend([reserve] * (count - len(charges)))
            entries.extend({'id':f"seed-{spec['id']}-{i}", 'model':spec['model'], 'charge_usd':str(charge)}
                           for i, charge in enumerate(charges))
        # Legacy traces describe only the initial smoke. Any additional files,
        # including archived pilots and evaluations, may represent later charges.
        for path in run_dir.rglob('*'):
            if path.is_symlink() or (path.is_file() and path not in allowed):
                raise ValueError('Unaccounted historical artifacts found; restore the original ledger before proceeding')
        if sum((_decimal(e['charge_usd']) for e in entries), Decimal(0)) > ledger.cap:
            raise BudgetExceeded('Initial calls exceed estimated budget cap')
        ledger._append({'event':'seed', 'catalog_sha256':catalog_hash, 'attempts':entries})


def select_models(protocol, requested=None, *, require_all=False):
    models = protocol['models']
    if any(type(m.get('enabled', True)) is not bool for m in models):
        raise ValueError('Model enabled must be a boolean')
    ids = [m['id'] for m in models]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate model IDs in protocol')
    active = [m for m in models if m.get('enabled', True)]
    active_ids = {m['id'] for m in active}
    selected = set(requested) if requested is not None else active_ids
    unknown = selected - set(ids)
    if unknown:
        raise ValueError('Unknown model IDs: ' + ', '.join(sorted(unknown)))
    deferred = selected - active_ids
    if deferred:
        raise ValueError('Deferred models are not enabled: ' + ', '.join(sorted(deferred)))
    if require_all and selected != active_ids:
        raise ValueError(f'Joint evaluation requires all {len(active)} active release models')
    if not active:
        raise ValueError('At least one release model must be enabled')
    return [m for m in active if m['id'] in selected]


def generation_settings(protocol, spec):
    defaults = protocol.get('generation_settings', {})
    overrides = spec.get('generation_settings', {})
    if not isinstance(defaults, dict) or not isinstance(overrides, dict):
        raise ValueError('Generation settings must be objects')
    return {**defaults, **overrides}


def judge_timeout(protocol):
    value = protocol.get('judge_timeout', protocol['generation_timeout'])
    if type(value) not in (int, float) or value <= 0:
        raise ValueError('Judge timeout must be a positive number')
    return value


def worker_count(value):
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('workers must be an integer from 1 to 64') from None
    if not 1 <= count <= 64:
        raise argparse.ArgumentTypeError('workers must be an integer from 1 to 64')
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'smoke', 'evaluate', 'summary'])
    parser.add_argument('--models', nargs='+', help='Protocol model IDs (default: all)')
    parser.add_argument('--workers', type=worker_count, default=1, help='Concurrent requests per process (1–64; default 1)')
    parser.add_argument('--max-new-calls', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.max_new_calls < 0: parser.error('max-new-calls must be nonnegative')
    protocol = json.loads((RUN / 'protocol.json').read_text())
    if protocol.get('authorized_inference_cap_usd') != 1000: raise ValueError('Protocol does not match the authorized $1000 cap')
    if any(m.get('profile') != 'tailscale' for m in protocol['models']): raise ValueError('Every protocol model must use tailscale')
    models = select_models(protocol, args.models, require_all=args.command == 'evaluate')
    prices = prices_from_catalog(CATALOG)
    # Native proxy name and OpenRouter catalog use different punctuation for this model.
    prices['anthropic/claude-haiku-5-5'] = prices['anthropic/claude-haiku-5.5']
    for model in [m['model'] for m in protocol['models']] + ['openai/gpt-5.6-terra']:
        if model not in prices: raise ValueError(f'No frozen catalog price for {model}')
    benchmark = Benchmark.load(ROOT / protocol['benchmark'])
    ledger = Ledger(RUN / 'spend-ledger.jsonl')
    catalog_hash = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    seed_ledger(ledger, benchmark, protocol, prices, catalog_hash)
    if args.command == 'summary':
        active_count = len(select_models(protocol))
        print(json.dumps({**ledger.summary(), 'active_models':active_count,
                          'deferred_models':len(protocol['models']) - active_count}, indent=2)); return
    client = BudgetedClient(LLMClient('tailscale', RUN / 'connection.toml'), ledger, prices, max_calls=args.max_new_calls, catalog_sha256=catalog_hash)
    if args.command in ('generate', 'smoke'):
        for spec in models:
            remaining = args.max_new_calls - client.started
            if remaining <= 0: break
            result = run_generation(benchmark, RUN / 'generations' / spec['id'], spec['model'], client,
                reasoning_effort=protocol['reasoning_effort'], timeout=protocol['generation_timeout'],
                workers=args.workers, max_new_calls=min(remaining, 1 if args.command == 'smoke' else remaining),
                **generation_settings(protocol, spec))
            print(json.dumps({'model':spec['id'], **{k:v for k,v in result.items() if k != 'requests'}}), flush=True)
    else:
        result = run_multi_evaluation(benchmark, [RUN / 'generations' / m['id'] for m in models], client,
            output=RUN / 'evaluation', include_new_pairs=True, workers=args.workers,
            max_new_calls=args.max_new_calls, timeout=judge_timeout(protocol),
            cache_dir=RUN / 'judge-cache')
        print(json.dumps({'complete':result.get('complete'), 'calls_started':client.started}), flush=True)
    print(json.dumps(ledger.summary()), flush=True)


if __name__ == '__main__': main()
