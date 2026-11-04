"""Explicit external decision interception; loss estimates/evidence are owner declarations."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _copy_json(value):
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError('analysis requires finite JSON with string keys')
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _loss(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class _WrappedAgent:
    def __init__(self, agent, decision_points, default, enabled, components, storage):
        if type(enabled) is not bool:
            raise TypeError('enabled must be a bool')
        if type(decision_points) is not dict or not decision_points:
            raise ValueError('decision_points must map public method names to proposers')
        points = {}
        for name, configured in decision_points.items():
            if not isinstance(name, str) or not name.isidentifier() or name.startswith('_') or name == 'enabled':
                raise ValueError('decision points must be public method names')
            configured = {name: configured} if callable(configured) else configured
            if type(configured) is not dict or not configured or any(
                not isinstance(k, str) or not k or not callable(v) for k, v in configured.items()
            ):
                raise ValueError('each decision point needs named callable proposers')
            points[name] = dict(configured)
        if default is not None and (not callable(default) or len(points) != 1):
            raise ValueError('default must be callable for exactly one decision point')
        names = {name for point in points.values() for name in point}
        if components is not None and (not isinstance(components, (list, tuple))
                                      or any(name not in names for name in components)):
            raise ValueError('components must name configured proposers')
        self._agent, self._points, self._default = agent, points, default
        self.enabled, self._components, self._storage = enabled, None if components is None else tuple(components), storage

    def __getattr__(self, name):
        if name not in self._points:
            return getattr(self._agent, name)
        def decide(*args, **kwargs):
            base = self._default if self._default is not None else getattr(self._agent, name)
            original = base(*args, **kwargs)
            selected = {component: proposer for component, proposer in self._points[name].items()
                        if self._components is None or component in self._components}
            if not self.enabled or not selected:
                return original
            try:
                analysis = _copy_json({'args': list(args), 'kwargs': kwargs, 'default': original})
            except (ValueError, TypeError, RecursionError, OverflowError):
                return original
            for component, proposer in selected.items():
                try:
                    proposal = _copy_json(proposer(_copy_json(analysis)))
                    if type(proposal) is not dict or not {'action', 'loss', 'default_loss', 'evidence'} <= proposal.keys():
                        continue
                    if (not _loss(proposal['loss']) or not _loss(proposal['default_loss'])
                            or not proposal['loss'] < proposal['default_loss']):
                        continue
                    if json.dumps(proposal['action'], sort_keys=True) == json.dumps(analysis['default'], sort_keys=True):
                        continue
                    evidence = proposal['evidence']
                    if type(evidence) is not list or not evidence or any(type(e) is not str or not e.strip() for e in evidence):
                        continue
                except Exception:  # callback failure is no evidence to override; never catches storage failures
                    continue
                self._record(name, component, proposer, analysis, proposal)
                return proposal['action']
            return original
        return decide

    def _record(self, point, component, proposer, analysis, proposal):
        from ..core.store import JSONLStorage, SQLiteStorage
        from ..core.provenance import code_fingerprint
        storage = self._storage
        if isinstance(storage, (str, Path)):
            storage = (SQLiteStorage(storage) if str(storage).endswith(('.db', '.sqlite', '.sqlite3'))
                       else JSONLStorage(storage, fsync=True))
            self._storage = storage
        if isinstance(storage, JSONLStorage):
            if not storage.fsync or storage.rotates:
                raise ValueError('overrides require nonrotating JSONLStorage(fsync=True)')
        elif isinstance(storage, SQLiteStorage):
            databases = storage.db.execute('PRAGMA database_list').fetchall()
            main = next((row[2] for row in databases if row[1] == 'main'), '')
            synchronous = storage.db.execute('PRAGMA synchronous').fetchone()[0]
            journal_mode = storage.db.execute('PRAGMA journal_mode').fetchone()[0].lower()
            if not main or synchronous < 2 or journal_mode in ('off', 'memory'):
                raise ValueError('overrides require persistent SQLite with synchronous FULL/EXTRA and a disk journal')
        else:
            raise ValueError('overrides require durable JSONL or SQLite storage')
        if getattr(storage, '_batch', None) is not None:
            raise ValueError('override journal cannot commit inside an active storage batch')
        record = {'v': 1, 'kind': 'external_override', 'point': point, 'component': component,
                  'proposer': {'module': getattr(proposer, '__module__', None),
                               'name': getattr(proposer, '__qualname__', None),
                               'code': code_fingerprint(proposer)},
                  'analysis': analysis, 'proposal': proposal}
        with storage.batch():
            storage._append(_copy_json(record))


def wrap(agent, *, decision_points, default=None, enabled=True, components=None, storage=None):
    """Intercept explicitly named external methods; other methods forward to the original agent.

    Proposers receive detached finite JSON {args, kwargs, default}; return None or
    {action, default_loss, loss, evidence: [nonempty string IDs]}. A strict finite
    nonnegative loss reduction and declared evidence permit an override, journaled
    durably before return. Invalid/unsupported analysis keeps the original object.
    The first valid selected component wins. Disabled calls the base exactly once
    without analysis, proposer calls or journal writes. Internal self.calls are not
    patched. Estimates, evidence validity and proposer purity are owner assumptions;
    this adapter is neither a sandbox nor a theorem about real-world utility.
    """
    return _WrappedAgent(agent, decision_points, default, enabled, components, storage)


def select_components(factory, evaluator, *, decision_points, seeds, directory,
                      configurations=None, max_evaluations=None, costs=None):
    """Training-only paired utilities on fresh factory(seed) agents; no holdout or certificate.

    Defaults to baseline plus each component separately. Extra combinations must be
    preregistered in configurations (tuples of component names). Costs are optional
    declared per-configuration per-cell metadata, not measured costs or utility.
    Each attempted cell is saved/fsynced; errors continue. Only complete paired
    configurations can be recommended, with baseline winning a zero-gain tie.
    Use a new directory per invocation; partial artifacts are not silently resumed.
    """
    import time
    import weakref
    import statistics
    from ..core.provenance import code_fingerprint
    if not callable(factory) or not callable(evaluator):
        raise TypeError('factory(seed) and evaluator(agent, seed) must be callable')
    if (not isinstance(seeds, (list, tuple)) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be distinct explicit builtin integers')
    seeds = tuple(seeds)
    if max_evaluations is not None and (type(max_evaluations) is not int or max_evaluations < 0):
        raise ValueError('max_evaluations must be a nonnegative integer')
    template = wrap(None, decision_points=decision_points, enabled=False)
    names = list(dict.fromkeys(name for point in template._points.values() for name in point))
    configurations = [(name,) for name in names] if configurations is None else configurations
    if not isinstance(configurations, (list, tuple)):
        raise ValueError('configurations must be a finite list of component tuples')
    configs = [()]
    for configured in configurations:
        if (not isinstance(configured, tuple) or any(type(n) is not str or n not in names for n in configured)
                or len(set(configured)) != len(configured)):
            raise ValueError('configurations must contain distinct registered component names')
        canonical = tuple(name for name in names if name in configured)
        if canonical == ():
            continue
        if canonical in configs:
            raise ValueError('duplicate component configuration')
        configs.append(canonical)
    costs = {} if costs is None else costs
    if type(costs) is not dict or any(key not in configs or not _loss(value) for key, value in costs.items()):
        raise ValueError('costs maps registered configuration tuples to finite nonnegative declared costs')
    costs = dict(costs)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'training_only': True, 'seeds': list(seeds), 'configurations': [list(c) for c in configs],
                'max_evaluations': max_evaluations, 'declared_costs': [costs.get(c, 0) for c in configs],
                'factory_code': code_fingerprint(factory), 'evaluator_code': code_fingerprint(evaluator),
                'proposers': {point: {name: code_fingerprint(fn) for name, fn in values.items()}
                              for point, values in template._points.items()}}
    def durable_json(path, value):
        with path.open('x') as output:
            json.dump(value, output, allow_nan=False, indent=2)
            output.write('\n'); output.flush()
            import os
            os.fsync(output.fileno())
    durable_json(directory/'manifest.json', manifest)
    rows, weak_agents, strong_agents = [], [], []
    import os
    with (directory/'cells.jsonl').open('x') as output:
        for config in configs:
            for seed in seeds:
                if max_evaluations is not None and len(rows) >= max_evaluations:
                    break
                index = len(rows)
                journal = directory/f'cell_{index:05d}.overrides.jsonl'
                row = {'index': index, 'configuration': list(config), 'seed': seed,
                       'override_journal': str(journal), 'declared_cost': costs.get(config, 0)}
                started = time.monotonic()
                try:
                    agent = factory(seed)  # seed is supplied before constructing each independent base
                    if any(ref() is agent for ref in weak_agents) or any(old is agent for old in strong_agents):
                        raise ValueError('factory reused an agent object; paired runs require fresh agents')
                    try:
                        weak_agents.append(weakref.ref(agent))
                    except TypeError:
                        strong_agents.append(agent)
                    wrapped = wrap(agent, decision_points=template._points, enabled=bool(config),
                                   components=config, storage=journal)
                    utility = evaluator(wrapped, seed)
                    if type(utility) not in (int, float) or not math.isfinite(utility):
                        raise ValueError('evaluator must return a finite numeric utility')
                    row.update(status='ok', utility=utility)
                except Exception as error:
                    row.update(status='error', error=f'{type(error).__name__}: {error}')
                row['seconds'] = time.monotonic()-started
                output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush(); os.fsync(output.fileno())
                rows.append(row)
    baseline = {r['seed']: r['utility'] for r in rows if r['configuration'] == [] and r['status'] == 'ok'}
    comparisons, best, best_gain = [], None, 0.0
    if len(baseline) == len(seeds):
        best = []
    for config in configs[1:]:
        available = {r['seed']: r['utility'] for r in rows
                     if r['configuration'] == list(config) and r['status'] == 'ok'}
        pairs, invalid_deltas = [], []
        for seed in seeds:
            if seed not in baseline or seed not in available:
                continue
            delta = available[seed]-baseline[seed]
            try:
                finite = math.isfinite(delta)
            except OverflowError:
                finite = False
            if finite:
                pairs.append({'seed': seed, 'delta': delta})
            else:
                invalid_deltas.append({'seed': seed, 'error': 'paired utility difference is not finite'})
        mean = statistics.mean(p['delta'] for p in pairs) if pairs else None
        complete = len(pairs) == len(seeds)
        comparisons.append({'configuration': list(config), 'paired_deltas': pairs,
                            'mean_paired_delta': mean, 'complete_pairs': complete, 'invalid_deltas': invalid_deltas})
        if complete and mean > best_gain:
            best, best_gain = list(config), mean
    total_cost = sum(r['declared_cost'] for r in rows)
    try:
        finite_cost = math.isfinite(total_cost)
    except OverflowError:
        finite_cost = False
    if not finite_cost:
        total_cost = None
    result = {'training_only': True, 'cells': rows, 'comparisons': comparisons,
              'best_registered_config_training': best, 'best_mean_paired_delta': best_gain if best is not None else None,
              'budget_exhausted': len(rows) < len(configs)*len(seeds),
              'declared_total_cost': total_cost, 'declared_total_cost_overflow': total_cost is None,
              'directory': str(directory),
              'scope': 'descriptive paired training utilities; no confidence, holdout or welfare certificate'}
    durable_json(directory/'result.json', result)
    return result


__all__ = ['wrap', 'select_components']
