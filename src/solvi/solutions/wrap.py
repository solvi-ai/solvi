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
        elif not isinstance(storage, SQLiteStorage):
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


__all__ = ["wrap"]
