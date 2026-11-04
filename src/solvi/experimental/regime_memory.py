"""Bounded asynchronous finite-regime filtering with owner-declared noise and transitions."""
import copy
import hashlib
import json
import math


def _id(x):
    if not isinstance(x, str) or not x:
        raise ValueError('nonempty stable IDs required')
    return x


def _vector(xs, n, *, distribution=False):
    if not isinstance(xs, (list, tuple)) or len(xs) != n:
        raise ValueError('one probability per regime required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in xs):
        raise ValueError('finite probabilities in [0,1] required')
    xs = tuple(float(x) for x in xs)
    if distribution and not math.isclose(math.fsum(xs), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('normalized distribution required')
    return xs


def _lse(xs):
    top = max(xs)
    return -math.inf if top == -math.inf else top + math.log(math.fsum(math.exp(x-top) for x in xs))


def _log(x):
    return math.log(x) if x else -math.inf


class RegimeMemory:
    """Filtering, not change detection or an automatic LawSet activator.

    Regimes map stable regime IDs to owner-managed LawSet IDs. Outcomes at one
    issue step are conditionally independent given that step's latent regime.
    Epoch/model identities and likelihood calibration are owner declarations.
    """
    def __init__(self, *, epoch, model_id, regimes, prior, transition, probes,
                 max_steps=128, max_pending=128, max_delay=32):
        _id(epoch); _id(model_id)
        if not isinstance(regimes, dict) or not regimes:
            raise ValueError('regime to lawset-ID mapping required')
        for k, v in regimes.items(): _id(k); _id(v)
        if any(type(x) is not int or x < 1 for x in (max_steps, max_pending)) or type(max_delay) is not int or max_delay < 0:
            raise ValueError('finite positive history/pending limits required')
        n = len(regimes)
        prior = _vector(prior, n, distribution=True)
        if not isinstance(transition, (list, tuple)) or len(transition) != n:
            raise ValueError('square transition matrix required')
        transition = tuple(_vector(row, n, distribution=True) for row in transition)
        if not isinstance(probes, dict) or not probes:
            raise ValueError('explicit probe likelihood tables required')
        tables = {}
        for probe, outcomes in probes.items():
            _id(probe)
            if not isinstance(outcomes, dict) or not outcomes: raise ValueError('finite outcomes required')
            tables[probe] = {_id(k): _vector(v, n) for k, v in outcomes.items()}
            for i in range(n): _vector([v[i] for v in tables[probe].values()], len(outcomes), distribution=True)
        self._config = dict(epoch=epoch, model_id=model_id, regimes=dict(sorted(regimes.items())), prior=prior,
                            transition=transition, probes=tables, max_steps=max_steps,
                            max_pending=max_pending, max_delay=max_delay)
        self._fingerprint = hashlib.sha256(json.dumps(self._config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self._tick, self._tickets = 0, {}
        self._weights = prior

    @property
    def weights(self): return self._weights

    @property
    def regime_ids(self): return tuple(self._config['regimes'])

    @property
    def lawset_ids(self): return tuple(self._config['regimes'].values())

    def _forward(self, tickets, tick):
        n = len(self.weights)
        logs = tuple(_log(x) for x in self._config['prior'])
        for step in range(tick+1):
            if step:
                logs = tuple(_lse([logs[i] + _log(self._config['transition'][i][j]) for i in range(n)]) for j in range(n))
            for key, row in sorted(tickets.items()):
                if row['step'] == step and row['outcome'] is not None:
                    likelihood = self._config['probes'][row['probe']][row['outcome']]
                    logs = tuple(x + _log(p) for x, p in zip(logs, likelihood))
            norm = _lse(logs)
            if not math.isfinite(norm): raise ValueError('observations impossible under declared model')
            logs = tuple(x-norm for x in logs)
        return tuple(math.exp(x) for x in logs)

    def issue(self, ticket_id, probe):
        _id(ticket_id)
        if probe not in self._config['probes'] or ticket_id in self._tickets:
            raise ValueError('unknown probe or reused ticket ID')
        if len(self._tickets) >= self._config['max_steps'] * self._config['max_pending'] or sum(r['outcome'] is None for r in self._tickets.values()) >= self._config['max_pending']:
            raise ValueError('bounded ticket capacity exhausted')
        self._tickets[ticket_id] = dict(probe=probe, step=self._tick, outcome=None, resolved=None)
        return dict(ticket_id=ticket_id, epoch=self._config['epoch'], model=self._fingerprint, step=self._tick, probe=probe)

    def advance(self):
        if self._tick + 1 >= self._config['max_steps']: raise ValueError('bounded history exhausted; use a new epoch')
        weights = self._forward(self._tickets, self._tick+1)
        self._tick += 1
        self._weights = weights
        return weights

    def observe(self, ticket, outcome):
        if not isinstance(ticket, dict) or set(ticket) != {'ticket_id','epoch','model','step','probe'}:
            raise ValueError('issued ticket required')
        key = ticket['ticket_id']
        if not isinstance(key,str) or key not in self._tickets: raise ValueError('unknown ticket')
        row = self._tickets[key]
        expected = dict(ticket_id=key,epoch=self._config['epoch'],model=self._fingerprint,step=row['step'],probe=row['probe'])
        if ticket != expected or row['outcome'] is not None or self._tick-row['step'] > self._config['max_delay']:
            raise ValueError('stale, foreign or already resolved ticket')
        if not isinstance(outcome,str) or outcome not in self._config['probes'][row['probe']]: raise ValueError('unknown outcome')
        tickets = copy.deepcopy(self._tickets)
        tickets[key].update(outcome=outcome,resolved=self._tick)
        weights = self._forward(tickets,self._tick)
        self._tickets,self._weights = tickets,weights
        return weights

    def snapshot(self):
        return copy.deepcopy(dict(version=1,config=self._config,tick=self._tick,tickets=self._tickets))

    @classmethod
    def restore(cls, state):
        if not isinstance(state,dict) or set(state) != {'version','config','tick','tickets'} or type(state['version']) is not int or state['version'] != 1:
            raise ValueError('invalid regime snapshot')
        try:
            obj = cls(**state['config'])
            tick, tickets = state['tick'], copy.deepcopy(state['tickets'])
            if type(tick) is not int or not 0 <= tick < obj._config['max_steps'] or not isinstance(tickets,dict): raise ValueError('invalid tick/tickets')
            if len(tickets)>obj._config['max_steps']*obj._config['max_pending']:raise ValueError('ticket capacity exceeded')
            for k,r in tickets.items():
                _id(k)
                if not isinstance(r,dict) or set(r) != {'probe','step','outcome','resolved'}:raise ValueError('invalid ticket row')
                if r['probe'] not in obj._config['probes'] or type(r['step']) is not int or not 0 <= r['step'] <= tick:raise ValueError('invalid issue step')
                if r['outcome'] is None:
                    if r['resolved'] is not None:raise ValueError('invalid pending ticket')
                elif r['outcome'] not in obj._config['probes'][r['probe']] or type(r['resolved']) is not int or not r['step']<=r['resolved']<=min(tick,r['step']+obj._config['max_delay']):raise ValueError('invalid resolution')
            if sum(r['outcome'] is None for r in tickets.values())>obj._config['max_pending']:raise ValueError('pending capacity exceeded')
            weights=obj._forward(tickets,tick)
        except (TypeError,KeyError,OverflowError) as exc:
            raise ValueError('invalid regime snapshot') from exc
        obj._tick,obj._tickets,obj._weights=tick,tickets,weights
        return obj
