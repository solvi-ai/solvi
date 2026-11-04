"""Opt-in owner-model economics and independently labelled proposer history."""
import math
import time
from ...core.provenance import code_fingerprint
from ...core.runtime import vhash
from ._stages import StageVerdict


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


class Economics:
    """Refuse nonpositive expected net BEFORE later stages; all amounts share owner units.

    estimate(change, ctx) returns {gain, loss}. Costs explicitly name anticipated
    stages, not measured runtime. Admission never bypasses any subsequent stage.
    """
    name = 'economics'
    cost = 0.0

    def __init__(self, estimate, *, costs, units):
        if not callable(estimate) or not isinstance(units, str) or not units:
            raise ValueError('callable estimate and explicit units required')
        if not isinstance(costs, dict) or any(not isinstance(k, str) or not k for k in costs):
            raise ValueError('costs must name stages')
        self.estimate, self.costs, self.units = estimate, {k: _number(v, 'cost') for k, v in costs.items()}, units

    def run(self, change, ctx):
        prediction = self.estimate(change, ctx)
        gain, loss = _number(prediction['gain'], 'gain'), _number(prediction['loss'], 'loss')
        cost = math.fsum(self.costs.values())
        net = gain - loss - cost
        if not math.isfinite(net):
            raise ValueError('expected net must be finite')
        measured = dict(gain=gain, loss=loss, costs=dict(self.costs), units=self.units, expected_net=net,
                        policy_fingerprint=vhash({'estimate': code_fingerprint(self.estimate), 'costs': self.costs, 'units': self.units}))
        return StageVerdict.admit('positive owner-model net; fresh gates still required', measured) if net > 0 else StageVerdict.refuse('nonpositive owner-model net; skip later stages', measured)


class ProposerHistory:
    """Windowed, explicitly independently labelled outcomes; never infer truth from admission.

    Unknown proposers audit-and-continue by default, or hold until external labels.
    Identity and independent evidence authenticity are owner responsibilities.
    """
    name = 'proposer_history'
    cost = 0.0

    def __init__(self, *, window, min_labels=1, max_loss=.5, unknown='audit', clock=None):
        self.window = _number(window, 'window')
        self.max_loss = _number(max_loss, 'max_loss')
        if self.window == 0 or self.max_loss > 1 or type(min_labels) is not int or min_labels < 1 or unknown not in ('audit', 'hold'):
            raise ValueError('invalid history configuration')
        self.min_labels, self.unknown, self.clock = min_labels, unknown, clock or time.time

    def label(self, journal, *, proposer, loss, observed_at, evidence, event_id):
        now = _number(self.clock(), 'now')
        at, loss = _number(observed_at, 'observed_at'), _number(loss, 'loss')
        if at > now or loss > 1 or any(not isinstance(x, str) or not x for x in (proposer, evidence, event_id)):
            raise ValueError('past labelled outcome, stable proposer/evidence/event IDs required')
        if any(r.get('action') == 'proposer_outcome' and r.get('event_id') == event_id for r in journal.records()):
            raise ValueError('outcome event already labelled')
        return journal.write('proposer_outcome', proposer=proposer, loss=loss, observed_at=at,
                             labelled_at=now, evidence=evidence, event_id=event_id)

    def run(self, change, ctx):
        now = _number(self.clock(), 'now')
        if not isinstance(change.proposer, str) or not change.proposer:
            raise ValueError('stable proposer ID required')
        records = [r for r in ctx.journal.records() if r.get('action') == 'proposer_outcome'
                   and r.get('proposer') == change.proposer and now-self.window <= r['observed_at'] <= now
                   and r['labelled_at'] <= now]
        mean = math.fsum(r['loss'] for r in records)/len(records) if records else None
        measured = dict(proposer=change.proposer, labels=len(records), mean_loss=mean, window=self.window,
                        evidence=[r['evidence'] for r in records], unknown=self.unknown,
                        policy_fingerprint=vhash({'window': self.window, 'min_labels': self.min_labels,
                                                  'max_loss': self.max_loss, 'unknown': self.unknown}))
        if len(records) < self.min_labels:
            return (StageVerdict.hold if self.unknown == 'hold' else StageVerdict.admit)('unknown history; external labels required, no truth inferred', measured)
        return (StageVerdict.refuse if mean > self.max_loss else StageVerdict.admit)('independently labelled proposer history; fresh gates still required', measured)
