"""Finite S375 reference-only stable boundary selection, no labels/model corpus."""
import json
import re

import pytest

from solvi.core.deciders.llm import llm
from solvi.core.guarantees.canary import IdentityCanary
from test_identity_canary import Server, _model, PROBES, TEAMS


class Candidates(Server):
    def __init__(self):
        super().__init__()
        self.noisy = 0

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode())
        text = re.search(r'<text>\n(.*)\n</text>', body['messages'][1]['content'], re.S).group(1)
        if text == 'down':
            self.requests += 1
            raise TimeoutError('unavailable candidate')
        if text == 'noisy parcel':
            self.noisy += 1
            self.p = .6 if self.noisy % 2 else .8
        else:
            self.p = .999 if text == 'clear parcel' else .65
        return super().__call__(req, timeout)


def test_selects_stable_boundary_preserves_existing_canary_and_detects_finite_swap():
    srv = Candidates()
    m = llm('http://llm.example/v1', 'tiny-chat', opener=srv, workers=1, retries=0, sleep=lambda s: None)
    existing = IdentityCanary(PROBES).attach(m)
    existing.probe()
    reference = existing.reference
    events, calls = list(existing.events), existing.calls
    candidates = [(x, 'Which team?', TEAMS) for x in ['clear parcel', 'stable parcel', 'noisy parcel', 'down']]
    start = srv.requests
    report = IdentityCanary.pick_probes(m, candidates, repeats=3)
    assert report['probes'][0][0] == 'stable parcel' and report['selected'] == 1
    assert report['calls'] == 12 and srv.requests == start + 12
    assert report['answered_fraction'] == .75
    assert report['candidates'][0]['rejected'] == ['outside_probability_range']
    assert report['candidates'][2]['rejected'] == ['unstable']
    assert report['candidates'][3]['rejected'] == ['unanswered', 'unscored']
    assert report['detection_power'] is None
    assert m.canary is existing and existing.reference is reference
    assert existing.events == events and existing.calls == calls
    c = IdentityCanary(report['probes'], confirm=False).attach(m)
    event = c.probe()
    assert event['diagnostics']['reference_answered'] == 1
    assert event['diagnostics']['comparable_fraction'] == 1
    srv.swapped = True
    assert c.probe()['verdict'] == 'changed'


def test_no_informative_candidate_reports_empty_without_claiming_coverage():
    s = Server(); s.p = .999
    report = IdentityCanary.pick_probes(_model(s), PROBES, repeats=2)
    assert report['selected'] == 0 and report['probes'] == []
    assert not report['has_stable_boundary_probes'] and report['answered_fraction'] == 1


@pytest.mark.parametrize('kw', [{'repeats':1}, {'repeats':True}, {'max_probes':0},
                               {'max_repeat_tv':float('nan')}, {'top_probability':(.9,.5)}])
def test_invalid_policy_rejected_before_provider_calls(kw):
    s=Server()
    with pytest.raises(ValueError):
        IdentityCanary.pick_probes(_model(s), PROBES, **kw)
    assert s.requests == 0


def test_nonfinite_probability_is_explicitly_unscored(monkeypatch):
    def invalid(self):
        self.calls += len(self.probes)
        return [{"answer": "billing", "p": [float("nan"), float("nan")]} for _ in self.probes]
    monkeypatch.setattr(IdentityCanary, "_ask", invalid)
    report = IdentityCanary.pick_probes(_model(Server()), PROBES, repeats=2)
    assert report["probes"] == [] and report["answered_fraction"] == 0
    assert all(r["rejected"] == ["unscored"] for r in report["candidates"])
    assert report["calls"] == 8
