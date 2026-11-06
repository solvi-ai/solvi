import pytest
from test_experience_competences import setup_registry, row
from solvi.experimental.experience.competences import CompetenceRegistry
from solvi.experimental.experience.domain_checks import FiniteDomain, DomainReceipt
from solvi.experimental.experience._common import digest


class Oracle:
    source_id = 'audit'
    source_revision = 'v1'
    def __init__(self):
        self.calls = 0
    def read(self, request_id, revision, point):
        self.calls += 1
        return DomainReceipt(request_id, 'read:' + request_id, self.source_id, revision,
                             digest(point), 2 * point['x'] + 1)


def fixture():
    g, r, b, c = setup_registry()
    d = FiniteDomain('domain', 'm1', 's1', g, 'audit', 'v1', [{'x': 8}, {'x': 9}], budget=b)
    return g, r, b, c, d, Oracle()


def test_source_change_during_store_debit_cannot_activate():
    from solvi.core.knowledge import KnowledgeStore
    g, _, b, _, d, o = fixture()
    store = KnowledgeStore()
    store.gates = []
    r = CompetenceRegistry(g, store=store)
    c = r.propose([row(x, 'proposal') for x in (0, 2, 3)], b)
    def change(snapshot):
        if snapshot['counts'].get('competence_store', 0):
            o.source_revision = 'v2'
    b.on_charge = change
    result = r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b,
                             domain=d, oracle=o, checked_store_policy='evidence_only')
    assert not result.accepted and result.reason == 'interrupted'
    assert not r.snapshot()['competences']


def test_unavailable_oracle_is_not_read_or_certified():
    g, r, b, c, d, o = fixture()
    o.available = False
    result = r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    assert not result.accepted and o.calls == 0
    assert r.snapshot()['checked'][c.fingerprint]['result']['status'] == 'unknown'


def test_oracle_failure_with_withdrawal_revokes_previous_operation():
    g, r, b, c, d, o = fixture()
    assert r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o).accepted
    next_candidate = r.propose([row(x, 'proposal') for x in (10, 11, 12)], b)
    def unavailable(*args):
        o.available = False
        raise OSError('source withdrawn')
    o.read = unavailable
    result = r.admit_checked(next_candidate, [row(x, 'admission') for x in (14, 15, 16, 17)],
                             b, domain=d, oracle=o)
    assert not result.accepted
    assert r.snapshot()['competences'][c.fingerprint]['status'] == 'revoked'


def test_checked_roundtrip_and_membership():
    g, r, b, c, d, o = fixture()
    events = []
    rows = [row(x, 'admission') for x in (4, 5, 6, 7)]
    assert r.admit_checked(c, rows, b, domain=d, oracle=o,
                           on_event=lambda k, v: events.append(k)).accepted
    assert o.calls == 2
    assert events == ['domain_read_intent', 'domain_read_receipt'] * 2
    assert r.admit_checked(c, rows, b, domain=d, oracle=o).accepted
    assert o.calls == 2
    r = CompetenceRegistry.restore(r.snapshot(), grammar=g)
    kw = dict(scope='domain', mechanism='m1', schema='s1', budget=b)
    assert r.predict({'x': 8}, task_id='a', **kw).status == 'unknown'
    assert r.predict({'x': 8}, task_id='b', domain=d, **kw).value == 17
    assert r.predict({'x': 10}, task_id='c', domain=d, **kw).status == 'unknown'
    with pytest.raises(ValueError):
        r.admit(c, rows, b)


def test_commit_failure_poisoned_and_no_active():
    g, r, b, c, d, o = fixture()
    def broken():
        raise RuntimeError('disk')
    with pytest.raises(RuntimeError, match='disk'):
        r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b,
                        domain=d, oracle=o, on_committed=broken)
    with pytest.raises(RuntimeError, match='poisoned'):
        r.snapshot()
    with pytest.raises(RuntimeError):
        r.propose([], b)


def test_commit_failure_preserves_previously_saved_durable_state():
    g, r, b, c, d, o = fixture()
    saved = []
    def commit():
        saved.append(r.snapshot())
        raise RuntimeError('after durable write')
    with pytest.raises(RuntimeError):
        r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b,
                        domain=d, oracle=o, on_committed=commit)
    restored = CompetenceRegistry.restore(saved[0], grammar=g)
    assert restored.snapshot()['admissions'][c.fingerprint]['result']['accepted']


def test_source_retract_and_rehashed_corrupt_application():
    g, r, b, c, d, o = fixture()
    r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    r.predict({'x': 8}, task_id='app', scope='domain', mechanism='m1', schema='s1', budget=b, domain=d)
    state = r.snapshot()
    old = next(iter(state['applications']))
    app = state['applications'].pop(old)
    app['inputs'] = {'x': 10}
    state['applications'][digest(app)] = app
    state['digest'] = digest({k: v for k, v in state.items() if k != 'digest'})
    with pytest.raises(ValueError, match='outside verified domain'):
        CompetenceRegistry.restore(state, grammar=g)
    assert r.retract_source('audit', 'v1', event_id='withdraw') == (c.fingerprint,)


def test_source_version_changes_during_read_fail_closed():
    g, r, b, c, d, o = fixture()
    original = o.read
    def changed(*args):
        receipt = original(*args)
        o.source_revision = 'v2'
        return receipt
    o.read = changed
    result = r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    assert not result.accepted
    assert not r.snapshot()['competences']


def test_predict_debit_callback_valueerror_propagates():
    g, r, b, c, d, o = fixture()
    r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    def failed(snapshot):
        if snapshot['counts'].get('program_node', 0) > initial:
            raise ValueError('durable debit failed')
    initial = b.counts.get('program_node', 0)
    b.on_charge = failed
    with pytest.raises(ValueError, match='durable debit failed'):
        r.predict({'x': 8}, task_id='app', scope='domain', mechanism='m1', schema='s1', budget=b, domain=d)


def test_store_evidence_bound_to_writer_and_required_on_restore():
    from solvi.core.knowledge import KnowledgeStore
    g, _, b, _, d, o = fixture()
    store = KnowledgeStore()
    store.gates = []
    r = CompetenceRegistry(g, store=store)
    c = r.propose([row(x, 'proposal') for x in (0, 2, 3)], b)
    rows = [row(x, 'admission') for x in (4, 5, 6, 7)]
    with pytest.raises(ValueError, match='evidence_only'):
        r.admit_checked(c, rows, b, domain=d, oracle=o)
    result = r.admit_checked(c, rows, b, domain=d, oracle=o, checked_store_policy='evidence_only')
    assert result.accepted
    state = r.snapshot()
    sid = state['competences'][c.fingerprint]['store_id']
    assert store.item(sid)['kind'] == 'episode'
    assert store.item(sid)['by'] == d.source_id
    CompetenceRegistry.restore(state, grammar=g, store=store)
    with pytest.raises(ValueError, match='bound store'):
        CompetenceRegistry.restore(state, grammar=g)
    original_item = store.item
    def changed(iid):
        item = original_item(iid)
        item['by'] = 'forged'
        return item
    store.item = changed
    with pytest.raises(ValueError, match='provenance'):
        CompetenceRegistry.restore(state, grammar=g, store=store)


def test_real_store_source_retraction_prevents_checked_application():
    from solvi.core.knowledge import KnowledgeStore
    g, _, b, _, d, o = fixture()
    store = KnowledgeStore()
    store.gates = []
    r = CompetenceRegistry(g, store=store)
    c = r.propose([row(x, 'proposal') for x in (0, 2, 3)], b)
    result = r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b,
                             domain=d, oracle=o, checked_store_policy='evidence_only')
    assert result.accepted
    assert store.retract_source(d.source_id, why='withdraw oracle')
    app = r.predict({'x': 8}, task_id='post-withdrawal', scope='domain', mechanism='m1',
                    schema='s1', budget=b, domain=d)
    assert app.status == 'unknown'
    assert r.snapshot()['competences'][c.fingerprint]['status'] == 'revoked'


def test_checked_domain_duck_cannot_authorize_membership():
    g, r, b, c, d, o = fixture()
    r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    class Forged:
        fingerprint = d.fingerprint
        def contains(self, inputs, budget): return True
    with pytest.raises(ValueError, match='exact finite domain'):
        r.predict({'x': 90}, task_id='forged', scope='domain', mechanism='m1', schema='s1', budget=b, domain=Forged())


def test_keyboardinterrupt_commit_poison_writer():
    g, r, b, c, d, o = fixture()
    def interrupted(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b,
                        domain=d, oracle=o, on_committed=interrupted)
    with pytest.raises(RuntimeError, match='poisoned'):
        r.snapshot()
    with pytest.raises(RuntimeError):
        r.predict({'x': 8}, task_id='post-crash', scope='domain', mechanism='m1', schema='s1', budget=b, domain=d)


def test_restore_checked_source_must_still_be_allowed():
    g, r, b, c, d, o = fixture()
    r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    state = r.snapshot()
    state['allowed_sources'] = ['environment']
    state['digest'] = digest({k:v for k,v in state.items() if k != 'digest'})
    with pytest.raises(ValueError, match='untrusted restored checked source'):
        CompetenceRegistry.restore(state, grammar=g, allowed_sources=('environment',))


def test_bad_repeat_does_not_poison_and_application_type_tamper_rejected():
    g, r, b, c, d, o = fixture()
    rows = [row(x, 'admission') for x in (4, 5, 6, 7)]
    r.admit_checked(c, rows, b, domain=d, oracle=o)
    with pytest.raises(ValueError):
        r.admit_checked(c, rows[:-1], b, domain=d, oracle=o)
    r.predict({'x': 8}, task_id='app', scope='domain', mechanism='m1', schema='s1', budget=b, domain=d)
    state = r.snapshot()
    app = state['applications'].pop(next(iter(state['applications'])))
    app['prediction'] = True
    state['applications'][digest(app)] = app
    state['digest'] = digest({k:v for k,v in state.items() if k != 'digest'})
    with pytest.raises(ValueError, match='prediction mismatch'):
        CompetenceRegistry.restore(state, grammar=g)


def test_nonaccepted_proof_dto_also_validated():
    g, r, b, c, d, o = fixture()
    o.source_revision = 'v2'
    r.admit_checked(c, [row(x, 'admission') for x in (4, 5, 6, 7)], b, domain=d, oracle=o)
    state=r.snapshot()
    state['checked'][c.fingerprint]['result']={'status':'verified','checked_count':9999,'proof_id':'INVALID'}
    state['digest']=digest({k:v for k,v in state.items() if k!='digest'})
    with pytest.raises(ValueError):
        CompetenceRegistry.restore(state,grammar=g)


def test_first_manifest_debit_reentry_blocked_before_empirical_admission():
    g, r, b, c, d, o = fixture()
    rows=[row(x, 'admission') for x in (4, 5, 6, 7)]
    attempted=[]
    def reenter(snapshot):
        b.on_charge=None
        with pytest.raises(RuntimeError):
            r.admit(c, rows, b)
        attempted.append(True)
    b.on_charge=reenter
    result=r.admit_checked(c, rows, b, domain=d, oracle=o)
    assert result.accepted and attempted == [True]
    assert r.snapshot()['admissions'][c.fingerprint]['mode']=='checked'
