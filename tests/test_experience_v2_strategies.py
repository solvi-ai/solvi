import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from experience_v2_strategies import ARMS, run_trial


@pytest.mark.parametrize('arm', ['FIXED_MEASURE', 'FORMULA', 'FORMULA_CHECKED'])
def test_operational_known_controls(tmp_path, arm):
    result = run_trial(tmp_path / arm, 123, 'eligibility', arm)
    assert result['status'] == 'completed', result.get('error')
    assert len(result['transfers']) == 32
    assert all(e['status'] == 'correct' for e in result['transfers'])
    assert not any(e['useful'] for e in result['transfers'])
    assert result['external_calls']['act'] == 32
    assert result['counts']['external_act'] == 256
    if arm == 'FIXED_MEASURE':
        assert result['counts']['observation'] == 256
        assert result['oracle_calls'] == 0
    elif arm == 'FORMULA_CHECKED':
        assert result['oracle_calls'] == 50
        assert result['counts']['proposal_label'] == result['counts']['admission_label'] == 64
        assert result['proof_id']
        assert all(e['candidate_id'] and e['application_id'] and e['proof_id'] for e in result['transfers'])


def test_wire_messages_really_change_and_outcomes_match(tmp_path):
    original = run_trial(tmp_path / 'plain', 123, 'eligibility', 'FIXED_MEASURE')
    wire = run_trial(tmp_path / 'wire', 123, 'eligibility', 'FIXED_MEASURE', wire=True)
    assert [e['status'] for e in original['transfers']] == [e['status'] for e in wire['transfers']]
    plain_event = json.loads((tmp_path / 'plain' / 'events.jsonl').read_text().splitlines()[0])
    wire_event = json.loads((tmp_path / 'wire' / 'events.jsonl').read_text().splitlines()[0])
    assert plain_event['payload']['message_bytes'] != wire_event['payload']['message_bytes']
    assert 'w00' in wire_event['payload']['message_bytes'] and 'wt' in wire_event['payload']['message_bytes']


def test_budget_keeps_all_assigned_failures_and_no_hidden_measure(tmp_path):
    result = run_trial(tmp_path / 'budget', 123, 'inventory', 'NEW_CHECKED', limit=3)
    assert result['status'] == 'budget'
    assert len(result['transfers']) == 32
    assert not any(e['useful'] for e in result['transfers'])
    assert result['external_calls']['act'] == 0
    assert result['spent'] <= 3
    assert (tmp_path / 'budget' / 'result.json').exists()


def test_closed_seeds_and_missing_pinned_old_fail_before_trial_creation(tmp_path):
    for seed in (5001, 8101):
        with pytest.raises(ValueError):
            run_trial(tmp_path / str(seed), seed, 'inventory', 'FORMULA')
    with pytest.raises(ValueError, match='pinned OLD'):
        run_trial(tmp_path / 'old', 123, 'inventory', 'OLD_EMPIRICAL')
    assert len(ARMS) == 7


@pytest.mark.parametrize('arm', ['OLD_EMPIRICAL', 'NEW_EMPIRICAL', 'OLD_CHECKED', 'NEW_CHECKED'])
def test_learning_arms_preserve_budget_failure_without_measure_fallback(tmp_path, arm):
    # NEW-as-OLD tests the adapter ABI only; comparison runs require pinned OLD bytes.
    from solvi.experimental.experience import programs
    result = run_trial(tmp_path / arm, 123, 'eligibility', arm, old=programs, limit=300)
    assert result['status'] == 'budget'
    assert len(result['transfers']) == 32
    assert not any(e['useful'] for e in result['transfers'])
    assert result['external_calls']['act'] == 0
    assert result['counts'].get('observation', 0) == 0
    assert result['spent'] <= 300


def test_checked_formula_wire_adapter_preserves_proof_to_sql_receipt(tmp_path):
    result = run_trial(tmp_path / 'checked-wire', 123, 'eligibility', 'FORMULA_CHECKED', wire=True)
    assert result['status'] == 'completed', result.get('error')
    assert result['oracle_calls'] == 50
    assert all(e['status'] == 'correct' and e['proof_id'] and e['application_id'] for e in result['transfers'])
    events = [json.loads(line) for line in (tmp_path / 'checked-wire' / 'events.jsonl').read_text().splitlines()]
    request = next(e for e in events if e['kind'] == 'intent')['payload']['message_bytes']
    assert 'w0' in request and 'w1' in request and 'w2' in request
    assert sum(e['kind'] == 'journal_event' for e in events) == 0
    assert result['counts']['journal_event'] == len(events)


def pinned_old():
    import hashlib
    import subprocess
    from experience_v2 import load_baseline
    configured = os.environ.get('SOLVI_EXPERIENCE_V2_BASELINE')
    if configured is None:
        pytest.skip('set SOLVI_EXPERIENCE_V2_BASELINE for historical checkout integration')
    root = Path(configured)
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == '3850ef2eece111c5b82de6647867f2977f00495c'
    hashes = {leaf: hashlib.sha256((root / 'src/solvi/experimental/experience' / (leaf + '.py')).read_bytes()).hexdigest()
              for leaf in ('programs', 'acquisition')}
    return load_baseline(root, hashes)['programs']


@pytest.mark.parametrize('arm', ARMS)
def test_all_seven_arms_with_real_pinned_old_search(tmp_path, arm):
    result = run_trial(tmp_path / arm, 123, 'eligibility', arm, old=pinned_old())
    assert result['status'] in ('completed', 'rejected'), result.get('error')
    assert len(result['transfers']) == 32
    if arm.startswith(('OLD', 'NEW')):
        assert result['search']['reason'] == 'found'
        assert result['search']['ast'] is not None
    assert result['spent'] <= 200000


def test_empirical_external_counterexample_revokes_and_stops_applications(tmp_path, monkeypatch):
    import experience_v2_strategies as strategy
    zero_rows = [[0, r, d] for r in range(5) for d in range(5)]
    others = [[s, r, d] for s in range(1, 5) for r in range(5) for d in range(5)]
    chosen = [4, 1, 0]
    others.remove(chosen)
    transfer = [chosen, *others[:31]]
    monkeypatch.setattr(strategy, 'roles', lambda seed, family: dict(
        proposal=zero_rows[:8], admission=zero_rows[8:16], transfer_R=transfer[:24], transfer_H=transfer[24:]))
    result = run_trial(tmp_path / 'counterexample', 123, 'inventory', 'NEW_EMPIRICAL')
    assert result['status'] == 'completed', result.get('error')
    assert result['transfers'][0]['status'] == 'wrong'
    assert result['transfers'][0]['receipt']['value']['answer'] == 3
    assert all(e['status'] == 'abstain' for e in result['transfers'][1:])
    assert result['external_calls']['act'] == 1
    registry = result['registry']
    assert len(registry['applications']) == 1
    assert next(iter(registry['competences'].values()))['reason'] == 'external_counterexample'
    event = result['transfers'][0]['receipt']['event_id']
    assert registry['events'][event]['split'] == 'evaluation'
    assert result['counts']['competence_observe'] == 1


def test_sql_initialization_failure_is_durable_technical_censor(tmp_path, monkeypatch):
    import experience_v2_strategies as strategy
    def unavailable(*args, **kwargs):
        raise OSError('SQLite initialization unavailable')
    monkeypatch.setattr(strategy, 'TabularEnvironment', unavailable)
    result = run_trial(tmp_path / 'censored', 123, 'inventory', 'NEW_CHECKED')
    assert result['status'] == 'censored' and result['technical_censor']
    assert result['spent'] == 0
    assert len(result['transfers']) == 32
    assert all(not e['useful'] for e in result['transfers'])
    assert json.loads((tmp_path / 'censored' / 'result.json').read_text()) == result


def test_observer_failure_is_logged_trial_error_without_followup_action(tmp_path, monkeypatch):
    import experience_v2_strategies as strategy

    def fail_observation(self, application_id, example, budget):
        budget.charge('competence_observe')
        raise ValueError('registry observation failed')

    monkeypatch.setattr(strategy.Registry, 'observe', fail_observation)
    result = run_trial(tmp_path / 'observer-error', 123, 'eligibility', 'FORMULA_CHECKED')
    assert result['status'] == 'error'
    assert 'registry observation failed' in result['error']
    assert result['external_calls']['act'] == 1
    assert len(result['transfers']) == 32
    assert not any(e['useful'] for e in result['transfers'])
    stored = json.loads((tmp_path / 'observer-error' / 'result.json').read_text())
    assert stored['status'] == 'error'
    events = [json.loads(line) for line in (tmp_path / 'observer-error' / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'trial_terminal'
    assert events[-1]['payload']['status'] == 'error'
    assert not any(e['kind'] == 'registry_revision' for e in events)


def test_real_search_has_three_paid_events_and_formula_registration_is_distinct(tmp_path):
    for arm in ('NEW_CHECKED', 'FORMULA_CHECKED'):
        result = run_trial(tmp_path / arm, 123, 'eligibility', arm)
        events = [json.loads(line) for line in (tmp_path / arm / 'events.jsonl').read_text().splitlines()]
        search_events = [e for e in events if e['kind'].startswith('search_')]
        if arm == 'NEW_CHECKED':
            assert [e['kind'] for e in search_events] == ['search_intent', 'search_receipt', 'search_terminal']
            receipt = search_events[1]['payload']
            assert receipt['tried'] == result['search']['tried'] > 0
            assert receipt['counts']['search_generate'] >= receipt['tried']
        else:
            assert not search_events
            assert len([e for e in events if e['kind'].startswith('formula_registration_')]) == 3


@pytest.mark.parametrize('family', ['inventory', 'schedule'])
@pytest.mark.parametrize('limit', [300, 1000])
def test_real_engineering_search_budget_failure_always_retains_paid_prefix(tmp_path, family, limit):
    result = run_trial(tmp_path / (family + str(limit)), 0, family, 'NEW_EMPIRICAL', limit=limit)
    assert result['status'] == 'budget'
    search = result['search']
    assert search is not None
    assert search['reason'] == 'budget_exhausted'
    assert search['spent'] > 0 and search['spent'] <= result['spent'] <= limit
    assert sum(search['counts'].values()) == search['spent']
    assert not result['candidate_id']
    assert not any(entry['useful'] for entry in result['transfers'])
    assert result['external_calls']['act'] == 0
    stored = json.loads((tmp_path / (family + str(limit)) / 'result.json').read_text())
    assert stored['search'] == search


def test_paid_search_receipt_failure_retains_engine_result_and_no_activation(tmp_path, monkeypatch):
    import experience_v2_strategies as strategy
    original = strategy.Registry.propose

    def fail_after_synthesis(self, examples, budget):
        trace = self.trace
        def fail_receipt(kind, payload, paid):
            if kind == 'search_receipt':
                raise strategy.BudgetStop('semantic receipt unaffordable')
            return trace(kind, payload, paid)
        self.trace = fail_receipt
        return original(self, examples, budget)

    monkeypatch.setattr(strategy.Registry, 'propose', fail_after_synthesis)
    result = run_trial(tmp_path / 'trace-edge', 123, 'eligibility', 'NEW_EMPIRICAL')
    assert result['status'] == 'budget'
    search = result['search']
    assert search['reason'] == 'budget_exhausted'
    assert search['engine_reason'] == 'found'
    assert type(search['tried']) is int and search['tried'] > 0
    assert search['ast'] is not None
    assert sum(search['counts'].values()) == search['spent']
    assert search['terminal_censored']
    technical = json.loads((tmp_path / 'trace-edge' / 'technical.jsonl').read_text())
    assert technical['search'] == search
    assert not result['candidate_id'] and not result['registry']['competences']
    assert result['external_calls']['act'] == 0
