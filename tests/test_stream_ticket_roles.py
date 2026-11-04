"""Trusted role boundary: payload blindness does not imply arbitrary access blindness."""
import asyncio

import pytest

from solvi.core.store import JSONLStorage
from solvi.serve import NotFound, Service, call_tool, mcp_tools
from test_stream_tickets import _desk, _gold, _run


def test_irrelevant_input_reprobe_is_still_possible_for_trusted_client(tmp_path):
    desk, sp, _ = _desk(JSONLStorage(tmp_path / 'probe.jsonl'))
    try:
        _run(desk, range(400))
        assert sp.threshold < 1
        audit = None
        for case in range(400, 800):
            r = desk.decide({'case': case})
            if r['status'] == 'needs_verdict':
                if desk._open[r['ticket']]['kind'] == 'audit':
                    audit = r
                    break
                desk.resolve(r['ticket'], _gold(case))
        assert audit is not None
        n = desk.dispatcher.n
        # This unused field bypasses serialized-input dedup, and the model answer is revealed.
        replies = [desk.decide({**audit['input'], 'unused_probe': i}) for i in range(12)]
        guesses = [r['answer'] for r in replies if r['status'] == 'decided']
        assert guesses and set(guesses) == {_gold(audit['input']['case'])}
        assert desk.dispatcher.n == n + 12
        oracle = Service(agent_oracle=desk)
        n = desk.dispatcher.n
        with pytest.raises(KeyError):
            call_tool(oracle, 'decide', {'state': {**audit['input'], 'unused_probe': 99}})
        assert desk.dispatcher.n == n
    finally:
        desk.close()


def test_roles_recover_blind_queue_and_ack_without_answer_or_trace(tmp_path):
    path = tmp_path / 'queue.jsonl'
    desk, _, _ = _desk(JSONLStorage(path))
    client, oracle = Service(agent_oracle=desk.client()), Service(agent_oracle=desk.oracle())
    invalid, bad = call_tool(client, 'decide', {'state': 5})
    assert bad and 'JSON object' in invalid['error']
    ticket, bad = call_tool(client, 'decide', {'state': {'case': 1}, 'request_id': 'one'})
    assert not bad
    with pytest.raises(KeyError):
        call_tool(client, 'resolve', {'ticket': ticket['ticket'], 'label': _gold(1)})
    desk.close()
    desk, _, _ = _desk(JSONLStorage(path))
    try:
        oracle = Service(agent_oracle=desk)
        pending, bad = call_tool(oracle, 'open_tickets', {})
        assert not bad and pending == {'tickets': [ticket]}
        assert {t['name'] for t in mcp_tools(oracle)} == {'resolve', 'open_tickets', 'give_up'}
        result, bad = call_tool(oracle, 'resolve', {'ticket': ticket['ticket'], 'label': _gold(1)})
        assert not bad and result == {'status': 'resolved'}
        assert desk.dispatcher.replay_stream()['ok']
        assert call_tool(oracle, 'open_tickets', {}) == ({'tickets': []}, False)
    finally:
        desk.close()


def test_service_generic_entrypoints_cannot_bypass_role(tmp_path):
    desk, _, _ = _desk(JSONLStorage(tmp_path / 'bypass.jsonl'))
    try:
        for view in (desk.client(), desk.oracle()):
            svc = Service(agent_oracle=view, decider=type("Decider", (), {"model_id": "test"})())
            for op in (lambda: svc.ask({'case': 1}), lambda: svc.tool('intent', {'case': 1}),
                       lambda: svc.ask_text('case 1'), lambda: svc.systemone({}),
                       lambda: asyncio.run(svc.aask({'case': 1})),
                       lambda: asyncio.run(svc.aask_text('case 1'))):
                with pytest.raises(NotFound, match='permitted MCP tools'):
                    op()
        assert desk.dispatcher.n == 0
    finally:
        desk.close()
