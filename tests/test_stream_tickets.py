"""solvi.experimental.stream, stage 3a: tickets — the agent as the oracle of a stream-certified Dispatcher, blind,
durable, never silently dropped; and `solvi serve --mcp --agent-oracle`."""
import hashlib
import io
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import RiskDetector, StreamPromise
    from solvi.experimental.stream.tickets import UNKNOWN, AgentOracle, TicketDesk, TicketError

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.dispatch import Dispatcher
from solvi.core.store import KEPT, JSONLStorage

GRID = [round(1 - 0.05 * k, 2) for k in range(1, 20)]
TEXT = "Which team handles this case?"


@pytest.mark.parametrize('flags,message', [([], 'add --mcp'),
    (['--mcp', '--backend', 'torch'], '--backend'), (['--mcp', '--pull'], '--pull'),
    (['--mcp', '--allow-client-no-store'], '--allow-client-no-store')])
def test_oracle_cli_refuses_ignored_flags_before_loading(flags, message, monkeypatch, capsys):
    import solvi.serve as serve
    from solvi.cli import main
    monkeypatch.setattr(serve, 'load_object', lambda _: pytest.fail('must reject flags before loading'))
    with pytest.raises(SystemExit) as exc:
        main(['serve', 'desk:build', '--agent-oracle', *flags])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_oracle_initialize_instructions_and_label_schema(tmp_path):
    from solvi.serve import Service, run_builtin
    desk, _, _ = _desk(JSONLStorage(tmp_path / 'instructions.jsonl'))
    try:
        output = io.StringIO()
        run_builtin(Service(agent_oracle=desk), io.StringIO(json.dumps({
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}})+'\n'), output)
        instructions = json.loads(output.getvalue())['result']['instructions']
        assert 'oracle' in instructions and 'request_id' in instructions and 'certify' in instructions
        resolve = next(t for t in desk.mcp_tools() if t['name'] == 'resolve')
        assert resolve['inputSchema']['properties']['label']['enum'] == ['billing', 'cards']
    finally:
        desk.close()


def test_idle_mcp_pipe_expires_ticket_without_tools_call(tmp_path):
    import threading
    from solvi.serve import Service, run_builtin
    clock = Clock()
    desk, _, _ = _desk(JSONLStorage(tmp_path / 'idle.jsonl'), clock=clock, ttl=1)
    ticket = desk.decide({'case': 1})['ticket']
    reader, writer = os.pipe()
    output = io.StringIO()
    errors = []
    def run():
        try:
            with os.fdopen(reader) as stream:
                run_builtin(Service(agent_oracle=desk), stream, output)
        except BaseException as exc:  # noqa: BLE001 — surface every server-thread failure to the test owner
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        clock.t += 2
        import time
        until = time.monotonic() + 3
        while time.monotonic() < until:
            if _records(desk.dispatcher.storage, 'pending_expired'):
                break
            time.sleep(.02)
        expired = _records(desk.dispatcher.storage, 'pending_expired')
        assert expired and expired[0][1]['pending']['ticket'] == ticket
    finally:
        os.close(writer)
        thread.join(timeout=3)
        desk.close()
    assert not thread.is_alive() and not errors


def _p(case):
    """System 1's probability of "billing" for a case: from a hash of its number, never the number itself."""
    return 0.02 + 0.96 * (int(hashlib.sha256(str(case).encode()).hexdigest()[:8], 16) / 16 ** 8)


def _s1():
    cat = Catalog()

    @cat.rule("intent")
    def intent(case: int):
        p = _p(case)
        return Decision("billing" if p > 0.5 else "cards", {"billing": p, "cards": 1 - p}, confidence=max(p, 1 - p))
    return System(cat, [Question("intent", TEXT, Answer.choice(["billing", "cards"]))])


def _gold(case):
    return "billing" if _p(case) > 0.5 else "cards"            # System 1 is right: the certificate certifies


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _desk(store, clock=None, probation=50, ttl=600, arl=200, deduplicate_open=True):
    s1 = _s1()
    sp = StreamPromise(max_risk=0.05, grid=GRID)
    s1.guarantee("intent", promise=sp)
    det = RiskDetector(0.05, arl=arl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = Dispatcher(s1, AgentOracle(s1), audit="adaptive", promise=sp, detector=det, on_drift="recertify",
                       probation=probation, secret="k", storage=store)
    return TicketDesk(d, ttl=ttl, clock=clock or Clock(), deduplicate_open=deduplicate_open), sp, det


def _run(desk, cases, answer=_gold, keep_open=()):
    """Each case decided; every ticket resolved at once with `answer` (except the cases in keep_open)."""
    tickets, out = {}, []
    for c in cases:
        r = desk.decide({"case": c})
        out.append(r)
        if r["status"] == "needs_verdict":
            tickets[c] = r
            if c not in keep_open:
                desk.resolve(r["ticket"], answer(c))
    return out, tickets


def _records(store, kind):
    return [(s.id, s.data) for s in store.iter(kind=kind)]


def test_a_ticket_is_blind_and_audit_and_escalation_tickets_look_alike(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    desk, sp, _ = _desk(store)
    _, tickets = _run(desk, range(400))
    kinds = {r["pending"]["ticket"]: r["pending"]["kind_hidden"] for _, r in _records(store, "pending")}
    issues = {r["pending"]["ticket"]: r["issue"] for _, r in _records(store, "pending")}
    seen = {kinds[t["ticket"]] for t in tickets.values()}
    assert seen == {"think", "audit"}                         # both kinds were issued (escalations, then audits)
    shapes = set()
    for c, t in tickets.items():
        # exactly the ticket id, the question, all options, the input as given, the question's text, the time left
        assert t == {"status": "needs_verdict", "ticket": t["ticket"], "question": "intent",
                     "options": ["billing", "cards"], "input": {"case": c}, "instructions": TEXT,
                     "expires_in_s": 600.0}
        assert len(t["ticket"]) == 24
        shapes.add(json.dumps({**t, "ticket": None, "input": None}, sort_keys=True))
        text = json.dumps({**t, "ticket": ""})                 # the id is random: its letters say nothing
        for word in ("audit", "sample", "think", "check", "guess", "confidence", "prob", "reason", "s1", "score"):
            assert word not in text.lower()
        r1 = issues[t["ticket"]]["s1"]["results"]["intent"]
        for v in list((r1.get("probs") or {}).values()) + [r1.get("confidence")]:
            assert repr(v) not in text and f"{v:.4f}" not in text
    assert len(shapes) == 1                                   # byte for byte the same payload for both kinds
    assert "pending" in KEPT
    with pytest.raises(RuntimeError, match="tickets"):        # the agent never runs in the process
        _desk(JSONLStorage(tmp_path / "u.jsonl"))[0].dispatcher.ask({"case": 1})   # uncertified: it escalates


def test_resolve_feeds_the_certificate_with_its_source_and_record(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    desk, sp, _ = _desk(store)
    out, tickets = _run(desk, range(300))
    assert sp.threshold < 1.0 and any(r["status"] == "decided" and r["by"] == "s1" for r in out)
    pend = {r["pending"]["ticket"]: rid for rid, r in _records(store, "pending")}
    disp = [r for _, r in _records(store, "dispatch")]
    with_ticket = [r for r in disp if r.get("pending")]
    assert len(with_ticket) == len(tickets) == len(pend)
    for r in with_ticket:
        s = r["stream"]["sample"]
        if s["sampled"]:
            assert s["label_source"] == "s2" and s["label_of"] == pend[r["pending"]] and not s.get("lost")
        assert r["s2"]["mode"] == "agent" and r["s2"]["record"]["oracle"]["ticket"] == r["pending"]
    for r in disp:
        if not r.get("pending"):
            assert r["stream"]["sample"]["label_of"] is None and "lost" not in r["stream"]["sample"]
    assert sp.n == sum(r["stream"]["sample"]["sampled"] for r in disp)
    # a decision number is reserved at issue: the numbers are 1..300 in issue order, whatever the completion order
    assert sorted(r["n"] for r in disp) == list(range(1, 301))


def test_a_person_and_the_slow_path_are_recorded_by_source(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    desk, sp, _ = _desk(store)
    a = desk.decide({"case": 1})
    b = desk.decide({"case": 2})
    assert a["status"] == b["status"] == "needs_verdict"     # uncertified: every decision escalates
    ra = desk.resolve(a["ticket"], _gold(1), source="human")
    desk.resolve(b["ticket"], _gold(2))
    assert ra["answer"] == _gold(1) and ra["status"] == "resolved"
    recs = {r["pending"]: r for _, r in _records(store, "dispatch")}
    assert recs[a["ticket"]]["stream"]["sample"]["label_source"] == "human"
    assert recs[a["ticket"]]["s2"]["record"]["oracle"]["label_source"] == "human"
    assert recs[b["ticket"]]["stream"]["sample"]["label_source"] == "s2"
    with pytest.raises(ValueError, match="source"):
        desk.resolve(a["ticket"], "billing", source="model")


def test_errors_double_resolve_unknown_ticket_and_a_label_outside_the_options(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    desk, _, _ = _desk(store)
    t = desk.decide({"case": 3})["ticket"]
    with pytest.raises(TicketError, match="not an answer"):
        desk.resolve(t, "loans")
    assert [o["ticket"] for o in desk.open()] == [t]          # a refused label leaves the ticket open
    desk.resolve(t, _gold(3))
    for bad in (t, "x" * 24, None, 7):
        with pytest.raises(TicketError) as e:
            desk.resolve(bad, "billing")
        assert str(e.value) == UNKNOWN
    assert desk.open() == []
    assert len(_records(store, "dispatch")) == 1


def test_expiry_is_a_recorded_loss_for_the_certificate_and_the_detector(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    clock = Clock()
    desk, sp, det = _desk(store, clock=clock)
    _run(desk, range(300))                                    # certified: alone answers, audits drawn by the coin
    assert sp.threshold < 1.0
    # an audit ticket (System 1 would answer alone; the coin drew it) left open, then expired
    case = 300
    while True:
        r = desk.decide({"case": case})
        if r["status"] == "needs_verdict":
            break
        case += 1
    kind = [x["pending"]["kind_hidden"] for _, x in _records(store, "pending")][-1]
    assert kind == "audit"
    missing, dn = sp.state()["missing"], det.n
    clock.t += 601
    late = desk.sweep()
    assert len(late) == 1 and late[0].by == "human" and late[0].answer is None   # not System 1's answer, unchecked
    s = late[0].stream["sample"]
    assert s["sampled"] and s["lost"] and s["label_source"] is None and s["label_of"] is None
    assert sp.state()["missing"] == missing + 1 and det.n == dn + 1          # a loss for both
    assert late[0].stream["detector"]["n"] == det.n
    exp = _records(store, "pending_expired")
    assert len(exp) == 1 and exp[0][1]["pending"]["ticket"] == r["ticket"] and "expired" in exp[0][1]["pending"]["why"]
    with pytest.raises(TicketError):
        desk.resolve(r["ticket"], _gold(case))                # a verdict after expiry is refused: the loss stands
    # an expired escalation (no answer alone) is lost too; the expiry is swept on the next call
    t2 = None
    while t2 is None:
        case += 1
        x = desk.decide({"case": case})
        t2 = x.get("ticket")
    desk.expire(t2, "the agent could not decide")
    last = desk.dispatcher.stored()[-1]
    assert last.pending == t2 and last.by == "human" and "could not decide" in " ".join(last.reasons)
    assert desk.dispatcher.replay_all() == []
    assert desk.dispatcher.replay_stream()["ok"]


def test_a_restart_rebuilds_the_open_tickets_and_the_certificate(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    clock = Clock()
    keep = {5, 40, 120}
    desk, sp, det = _desk(store, clock=clock)
    _, tickets = _run(desk, range(200), keep_open=keep)
    open_before = {o["ticket"] for o in desk.open()}
    assert open_before == {tickets[c]["ticket"] for c in keep if c in tickets} and len(open_before) >= 2
    n_before, state_before = desk.dispatcher.n, sp.state()
    # a new process: the same configuration, the same store
    desk.close()
    desk2, sp2, det2 = _desk(JSONLStorage(tmp_path / "t.jsonl"), clock=clock)
    assert {o["ticket"] for o in desk2.open()} == open_before
    assert desk2.dispatcher.n == n_before and sp2.state() == state_before and det2.state() == det.state()
    # the open tickets resolve after the restart, and the stream goes on
    for c in sorted(keep):
        if c in tickets:
            desk2.resolve(tickets[c]["ticket"], _gold(c))
    r = desk2.decide({"case": 1000})
    assert desk2.dispatcher.n == n_before + 1
    if r["status"] == "needs_verdict":
        desk2.resolve(r["ticket"], _gold(1000))
    # replay of the whole store, across the restart: every decision and the stream, exact
    assert desk2.dispatcher.replay_all() == []
    rep = desk2.dispatcher.replay_stream()
    assert rep["ok"], rep["mismatches"][:3]
    assert rep["decisions"] == n_before + 1 and rep["coin"] == "verified"
    # an open ticket past its ttl at the restart is expired when the desk starts
    case, t = 2000, {}
    while "ticket" not in t:
        case += 1
        t = desk2.decide({"case": case})
    clock.t += 10_000
    desk2.close()
    desk3, _, _ = _desk(JSONLStorage(tmp_path / "t.jsonl"), clock=clock)
    assert desk3.open() == []
    assert _records(store, "pending_expired")[-1][1]["pending"]["ticket"] == t["ticket"]
    assert desk3.dispatcher.stored()[-1].by == "human"
    assert desk3.dispatcher.replay_all() == [] and desk3.dispatcher.replay_stream()["ok"]


def test_tickets_resolved_out_of_order_replay_exactly(tmp_path):
    store = JSONLStorage(tmp_path / "t.jsonl")
    desk, sp, _ = _desk(store)
    pending = []
    for c in range(260):
        r = desk.decide({"case": c})
        if r["status"] == "needs_verdict":
            pending.append((c, r["ticket"]))
        if len(pending) == 3:                                 # three at a time, the newest first
            for cc, t in reversed(pending):
                desk.resolve(t, _gold(cc) if cc % 17 else ("cards" if _gold(cc) == "billing" else "billing"))
            pending = []
    for cc, t in pending:
        desk.resolve(t, _gold(cc))
    d = desk.dispatcher
    assert d.replay_all() == []
    rep = d.replay_stream()
    assert rep["ok"], rep["mismatches"][:3]
    assert any(r.pending for r in d.stored())


# --- MCP
def test_mcp_round_trip_with_the_builtin_server(tmp_path):
    from solvi.serve import Service, mcp_tools, run_builtin
    store = JSONLStorage(tmp_path / "t.jsonl")
    desk, sp, _ = _desk(store)
    svc = Service(agent_oracle=desk)
    assert [t["name"] for t in mcp_tools(svc)] == ["decide", "resolve", "open_tickets", "give_up"]

    def call(i, name, args):
        return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}
    inp = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        call(2, "decide", {"state": {"case": 7}})]) + "\n")
    out = io.StringIO()
    run_builtin(svc, inp, out)
    rs = [json.loads(x) for x in out.getvalue().splitlines()]
    assert {t["name"] for t in rs[0]["result"]["tools"]} == {"decide", "resolve", "open_tickets", "give_up"}
    ticket = rs[1]["result"]["structuredContent"]
    assert ticket["status"] == "needs_verdict" and ticket["input"] == {"case": 7} and not rs[1]["result"]["isError"]
    inp = io.StringIO("\n".join(json.dumps(m) for m in [
        call(3, "resolve", {"ticket": ticket["ticket"], "label": "loans"}),
        call(4, "resolve", {"ticket": ticket["ticket"], "label": _gold(7)}),
        call(5, "resolve", {"ticket": ticket["ticket"], "label": _gold(7)}),
        call(6, "decide", {"state": 5}),
        call(7, "intent", {"case": 1})]) + "\n")
    out = io.StringIO()
    run_builtin(svc, inp, out)
    rs = [json.loads(x) for x in out.getvalue().splitlines()]
    assert rs[0]["result"]["isError"] and "not an answer" in rs[0]["result"]["structuredContent"]["error"]
    ok = rs[1]["result"]["structuredContent"]
    assert ok["status"] == "resolved" and ok["answer"] == _gold(7) and ok["stored_id"]
    assert rs[2]["result"]["isError"] and rs[2]["result"]["structuredContent"]["error"] == UNKNOWN
    assert rs[3]["result"]["isError"] and "JSON object" in rs[3]["result"]["structuredContent"]["error"]
    assert rs[4]["error"]["code"] == -32602                  # the question tools are not served in this mode
    assert sp.n == 1 and desk.dispatcher.replay_stream()["ok"]


DESK_FILE = """
import os, warnings
warnings.simplefilter("ignore")
from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.dispatch import Dispatcher
from solvi.core.store import JSONLStorage
from solvi.experimental.stream import RiskDetector, StreamPromise
from solvi.experimental.stream.tickets import AgentOracle, TicketDesk


def build():
    cat = Catalog()

    @cat.rule("intent")
    def intent(case: int):
        return Decision("billing", {"billing": 0.9, "cards": 0.1}, confidence=0.9)
    s1 = System(cat, [Question("intent", "Which team?", Answer.choice(["billing", "cards"]))])
    sp = StreamPromise(max_risk=0.05, grid=[0.9, 0.5])
    s1.guarantee("intent", promise=sp)
    d = Dispatcher(s1, AgentOracle(s1), audit="adaptive", promise=sp, detector=RiskDetector(0.05), secret="k",
                   storage=JSONLStorage(os.environ["DESK_STORE"]))
    return TicketDesk(d)
"""


def _session(path, store, msgs):
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "DESK_STORE": str(store),
           "PYTHONPATH": os.pathsep.join([src] + [x for x in [os.environ.get("PYTHONPATH")] if x])}
    args = [sys.executable, "-m", "solvi", "serve", f"{path}:build", "--mcp", "--agent-oracle", "--mcp-impl", "builtin"]
    p = subprocess.run(args, input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                       env=env, cwd=str(path.parent), timeout=60)
    assert p.returncode == 0, p.stderr
    return [json.loads(x) for x in p.stdout.splitlines()]


def test_serve_agent_oracle_across_two_processes(tmp_path):
    f = tmp_path / "desk.py"
    f.write_text(textwrap.dedent(DESK_FILE))
    store = tmp_path / "s.jsonl"
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "decide",
                                                                          "arguments": {"state": {"case": 1}}}}
    rs = _session(f, store, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, call])
    assert {t["name"] for t in rs[0]["result"]["tools"]} == {"decide", "resolve", "open_tickets", "give_up"}
    ticket = rs[1]["result"]["structuredContent"]["ticket"]
    # the server stops; a new one on the same store resolves the ticket
    rs = _session(f, store, [{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                              "params": {"name": "resolve", "arguments": {"ticket": ticket, "label": "cards"}}}])
    res = rs[0]["result"]["structuredContent"]
    assert res["status"] == "resolved" and res["answer"] == "cards"
    kinds = [json.loads(x)["kind"] for x in store.read_text().splitlines()]
    assert kinds.count("pending") == 1 and kinds.count("dispatch") == 1
    from solvi.cli import main
    with pytest.raises(SystemExit):
        main(["serve", f"{f}:build", "--agent-oracle"])          # an MCP mode only


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_only_one_active_owner_and_close_allows_resume(tmp_path, backend):
    from solvi.core.store import SQLiteStorage
    factory = JSONLStorage if backend == "jsonl" else SQLiteStorage
    path = tmp_path / ("owner." + backend)
    store = factory(path)
    desk, _, _ = _desk(store)
    ticket = desk.decide({"case": 1})["ticket"]
    before = store.head()
    with pytest.raises(TicketError, match="active owner"):
        _desk(factory(path))
    # A fresh interpreter cannot replay, expire or write through the live owner.
    script = """
import runpy, sys
ns = runpy.run_path(sys.argv[1])
from solvi.core.store import JSONLStorage, SQLiteStorage
factory = JSONLStorage if sys.argv[3] == 'jsonl' else SQLiteStorage
try:
    ns['_desk'](factory(sys.argv[2]))
except ns['TicketError'] as e:
    assert 'active owner' in str(e)
else:
    raise AssertionError('second process acquired ownership')
"""
    proc = subprocess.run([sys.executable, "-c", script, __file__, str(path), backend],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert store.head() == before
    desk.close()
    desk.close()
    for call in (lambda: desk.decide({"case": 2}), lambda: desk.resolve(ticket, _gold(1)),
                 lambda: desk.expire(ticket), desk.sweep, desk.open):
        with pytest.raises(TicketError, match="closed"):
            call()
    with _desk(factory(path))[0] as resumed:
        assert resumed.dispatcher.n == 1
        resumed.resolve(ticket, _gold(1))
        resumed.decide({"case": 2})
        assert resumed.dispatcher.n == 2
        assert resumed.dispatcher.replay_all() == []
    with _desk(factory(path))[0] as again:
        assert again.dispatcher.n == 2


def test_failed_load_releases_owner(tmp_path):
    store = JSONLStorage(tmp_path / "failed.jsonl")
    from unittest.mock import patch
    with patch.object(TicketDesk, "_load", side_effect=ValueError("bad replay")):
        with pytest.raises(ValueError, match="bad replay"):
            _desk(store)
    with _desk(JSONLStorage(store.path))[0] as desk:
        assert desk.open() == []



def test_postgres_ownership_uses_connection_and_prefix_without_network():
    from solvi.core.store import PostgresStorage

    class DB:
        def __init__(self):
            self.calls = []
            self.available = True

        def execute(self, sql, args):
            self.calls.append((sql, args))
            return self

        def fetchone(self):
            return (1234,) if self.calls[-1][0] == "SELECT to_regclass(%s)::oid" else (self.available,)

    db = DB()
    def store(prefix="solvi_"):
        obj = PostgresStorage.__new__(PostgresStorage)
        obj.db, obj.prefix = db, prefix
        return obj

    first, alias = store(), store()
    release = TicketDesk._claim_owner(first)
    with pytest.raises(TicketError, match="active owner"):
        TicketDesk._claim_owner(alias)  # advisory locks are reentrant on the same DB connection
    release()
    TicketDesk._claim_owner(alias)()
    assert [sql for sql, _ in db.calls if "advisory" in sql] == [
        "SELECT pg_try_advisory_lock(%s)", "SELECT pg_advisory_unlock(%s)",
        "SELECT pg_try_advisory_lock(%s)", "SELECT pg_advisory_unlock(%s)"]
    db.available = False
    with pytest.raises(TicketError, match="active owner"):
        TicketDesk._claim_owner(first)
    db.available = True
    TicketDesk._claim_owner(first)()


@pytest.mark.parametrize("same_connection", [False, True])
@pytest.mark.parametrize("prefixes", [("Solvi_", "solvi_"), ("a" * 63 + "x", "a" * 63 + "y")])
def test_postgres_relation_alias_ownership_without_network(same_connection, prefixes):
    from solvi.core.store import PostgresStorage

    # The server's relation lookup is stubbed: these identifiers resolve to one
    # head table. This checks lease identity, not PostgreSQL's identifier parser.
    held = set()
    class DB:
        def execute(self, sql, args):
            if sql == "SELECT to_regclass(%s)::oid":
                assert args[0] in {p + "meta" for p in prefixes}
                self.row = (5678,)
            elif sql == "SELECT pg_try_advisory_lock(%s)":
                self.row = (args[0] not in held,)
                held.add(args[0])
            else:
                assert sql == "SELECT pg_advisory_unlock(%s)"
                held.remove(args[0])
                self.row = (True,)
            return self

        def fetchone(self):
            return self.row

    def store(db, prefix):
        obj = PostgresStorage.__new__(PostgresStorage)
        obj.db, obj.prefix = db, prefix
        return obj

    db = DB()
    first = store(db, prefixes[0])
    alias = store(db if same_connection else DB(), prefixes[1])
    release = TicketDesk._claim_owner(first)
    try:
        with pytest.raises(TicketError, match="active owner"):
            TicketDesk._claim_owner(alias)
    finally:
        release()
    TicketDesk._claim_owner(alias)()
    assert not held


@pytest.mark.parametrize("outcome", ["normal", "validation", "sdk", "service", "disconnect"])
def test_serve_agent_oracle_closes_desk(tmp_path, monkeypatch, outcome):
    import solvi.serve as serve
    from solvi.cli import main

    desk, _, _ = _desk(JSONLStorage(tmp_path / "lifecycle.jsonl"))
    if outcome == "validation":
        monkeypatch.setattr(desk, "resolve", None)
    monkeypatch.setattr(serve, "load_object", lambda spec: desk)
    monkeypatch.setattr(serve, "sdk_available", lambda: False)

    def run(svc, impl):
        if outcome == "disconnect":
            raise BrokenPipeError("client disconnected")
    monkeypatch.setattr(serve, "run_mcp", run)
    if outcome == "service":
        def service(*args, **kwargs):
            raise ValueError("invalid service")
        monkeypatch.setattr(serve, "Service", service)
    args = ["serve", "desk:build", "--mcp", "--agent-oracle", "--mcp-impl",
            "sdk" if outcome == "sdk" else "builtin"]
    expected = {"validation": SystemExit, "sdk": SystemExit, "service": ValueError}.get(outcome)
    if expected:
        with pytest.raises(expected):
            main(args)
    else:
        assert main(args) == (2 if outcome == "disconnect" else 0)
    with pytest.raises(TicketError, match="closed"):
        desk.decide({"case": 1})


def test_serve_agent_oracle_owner_error_is_cli_error(tmp_path, monkeypatch, capsys):
    import solvi.serve as serve
    from solvi.cli import main

    store = JSONLStorage(tmp_path / "serve-owner.jsonl")
    desk, _, _ = _desk(store)
    def load(spec):
        return _desk(store)[0]
    monkeypatch.setattr(serve, "load_object", load)
    try:
        with pytest.raises(SystemExit) as error:
            main(["serve", "desk:build", "--mcp", "--agent-oracle"])
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert "serve --agent-oracle: cannot open desk:build" in stderr
        assert "active owner" in stderr
        assert "Traceback" not in stderr
        assert desk.decide({"case": 1})["status"] == "needs_verdict"
    finally:
        desk.close()


def test_declared_state_includes_external_guarantee_signal_and_can_certify(tmp_path):
    cat = Catalog()

    @cat.rule("intent")
    def intent(pred: str):
        return pred

    system = System(cat, [Question("intent", TEXT, Answer.choice(["billing", "cards"]))])
    promise = StreamPromise(max_risk=0.05, grid=GRID)
    system.guarantee("intent", promise=promise, signal="msp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dispatcher = Dispatcher(system, AgentOracle(system), audit="adaptive", promise=promise,
                                detector=RiskDetector(0.05, arl=200), on_drift="recertify",
                                probation=50, secret="schema", storage=JSONLStorage(tmp_path / "schema.jsonl"))
    desk = TicketDesk(dispatcher, clock=Clock())
    schema = desk.mcp_tools()[0]["inputSchema"]["properties"]["state"]
    assert set(schema["properties"]) == {"pred", "msp"}
    assert set(schema["required"]) == {"pred", "msp"}
    # A client supplies exactly the facts advertised by tools/list.
    values = {"pred": "billing", "msp": 0.99}
    outcomes = []
    for _ in range(300):
        result = desk.call_tool("decide", {"state": {key: values[key] for key in schema["properties"]}})
        outcomes.append(result)
        if result["status"] == "needs_verdict":
            desk.call_tool("resolve", {"ticket": result["ticket"], "label": "billing"})
    assert promise.threshold < 1
    assert any(r["status"] == "decided" and r["by"] == "s1" for r in outcomes)


def test_declared_state_never_offers_internal_guarantee_signals():
    from solvi.core._inputs import input_schema
    from solvi.core.guarantees.guarantee import QuestionGuard

    system = _s1()
    for signal in ("confidence", "probability", "act", "extra:msp", lambda result, vals: 0.99):
        system.guards["intent"] = QuestionGuard("intent", object(), signal=signal)
        assert set(input_schema(system, "intent")["properties"]) == {"case"}


def test_declared_state_uses_inputs_of_computed_guarantee_signal():
    from solvi.core._inputs import input_schema

    system = _s1()

    @system.catalog.fn
    def msp(sensor: float):
        return sensor

    system.guarantee("intent", promise=StreamPromise(max_risk=0.05, grid=GRID), signal="msp")
    schema = input_schema(system, "intent")
    assert set(schema["properties"]) == {"case", "sensor"}
    assert set(schema["required"]) == {"case", "sensor"}


def test_mcp_recovery_and_refusal_are_blind_and_durable(tmp_path):
    store = JSONLStorage(tmp_path / "recover.jsonl")
    clock = Clock()
    desk, _, _ = _desk(store, clock=clock)
    original = desk.call_tool("decide", {"state": {"case": 7}})
    clock.t += 10
    recovered = desk.call_tool("open_tickets", {})["tickets"]
    assert recovered == [{**original, "expires_in_s": 590.0}]
    desk.close()
    desk, _, _ = _desk(store, clock=clock)
    assert desk.call_tool("open_tickets", {})["tickets"] == recovered
    refused = desk.call_tool("give_up", {"ticket": original["ticket"], "why": "cannot determine"})
    assert refused["status"] == "expired"
    assert refused["ticket"] == original["ticket"]
    assert refused["stored_id"]
    assert desk.call_tool("open_tickets", {}) == {"tickets": []}
    expired = _records(store, "pending_expired")[-1][1]
    assert expired["pending"]["why"] == "cannot determine"
    assert _records(store, "dispatch")[-1][1]["by"] == "human"
    with pytest.raises(TicketError, match=UNKNOWN):
        desk.call_tool("give_up", {"ticket": original["ticket"]})
    for args in ({}, {"ticket": []}, {"ticket": "x", "why": 3}):
        with pytest.raises(TicketError):
            desk.call_tool("give_up", args)


def test_mcp_open_tickets_sweeps_and_does_not_deduplicate(tmp_path):
    clock = Clock()
    desk, _, _ = _desk(JSONLStorage(tmp_path / "recover.jsonl"), clock=clock, ttl=5, deduplicate_open=False)
    one = desk.call_tool("decide", {"state": {"case": 7}})
    two = desk.call_tool("decide", {"state": {"case": 7}})
    assert one["ticket"] != two["ticket"]
    assert desk.call_tool("open_tickets", {})["tickets"] == [one, two]
    clock.t += 5
    assert desk.call_tool("open_tickets", {}) == {"tickets": []}
    desk.close()
    for name in ("open_tickets", "give_up"):
        with pytest.raises(TicketError, match="closed"):
            desk.call_tool(name, {"ticket": one["ticket"]})


def test_mcp_recovery_and_refusal_use_normal_tool_error_envelope(tmp_path):
    from solvi.serve import Service, call_tool
    desk, _, _ = _desk(JSONLStorage(tmp_path / "errors.jsonl"))
    svc = Service(agent_oracle=desk)
    issued, bad = call_tool(svc, "decide", {"state": {"case": 1}})
    assert not bad
    recovered, bad = call_tool(svc, "open_tickets", {})
    assert not bad and recovered == {"tickets": [issued]}
    for args in ({}, {"ticket": []}, {"ticket": issued["ticket"], "why": 7}):
        result, bad = call_tool(svc, "give_up", args)
        assert bad and set(result) == {"error"}
    result, bad = call_tool(svc, "give_up", {"ticket": issued["ticket"]})
    assert not bad and result["status"] == "expired"
    result, bad = call_tool(svc, "give_up", {"ticket": issued["ticket"]})
    assert bad and result == {"error": UNKNOWN}


@pytest.mark.parametrize("window", ["before_start", "pending_committed", "intent_committed"])
def test_request_id_recovers_interrupted_save(tmp_path, monkeypatch, window):
    store = JSONLStorage(tmp_path / "requests.jsonl")
    desk, _, _ = _desk(store)
    if window == "before_start":
        monkeypatch.setattr(desk.dispatcher, "start", lambda state: (_ for _ in ()).throw(RuntimeError("timeout")))
    else:
        append = store._append
        def interrupted(record):
            result = append(record)
            if record["kind"] == ("pending" if window == "pending_committed" else "ticket_request"):
                raise RuntimeError("timeout")
            return result
        monkeypatch.setattr(store, "_append", interrupted)
    with pytest.raises(RuntimeError, match="timeout"):
        desk.decide({"case": 3}, request_id="retry")
    monkeypatch.undo()
    recovered, _, _ = _desk(store)
    with pytest.raises(TicketError, match="different state"):
        recovered.decide({"case": 4}, request_id="retry")
    if window != "pending_committed":
        with pytest.raises(TicketError, match="interrupted request"):
            recovered.decide({"case": 4})
    first = recovered.decide({"case": 3}, request_id="retry")
    assert recovered.decide({"case": 3}, request_id="retry") == first
    assert recovered.dispatcher.n == 1
    assert len(list(store.iter(kind="pending"))) == 1
    recovered.close()
    restarted, _, _ = _desk(store)
    assert restarted.decide({"case": 3}, request_id="retry") == first
    restarted.resolve(first["ticket"], _gold(3))
    assert restarted.decide({"case": 3}, request_id="retry") == first
    restarted.close()


def test_request_id_mcp_and_legacy_calls(tmp_path):
    desk, _, _ = _desk(JSONLStorage(tmp_path / "legacy.jsonl"), deduplicate_open=False)
    args = {"state": {"case": 2}, "request_id": "r"}
    result = desk.call_tool("decide", args)
    assert desk.call_tool("decide", args) == result
    assert "request_id" in desk.mcp_tools()[0]["inputSchema"]["properties"]
    assert desk.decide({"case": 2})["ticket"] != result["ticket"]
    assert set(result) == {"status", "ticket", "question", "options", "input", "instructions", "expires_in_s"}
    desk.close()


@pytest.mark.parametrize("interrupted", [False, True])
def test_request_id_direct_dispatch_recovery(tmp_path, monkeypatch, interrupted):
    store = JSONLStorage(tmp_path / "direct.jsonl")
    desk, _, _ = _desk(store)
    monkeypatch.setattr(desk.dispatcher.slow, "not_applicable",
                        lambda *args: desk.dispatcher.slow.lost("test refusal"))
    append = store._append
    def save(record):
        result = append(record)
        if interrupted and record["kind"] == "dispatch":
            raise RuntimeError("timeout after dispatch")
        return result
    monkeypatch.setattr(store, "_append", save)
    if interrupted:
        with pytest.raises(RuntimeError, match="timeout"):
            desk.decide({"case": 5}, request_id="direct")
        expected = None
    else:
        expected = desk.decide({"case": 5}, request_id="direct")
        assert desk.decide({"case": 5}, request_id="direct") == expected
        desk.close()
    monkeypatch.undo()
    recovered, _, _ = _desk(store)
    result = recovered.decide({"case": 5}, request_id="direct")
    assert result["status"] == "decided"
    if expected is not None:
        assert result == expected
    assert recovered.dispatcher.n == 1
    assert len(list(store.iter(kind="dispatch"))) == 1
    recovered.close()


@pytest.mark.parametrize("resolved", [False, True])
def test_redacted_request_intent_preserves_retry(tmp_path, resolved):
    path = tmp_path / "redacted-intent.jsonl"
    store = JSONLStorage(path)
    desk, _, _ = _desk(store)
    expected = desk.decide({"case": 3}, request_id="retry")
    if resolved:
        desk.resolve(expected["ticket"], _gold(3))
    desk.close()
    store.redact(next(store.iter(kind="ticket_request")).id)
    before = path.read_bytes()
    recovered, _, _ = _desk(store)
    assert recovered.decide({"case": 3}, request_id="retry") == expected
    assert recovered.dispatcher.n == 1
    assert len(list(store.iter(kind="pending"))) == 1
    with pytest.raises(TicketError, match="different state"):
        recovered.decide({"case": 4}, request_id="retry")
    assert path.read_bytes() == before
    recovered.close()


@pytest.mark.parametrize("kind", ["pending", "dispatch"])
def test_redacted_request_outcome_fails_closed(tmp_path, monkeypatch, kind):
    path = tmp_path / "redacted-outcome.jsonl"
    store = JSONLStorage(path)
    desk, _, _ = _desk(store)
    if kind == "dispatch":
        monkeypatch.setattr(desk.dispatcher.slow, "not_applicable",
                            lambda *args: desk.dispatcher.slow.lost("test refusal"))
    desk.decide({"case": 3}, request_id="retry")
    desk.close()
    store.redact(next(store.iter(kind=kind)).id)
    before = path.read_bytes()
    with pytest.raises(TicketError, match="outcome was redacted"):
        _desk(store)
    assert path.read_bytes() == before


def test_interrupted_request_refuses_changed_decision_number(tmp_path, monkeypatch):
    path = tmp_path / "changed-number.jsonl"
    store = JSONLStorage(path)
    desk, _, _ = _desk(store)
    monkeypatch.setattr(desk.dispatcher, "start",
                        lambda state: (_ for _ in ()).throw(RuntimeError("timeout")))
    with pytest.raises(RuntimeError, match="timeout"):
        desk.decide({"case": 3}, request_id="retry")
    recovered, _, _ = _desk(store)
    recovered.dispatcher.n = 1
    before = path.read_bytes()
    with pytest.raises(TicketError, match="decision number changed"):
        recovered.decide({"case": 3}, request_id="retry")
    assert path.read_bytes() == before
    assert not list(store.iter(kind="pending"))
    recovered.close()
