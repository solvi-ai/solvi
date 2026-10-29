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


def _desk(store, clock=None, probation=50, ttl=600, arl=200):
    s1 = _s1()
    sp = StreamPromise(max_risk=0.05, grid=GRID)
    s1.guarantee("intent", promise=sp)
    det = RiskDetector(0.05, arl=arl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = Dispatcher(s1, AgentOracle(s1), audit="adaptive", promise=sp, detector=det, on_drift="recertify",
                       probation=probation, secret="k", storage=store)
    return TicketDesk(d, ttl=ttl, clock=clock or Clock()), sp, det


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
    assert [t["name"] for t in mcp_tools(svc)] == ["decide", "resolve"]

    def call(i, name, args):
        return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}
    inp = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        call(2, "decide", {"state": {"case": 7}})]) + "\n")
    out = io.StringIO()
    run_builtin(svc, inp, out)
    rs = [json.loads(x) for x in out.getvalue().splitlines()]
    assert {t["name"] for t in rs[0]["result"]["tools"]} == {"decide", "resolve"}
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
    assert {t["name"] for t in rs[0]["result"]["tools"]} == {"decide", "resolve"}
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
