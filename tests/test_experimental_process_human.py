"""solvi.experimental.process, stage 1b: the channel to a person — HumanChannel (a queue: submit, poll, cancel,
fingerprint), the built-ins MemoryQueue and CallbackChannel, a node of kind "human", the hand-off of a node that
abstains or sends its decision to a person, pending / resume as segments of one run, the person's answer recorded as a
decision with its provenance (docs/experimental.md#process).

Admission (PROCESS_DESIGN §F, S1 with the channel): check_human_channel passes for both built-ins and a foreign
implementation; a pending run is resumed by a process loaded afresh from the same spec and stores; no ticket is lost on
a run of ten units where 2% of the decisions ask a person."""
import json
import random
import warnings
from pathlib import Path

import pytest

from solvi.core.store import JSONLStorage, chronology
from solvi.testing.conformance import ConformanceError, check_human_channel

from test_experimental_process_run import Echo, _unit_rule, _units, claim_systems, clock, store

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import (CallbackChannel, HumanAnswer, HumanChannel, MemoryQueue,
                                            NodeResult, Process, ProcessRunError)


# --- channels: the two built-ins and a foreign one

class FileInbox(HumanChannel):
    """A channel of one's own: each question a JSON file in a folder; a person answers by writing <ticket>.answer.json
    next to it."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.n = len(list(self.folder.glob("*.question.json")))

    def submit(self, request):
        self.n += 1
        t = f"inbox-{self.n:05d}"
        (self.folder / f"{t}.question.json").write_text(json.dumps(request.to_dict()))
        return t

    def poll(self, ticket):
        if not (self.folder / f"{ticket}.question.json").exists():
            raise KeyError(ticket)
        if (self.folder / f"{ticket}.cancelled").exists():
            return None
        a = self.folder / f"{ticket}.answer.json"
        if not a.exists():
            return None
        d = json.loads(a.read_text())
        return HumanAnswer(d["value"], d["reviewer"], d.get("note"))

    def cancel(self, ticket, why):
        (self.folder / f"{ticket}.cancelled").write_text(str(why))

    def fingerprint(self):
        return "FileInbox/1"

    def answer(self, ticket, value, reviewer):
        (self.folder / f"{ticket}.answer.json").write_text(json.dumps({"value": value, "reviewer": reviewer}))


def _by_facts(request):
    return HumanAnswer(f"answer-{request.facts.get('x')}", "callback")


def test_conformance_memory_queue():
    out = check_human_channel(MemoryQueue)
    assert out["lost"] == 0 and out["process"] == "pending"


def test_conformance_callback_channel_that_answers_at_once():
    out = check_human_channel(lambda: CallbackChannel(_by_facts),
                              answer=lambda ch, t, r: f"answer-{r.facts.get('x')}")
    assert out["process"] == "decided"                 # answered at submit: one segment, no pending


def test_conformance_callback_channel_that_answers_later():
    ready = set()                                      # the questions the person has answered (a review service)

    def later(request):
        k = json.dumps(request.facts, sort_keys=True)
        return _by_facts(request) if k in ready else None

    def answer(ch, t, r):
        ready.add(json.dumps(r.facts, sort_keys=True))
        return f"answer-{r.facts.get('x')}"
    out = check_human_channel(lambda: CallbackChannel(later), answer=answer)
    assert out["process"] == "pending"


def test_conformance_foreign_channel(tmp_path):
    k = iter(range(100))
    out = check_human_channel(lambda: FileInbox(tmp_path / f"inbox{next(k)}"),
                              answer=lambda ch, t, r: (ch.answer(t, f"v{r.facts['x']}", "anna"), f"v{r.facts['x']}")[1])
    assert out["lost"] == 0 and out["tickets"] == 20


def test_conformance_catches_a_channel_that_loses_or_reuses_tickets():
    class Reuses(MemoryQueue):
        def submit(self, request):
            super().submit(request)
            self._q["same"] = {"request": request, "answer": None, "cancelled": None}
            return "same"

    with pytest.raises(ConformanceError, match="its own ticket"):
        check_human_channel(Reuses)

    class Forgets(MemoryQueue):
        def poll(self, ticket):
            return None if ticket == next(iter(self._q)) else super().poll(ticket)

    with pytest.raises(ConformanceError, match="lost"):
        check_human_channel(Forgets)

    class NotAChannel:
        def fingerprint(self):
            return "x"
    with pytest.raises(ConformanceError, match="not a HumanChannel"):
        check_human_channel(NotAChannel)


# --- a human node: pending, resume across a fresh process, the record of the answer

def review_process(ps, storage, queue):
    p = Process("claim", storage=storage, human=queue)
    p.node("triage", ps["triage"], reads=["item"], writes={"kind": "answer"})
    p.node("review", kind="human", question="fraud_risk", reads=["item", "amount", "kind", "provenance"],
           writes={"fraud_risk": "answer"})
    p.node("payout", ps["payout"], reads=["amount", "kind", "fraud_risk"], writes={"payout": "answer"}, end=True)
    p.edge("triage", "review").edge("review", "payout")
    return p


def test_a_human_node_is_pending_then_resumed_by_a_fresh_process(tmp_path):
    clk = clock()
    ps = claim_systems(tmp_path, "p", clk)
    q = MemoryQueue()
    p = review_process(ps, store(tmp_path / "process.jsonl", clk), q)
    r = p.run({"item": "car", "amount": 300.0})
    assert r.status == "pending" and r.ticket and r.answer is None and "waiting for a person" in r.why
    assert [st["node"] for st in r.path] == ["triage"]
    (t, req), = q.open()
    assert t == r.ticket and req.node == "review" and req.question == "fraud_risk" and req.reason == "node"
    assert req.facts == {"item": "car", "amount": 300.0, "kind": "auto"} and req.path == ["triage"]
    assert req.provenance["kind"]["from"] == "triage" and req.run == r.run
    assert p.pending() == [{"run": r.run, "node": "review", "ticket": t, "question": "fraud_risk",
                            "asked_at": req.asked_at, "stored_id": r.stored_id}]
    assert p.resume(r.run).status == "pending"         # no answer yet: still pending, nothing written
    n = len(p._open())

    q.answer(t, "low", reviewer="anna", note="known customer")
    with pytest.raises(ValueError, match="already answered"):
        q.answer(t, "high", reviewer="boris")
    # a process loaded afresh from the same spec and stores
    fresh = Process.from_spec(p.spec(), systems={k: ps[k] for k in ("triage", "payout")}, human=q,
                              storage=JSONLStorage(tmp_path / "process.jsonl", clock=clk))
    assert [x["ticket"] for x in fresh.pending()] == [t] and len(fresh._open()) == n
    r2 = fresh.resume(r.run)
    assert r2.status == "decided" and r2.answer == "pay" and r2.run == r.run and r2.continues == r.stored_id
    assert [st["node"] for st in r2.path] == ["review", "payout"] and r2.path[0]["kind"] == "human"
    assert r2.path[0]["ticket"] == t and r2.path[0]["handoff"] is False and r2.path[0]["by"] == "human"
    assert r2.facts["fraud_risk"] == "low" and fresh.pending() == []
    hum = fresh._open().record(r2.path[0]["stored_id"])
    assert hum["kind"] == "human" and hum["ticket"] == t and hum["value"] == "low" and hum["reviewer"] == "anna"
    assert hum["note"] == "known customer" and hum["channel"] == {"channel": "MemoryQueue", "fingerprint": "MemoryQueue/1"}
    assert hum["run"] == {"id": r.run, "parent": r.stored_id}           # the answer follows from the pending segment
    assert r2.provenance["fraud_risk"] == {"from": "review", "decision": hum["id"], "by": "human", "confidence": None,
                                           "trace_hash": None, "reviewer": "anna"}
    cont = fresh._open().record(r2.stored_id)
    assert cont["continues"] == r.stored_id and cont["outcome"]["status"] == "decided"
    # one run across the stores: triage, the pending segment, the answer, payout, the continuation — in order
    stores = {"triage": ps["triage"].storage, "payout": ps["payout"].storage, "process": fresh._open()}
    rows = chronology(stores, run=r.run)
    assert [(x["store"], x["kind"]) for x in rows] == [("triage", "dispatch"), ("process", "process"),
                                                       ("process", "human"), ("payout", "dispatch"),
                                                       ("process", "process")]
    assert [x["parent"] for x in rows[1:]] == [rows[0]["id"], rows[1]["id"], rows[2]["id"], rows[3]["id"]]
    assert fresh._open().verify()["ok"]
    with pytest.raises(ProcessRunError, match="not pending"):
        fresh.resume(r.run)                            # one ticket, one answer
    assert len([s for s in fresh._open().iter("human")]) == 1
    assert "review (human): by human" in fresh.explain(r2)


def test_a_callback_that_answers_at_once_keeps_the_run_in_one_segment(tmp_path):
    clk = clock()
    ps = claim_systems(tmp_path, "p", clk)
    seen = []
    ch = CallbackChannel(lambda req: seen.append(req) or ("high" if req.facts["amount"] > 1000 else "low"),
                         reviewer="desk")
    p = review_process(ps, store(tmp_path / "process.jsonl", clk), ch)
    r = p.run({"item": "roof", "amount": 2500.0})
    assert r.status == "decided" and r.answer == "refuse" and r.continues is None
    assert [st["node"] for st in r.path] == ["triage", "review", "payout"] and len(seen) == 1
    hum = next(p._open().iter("human")).data
    assert hum["value"] == "high" and hum["reviewer"] == "desk" and hum["run"]["parent"] == r.path[0]["stored_id"]
    assert len(p.records()) == 1 and p.pending() == []


# --- the hand-off: a node that abstains or sends its decision to a person

class Abstains(Echo):
    def __init__(self, status="abstain"):
        super().__init__()
        self.status = status

    def run(self, facts):
        self.given.append(facts)
        return NodeResult(None, {"risk": None}, "s1", self.status, None)


@pytest.mark.parametrize("status", ["abstain", "human"])
def test_a_node_without_an_edge_for_its_status_hands_its_decision_to_the_person(tmp_path, status):
    q = MemoryQueue()
    p = Process("h", storage=tmp_path / "process.jsonl", human=q)
    p.node("risk", Abstains(status), kind="system", question="risk", reads=["x"], writes={"risk": "answer:risk"})
    p.node("end", Echo(), kind="system", reads=["risk"], writes={"done": "answer"}, end=True)
    p.edge("risk", "end", when={"status": "ok"})
    r = p.run({"x": 1})
    assert r.status == "pending" and r.path[0]["status"] == status
    (t, req), = q.open()
    assert req.reason == status and req.node == "risk" and req.question == "risk" and req.facts == {"x": 1}
    q.answer(t, "low", reviewer="anna")
    r2 = p.resume(r)
    assert r2.status == "decided" and r2.facts["risk"] == "low" and r2.path[0]["handoff"] is True
    assert r2.path[0]["status"] == "ok" and r2.path[0]["edge"]["to"] == "end"   # answered: the edge for "ok" holds
    assert p.nodes["end"] and r2.provenance["risk"]["by"] == "human"


def test_a_declined_question_stops_the_run_and_a_cancelled_one_closes_it(tmp_path):
    q = MemoryQueue()
    p = Process("h", storage=tmp_path / "process.jsonl", human=q)
    p.node("ask", kind="human", question="ok", writes={"ok": "answer"}, end=True)
    a, b = p.run({"x": 1}), p.run({"x": 2})
    q.decline(a.ticket, "deadline passed")
    ra = p.resume(a.run)
    assert ra.status == "abstain" and "deadline passed" in ra.why and "ok" not in ra.facts
    rb = p.cancel(b.run, "case closed")
    assert rb.status == "cancelled" and q.poll(b.ticket) is None and p.pending() == []
    hums = [s.data for s in p._open().iter("human")]
    assert [h.get("declined") for h in hums] == [True, None] and hums[1]["cancelled"] is True
    with pytest.raises(ProcessRunError, match="not pending"):
        p.resume(b.run)
    with pytest.raises(ValueError, match="cancelled"):
        q.answer(b.ticket, "yes", reviewer="anna")
    assert p._open().verify()["ok"]


def test_what_a_question_to_a_person_needs(tmp_path):
    p = Process("h", storage=tmp_path / "p.jsonl")
    p.node("ask", kind="human", question="ok", writes={"ok": "answer"}, end=True)
    with pytest.raises(ProcessRunError, match="no channel"):
        p.run({})
    nostore = Process("h", human=MemoryQueue()).node("ask", kind="human", question="ok", writes={"ok": "answer"},
                                                     end=True)
    with pytest.raises(ProcessRunError, match="storage"):
        nostore.run({})
    q = MemoryQueue()
    p.human = q
    r = p.run({})
    other = Process.from_spec(p.spec(), human=CallbackChannel(_by_facts), storage=tmp_path / "p.jsonl", strict=False)
    with pytest.raises(ProcessRunError, match="not through this channel"):
        other.resume(r.run)
    changed = Process("h", storage=tmp_path / "p.jsonl", human=q)
    changed.node("ask", kind="human", question="ok", writes={"ok": "answer", "who": "by"}, end=True)
    with pytest.raises(ProcessRunError, match="process spec"):
        changed.resume(r.run)
    q.answer(r.ticket, "yes", reviewer="")
    with pytest.raises(ProcessRunError, match="no reviewer"):
        p.resume(r.run)


def test_a_recorded_answer_stands_when_the_run_broke_after_it(tmp_path):
    q = MemoryQueue()
    p = Process("h", storage=tmp_path / "p.jsonl", human=q)
    p.node("ask", kind="human", question="ok", writes={"ok": "answer"}, end=True)
    r = p.run({})
    q.answer(r.ticket, "yes", reviewer="anna")
    go = p._go
    p._go = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash after the answer was recorded"))
    with pytest.raises(RuntimeError):
        p.resume(r.run)
    p._go = go
    assert len(list(p._open().iter("human"))) == 1 and [x["ticket"] for x in p.pending()] == [r.ticket]
    r2 = p.resume(r.run)
    assert r2.status == "decided" and r2.answer == "yes" and len(list(p._open().iter("human"))) == 1


def test_a_verifier_binds_the_answer_to_who_gave_it(tmp_path):
    q = MemoryQueue()
    p = Process("h", storage=tmp_path / "p.jsonl", human=q,
                user_verifier=lambda ans, req: {"tok-anna": "anna"}.get(ans.token))
    p.node("ask", kind="human", question="ok", writes={"ok": "answer"}, end=True)
    r = p.run({})
    q.answer(r.ticket, "yes", reviewer="anna", token="forged")
    with pytest.raises(PermissionError, match="did not verify"):
        p.resume(r.run)
    r = p.run({})
    q.answer(r.ticket, "yes", reviewer="anna", token="tok-anna")
    done = p.resume(r.run)
    rec = p._open().record(done.path[0]["stored_id"])
    assert done.status == "decided" and rec["verified"] == "anna" and "tok-anna" not in json.dumps(rec)


# --- S1 admission: E2 at N=10 with 2% of the decisions asking a person — no ticket lost

def test_e2_ten_units_with_two_percent_asking_a_person_lose_no_ticket(tmp_path):
    from solvi.experimental.process import DecisionNode
    n, rounds, runs = 10, 3, 30
    clk = clock()
    systems = _units(tmp_path, clk, n)
    rng = random.Random(7)
    asked = []

    class Sometimes(DecisionNode):                     # the unit decides; 2% of its decisions go to a person
        def run(self, facts):                          # (not the last unit's: its loop edge holds always)
            res = super().run(facts)
            if rng.random() < 0.02 and not self.spec.get("end"):
                asked.append(res.stored_id)
                res.status = "human"
            return res

    q = MemoryQueue()
    pst = store(tmp_path / "process.jsonl", clk)
    p = Process("archipelagos", storage=pst, human=q)
    for i in range(n):
        u = f"u{i}"
        p.node(u, systems[u], reads=[f"res_{i}", f"act_{(i - 1) % n}"], writes={f"act_{i}": "answer"},
               end=i == n - 1)
    p.systems = {u: Sometimes(s, p.nodes[u]) for u, s in p.systems.items()}
    for i in range(n - 1):
        p.edge(f"u{i}", f"u{i + 1}", when={"status": "ok"})
    p.edge(f"u{n - 1}", "u0", loop=True, max_iter=rounds - 1)

    def person(t, req):                                # answers with what the unit's rule says
        i = int(req.node[1:])
        return _unit_rule(i, (i - 1) % n)(req.facts[f"res_{i}"], req.facts[f"act_{(i - 1) % n}"])

    gives, final, tickets = {}, {}, []
    for k in range(runs):
        for t, req in q.open():                        # the person answers the questions of the round before
            q.answer(t, person(t, req), reviewer="sim")
        for x in p.pending():
            r = p.resume(x["run"])
            final[x["run"]] = r
            tickets += [r.ticket] if r.status == "pending" else []
        given = {f"res_{i}": rng.randint(0, 9) for i in range(n)} | {f"act_{n - 1}": "hold"}
        r = p.run(given, run=f"e2-{k}")
        gives[r.run], final[r.run] = given, r
        tickets += [r.ticket] if r.status == "pending" else []
    while p.pending():                                 # drain
        for t, req in q.open():
            q.answer(t, person(t, req), reviewer="sim")
        for x in p.pending():
            r = p.resume(x["run"])
            final[x["run"]] = r
            tickets += [r.ticket] if r.status == "pending" else []
    assert len(asked) >= 5 and len(asked) / (runs * n * rounds) < 0.05
    assert len(tickets) == len(asked) == len(set(tickets))
    assert q.open() == [] and p.pending() == []
    hums = [s.data for s in pst.iter("human")]
    assert sorted(h["ticket"] for h in hums) == sorted(tickets)          # every ticket answered once, none lost
    segs = [s for s in pst.iter("process")]
    assert sum(s.data["outcome"]["status"] == "pending" for s in segs) == len(tickets)
    conts = [s.data["continues"] for s in segs if s.data["continues"]]
    assert len(conts) == len(set(conts)) == len(tickets)                 # each pending segment continued once
    for run, given in gives.items():
        r = final[run]
        assert r.status == "decided", (run, r.status, r.why)
        acts = {f"act_{n - 1}": "hold"}
        for _ in range(rounds):
            for i in range(n):
                acts[f"act_{i}"] = _unit_rule(i, (i - 1) % n)(given[f"res_{i}"], acts[f"act_{(i - 1) % n}"])
        assert {f: r.facts[f] for f in acts} == acts
        rows = chronology({f"u{i}": systems[f"u{i}"].storage for i in range(n)} | {"process": pst}, run=run)
        assert sum(x["store"] != "process" for x in rows) == n * rounds     # every node decision of the run, once
    for st in [pst] + [s.storage for s in systems.values()]:
        assert st.verify()["ok"]
