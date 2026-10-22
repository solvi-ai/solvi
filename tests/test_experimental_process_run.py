"""solvi.experimental.process, stage 1a: running a process — the node runners of the three kinds of system, one run
across the node stores (run_context with the previous node's decision as parent), the "process" index record,
explain, and the hooks left for the channel to a person (docs/experimental.md#process).

Admission (PROCESS_DESIGN §F, S1 without the channel): the node decisions are the ones calling the systems by hand in
the same order makes (record for record, timings aside); chronology(run=) gives the recorded path; ten systems that
exchange facts run end to end with no record lost."""
import itertools
import random
import warnings

import pytest

import solvi
from solvi import Answer, Catalog, Question, System
from solvi.core.runtime import trace_hash
from solvi.core.store import JSONLStorage, chronology, run_context
from solvi.solutions.guard import Guard

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import (NodeResult, NodeRunner, Process, ProcessRunError, ProcessSpecError,
                                            fingerprint_of)


def clock():
    """A counting clock: stores that share one are in true order by time; two sets of stores written in the same order,
    each with its own, hold the same times."""
    t = itertools.count(1_000)
    return lambda: float(next(t))


def store(path, clk=None):
    return JSONLStorage(path, clock=clk or clock())


# --- three solvi.build systems over a claim

KIND = Question("kind", "What kind of claim?", Answer.choice(["auto", "property", "glass"]))
RISK = Question("fraud_risk", "Fraud risk?", Answer.choice(["low", "high"]))
PAY = Question("payout", "Pay?", Answer.choice(["pay", "refuse"]))


def _kind(item: str) -> str:
    return "glass" if item == "windscreen" else ("auto" if item in ("car", "bumper") else "property")


def _risk(amount: float, kind: str) -> str:
    return "high" if amount > (500 if kind == "auto" else 2_000) else "low"


def _pay(amount: float, kind: str, fraud_risk: str) -> str:
    return "refuse" if fraud_risk == "high" or amount > 5_000 else "pay"


def _glass(amount: float) -> str:
    return "pay" if amount <= 1_000 else "refuse"


ITEMS = ["car", "bumper", "roof", "door", "windscreen"]


def _claims(n, seed):
    rng = random.Random(seed)
    return [{"item": rng.choice(ITEMS), "amount": float(rng.randint(10, 6_000))} for _ in range(n)]


def _catalog(q, fn):
    cat = Catalog()
    cat.rule(q)(fn)
    return cat


def claim_systems(tmp, tag, clk):
    """triage, fraud, payout and glass built on examples labelled by their own rules, each with its own store (the
    stores share the clock clk)."""
    ex = _claims(150, 1)
    tri = solvi.build(KIND, [({"item": c["item"]}, _kind(c["item"])) for c in ex], catalog=_catalog("kind", _kind),
                      max_risk=0.05, storage=store(tmp / f"{tag}_triage.jsonl", clk))
    fr_ex = [({"amount": c["amount"], "kind": _kind(c["item"])}, _risk(c["amount"], _kind(c["item"]))) for c in ex]
    fraud = solvi.build(RISK, fr_ex, catalog=_catalog("fraud_risk", _risk), max_risk=0.05,
                        storage=store(tmp / f"{tag}_fraud.jsonl", clk))
    pay_ex = [({"amount": c["amount"], "kind": _kind(c["item"]), "fraud_risk": _risk(c["amount"], _kind(c["item"]))},
               _pay(c["amount"], _kind(c["item"]), _risk(c["amount"], _kind(c["item"])))) for c in ex]
    payout = solvi.build(PAY, pay_ex, catalog=_catalog("payout", _pay), max_risk=0.05,
                         storage=store(tmp / f"{tag}_payout.jsonl", clk))
    glass = solvi.build(PAY, [({"amount": c["amount"]}, _glass(c["amount"])) for c in ex],
                        catalog=_catalog("payout", _glass), max_risk=0.05,
                        storage=store(tmp / f"{tag}_glass.jsonl", clk))
    return {"triage": tri, "fraud": fraud, "payout": payout, "glass": glass}


def claim_process(systems, storage=None):
    p = Process("claim", storage=storage, on_abstain="abstain", on_human="abstain")
    p.node("triage", systems["triage"], reads=["item"], writes={"kind": "answer"})
    p.node("fraud", systems["fraud"], reads=["amount", "kind"], writes={"fraud_risk": "answer"})
    p.node("payout", systems["payout"], reads=["amount", "kind", "fraud_risk"], writes={"payout": "answer"}, end=True)
    p.node("glass", systems["glass"], reads=["amount"], writes={"payout": "answer"}, end=True)
    p.edge("triage", "fraud", when={"answer": "kind", "in": ["auto", "property"]})
    p.edge("triage", "glass", when={"answer": "kind", "eq": "glass"})
    p.edge("fraud", "payout", when={"status": "ok"})
    return p


def by_hand(systems, claim, run):
    """The same process glued by hand: the same order, the same facts, the same run_context chain."""
    out = []
    with run_context(run=run):
        parent = None
        facts = dict(claim)
        for name, reads, fact in (("triage", ["item"], "kind"), ("fraud", ["amount", "kind"], "fraud_risk"),
                                  ("payout", ["amount", "kind", "fraud_risk"], "payout"), ("glass", ["amount"], "payout")):
            if name != "triage" and (name in ("fraud", "payout")) == (facts["kind"] == "glass"):
                continue
            with run_context(parent=parent):
                d = systems[name].ask({r: facts[r] for r in reads if r in facts})
            facts[fact] = d.answer
            parent = d.stored_id
            out.append((name, d))
    return out, facts


TIMING = {"ms", "timings"}
ENVELOPE = {"id", "hash", "prev", "run"}       # hashed over the timings, so they differ with them; run checked apart


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _decision(st, sid):
    return {k: v for k, v in _strip(st.record(sid)).items() if k not in ENVELOPE}


# --- S1 admission: parity with the systems glued by hand

def test_node_decisions_are_the_ones_glued_by_hand(tmp_path):
    pc, hc = clock(), clock()
    ps, hs = claim_systems(tmp_path, "p", pc), claim_systems(tmp_path, "h", hc)
    p = claim_process(ps, storage=store(tmp_path / "process.jsonl", pc))
    claims = _claims(40, 7)
    seen_paths = set()
    for i, c in enumerate(claims):
        r = p.run(c, run=f"r{i}")
        hand, facts = by_hand(hs, c, f"r{i}")
        hc()                                          # the process record's tick: the clocks stay in step
        assert [st["node"] for st in r.path] == [n for n, _ in hand]
        seen_paths.add(tuple(st["node"] for st in r.path))
        kind = _kind(c["item"])
        right = _glass(c["amount"]) if kind == "glass" else _pay(c["amount"], kind, _risk(c["amount"], kind))
        assert r.status == "decided" and r.answer == facts["payout"] == right
        for st, (name, d) in zip(r.path, hand):
            mine = ps[name].storage
            assert _decision(mine, st["stored_id"]) == _decision(hs[name].storage, d.stored_id)   # the same decision
            assert st["trace_hash"] == trace_hash(d.s1)
            assert r.nodes[name].answer == d.answer and r.nodes[name].by == d.by
        # the run field: the same run, and the parent is the previous node's decision in both
        mine = [ps[st["node"]].storage.record(st["stored_id"])["run"] for st in r.path]
        theirs = [hs[n].storage.record(d.stored_id)["run"] for n, d in hand]
        assert [m["id"] for m in mine] == [t["id"] for t in theirs] == [f"r{i}"] * len(hand)
        assert mine[0]["parent"] is None and theirs[0]["parent"] is None
        assert [m["parent"] for m in mine[1:]] == [st["stored_id"] for st in r.path[:-1]]
        assert [t["parent"] for t in theirs[1:]] == [d.stored_id for _, d in hand[:-1]]
    assert seen_paths == {("triage", "fraud", "payout"), ("triage", "glass")}       # both branches were taken
    for st in [s.storage for s in ps.values()] + [p._store]:
        assert st.verify()["ok"]


def test_a_system_and_a_guard_are_nodes_with_the_same_decisions_as_by_hand(tmp_path):
    def parts(tag):
        cat = Catalog()

        @cat.rule("size")
        def size(amount: float) -> str:
            return "big" if amount > 100 else "small"
        sysm = System(cat, [Question("size", "Size?", Answer.choice(["small", "big"]))],
                      storage=store(tmp_path / f"{tag}_sys.jsonl"))
        g = Guard(storage=store(tmp_path / f"{tag}_guard.jsonl"), fact_names=["size"])

        @g.tool
        def pay(amount: float) -> str:
            """Pay."""
            return f"paid {amount}"

        @g.policy("pay", on_fail="escalate")
        def small_only(size: str) -> bool:
            return size == "small"
        return sysm, g

    (s1, g1), (s2, g2) = parts("p"), parts("h")
    p = Process("pay", on_human="abstain")
    p.node("size", s1, reads=["amount"], writes={"size": "answer"})
    p.node("guard", g1, reads=["call", "size"], writes={"outcome": "answer", "paid": "answer:result"}, end=True)
    p.edge("size", "guard")
    assert p.spec()["nodes"]["guard"]["kind"] == "guard"
    for i, amount in enumerate([30.0, 500.0, 70.0]):
        call = {"name": "pay", "arguments": {"amount": amount}}
        r = p.run({"amount": amount, "call": call}, run=f"g{i}")
        with run_context(run=f"g{i}"):
            resp = s2.ask({"amount": amount})
            with run_context(parent=resp.stored_id):
                d = g2.call(call, context=None, facts={"size": resp["size"].answer})
        a, b = r.path
        assert _decision(s1.storage, a["stored_id"]) == _decision(s2.storage, resp.stored_id)
        assert _decision(g1.storage, b["stored_id"]) == _decision(g2.storage, d.stored_id)
        assert g1.storage.record(b["stored_id"])["run"] == {"id": f"g{i}", "parent": a["stored_id"]}
        if amount <= 100:
            assert r.status == "decided" and r.answer == "allow" and r.facts["paid"] == f"paid {amount}"
        else:                                         # an escalation is a person's decision: on_human="abstain" stops
            assert r.status == "abstain" and r.nodes["guard"].status == "human" and "person" in r.why


# --- S1 admission: chronology(run=) gives the recorded path

def test_chronology_of_a_run_is_the_recorded_path(tmp_path):
    clk = clock()
    ps = claim_systems(tmp_path, "p", clk)
    pst = store(tmp_path / "process.jsonl", clk)
    p = claim_process(ps, storage=pst)
    for i, c in enumerate(_claims(12, 3)):
        p.run(c, run=f"r{i}")                         # other runs interleave in the same stores
    stores = {n: s.storage for n, s in ps.items()} | {"process": pst}
    for rec in p.records():
        run = rec.run
        rows = chronology(stores, run=run)
        nodes = [x for x in rows if x["kind"] != "process"]
        path = rec.data["path"]
        assert [(x["store"], x["id"]) for x in nodes] == [(st["node"], st["stored_id"]) for st in path]
        assert [x["parent"] for x in nodes] == [None] + [st["stored_id"] for st in path[:-1]]
        assert rows[-1]["kind"] == "process" and rows[-1]["id"] == rec.id and rows[-1]["parent"] == path[-1]["stored_id"]
    assert len(p.records()) == 12 and pst.verify()["ok"]


def test_the_process_record_is_an_index_of_hashes(tmp_path):
    ps = claim_systems(tmp_path, "p", clock())
    p = claim_process(ps, storage=tmp_path / "process.jsonl")
    p.promise(mode="union", risk=0.06)
    r = p.run({"item": "car", "amount": 300.0})
    rec = p._store.record(r.stored_id)
    assert rec["kind"] == "process" and rec["process"] == "claim" and rec["spec_hash"] == p.spec()["hash"]
    assert rec["mode"] == "union" and rec["promise"]["risk"] == 0.06
    assert rec["outcome"]["status"] == "decided" and rec["outcome"]["at"] == "payout"
    step = rec["path"][1]
    assert set(step) == {"node", "kind", "store", "stored_id", "trace_hash", "by", "status", "answers_hash",
                         "facts_in_hash", "facts_out_hash", "knowledge", "cost", "edge"}
    assert step["edge"] == {"to": "payout", "when": {"status": "ok"}, "loop": False}
    assert "300" not in str(rec["path"]) and "car" not in str(rec)        # values stay in the node stores
    text = p.explain(r)
    assert "triage (decision): by s1, ok answered \"auto\"" in text and "→ fraud because kind in" in text
    assert 'decided: "pay" (automatically)' in text and "promise (union): P(a run ends automatically" in text


# --- facts and their provenance

class Echo(NodeRunner):
    """A runner of one's own: answers "seen" and keeps the facts it was given."""
    kind = "system"

    def __init__(self):
        super().__init__(None, {})
        self.given = []

    def fingerprint(self):
        return "echo"

    def run(self, facts):
        self.given.append(facts)
        return NodeResult("seen", {"echo": "seen"}, "s1", "ok", 1.0)


def test_a_reader_of_provenance_gets_the_labels_others_get_plain_values(tmp_path):
    ps = claim_systems(tmp_path, "p", clock())
    plain, labelled = Echo(), Echo()
    p = Process("prov")
    p.node("triage", ps["triage"], reads=["item"], writes={"kind": "answer"})
    p.node("plain", plain, kind="system", reads=["kind", "item"], writes={"x": "answer"})
    p.node("labelled", labelled, kind="system", reads=["kind", "item", "provenance"], writes={"y": "answer"}, end=True)
    p.edge("triage", "plain").edge("plain", "labelled")
    r = p.run({"item": "roof"})
    tri = r.nodes["triage"]
    assert plain.given == [{"kind": "property", "item": "roof"}]
    label = {"from": "triage", "decision": tri.stored_id, "by": "s1", "confidence": tri.confidence,
             "trace_hash": tri.trace_hash}
    assert labelled.given == [{"kind": "property", "item": "roof", "provenance": {"kind": label}}]
    assert r.provenance["kind"] == label and r.provenance["y"]["from"] == "labelled"
    assert r.stored_id is None                         # no storage=: no process record, the node stores hold it all


def test_a_loop_runs_until_its_limit_then_the_next_edge(tmp_path):
    e = Echo()
    p = Process("loop", on_abstain="abstain")
    p.node("a", e, kind="system", writes={"n": "answer"})
    p.node("b", Echo(), kind="system", writes={"m": "answer"}, end=True)
    p.edge("a", "a", loop=True, max_iter=3)
    p.edge("a", "b")
    r = p.run({})
    assert [st["node"] for st in r.path] == ["a", "a", "a", "a", "b"] and len(e.given) == 4
    assert [st["edge"]["loop"] for st in r.path[:-1]] == [True, True, True, False]


# --- a question for a person without a channel (the channel itself: test_experimental_process_human.py)

def test_a_human_node_and_a_handoff_need_a_channel(tmp_path):
    ps = claim_systems(tmp_path, "p", clock())
    p = Process("h")
    p.node("triage", ps["triage"], reads=["item"], writes={"kind": "answer"})
    p.node("review", kind="human", question="kind", writes={"kind": "answer"}, end=True)
    p.edge("triage", "review")
    with pytest.raises(ProcessRunError, match="no channel"):
        p.run({"item": "car"})

    class Abstains(Echo):
        def run(self, facts):
            return NodeResult(None, {}, "s1", "abstain", None)
    q = Process("a").node("x", Abstains(), kind="system", end=True)
    with pytest.raises(ProcessRunError, match="no channel"):
        q.run({})                                      # on_abstain="handoff" (the default) needs the channel
    q.on_abstain = "abstain"
    r = q.run({})
    assert r.status == "abstain" and r.answer is None and "abstained" in r.why


def test_a_merge_of_all_inputs_and_an_unbound_node_are_refused(tmp_path):
    p = Process("m")
    for n in ("a", "b", "c"):
        p.node(n, Echo(), kind="system", writes={n: "answer"}, end=n == "c")
    p.edge("a", "b").edge("a", "c").edge("b", "c").merge("c", "all")
    with pytest.raises(NotImplementedError, match="merges 'all'"):
        p.run({})
    spec = Process("u").node("a", Echo(), kind="system", end=True).spec()
    with pytest.raises(ProcessRunError, match="without a system"):
        Process.from_spec(spec).run({})


def test_a_guard_node_needs_the_call():
    g = Guard()

    @g.tool
    def ping() -> str:
        """Ping."""
        return "pong"
    p = Process("g").node("g", g, reads=["call"], writes={"o": "answer"}, end=True)
    with pytest.raises(ProcessRunError, match="'call'"):
        p.run({})
    assert p.run({"call": {"name": "ping", "arguments": {}}}).answer == "allow"


# --- Guard.fingerprint (public; the process no longer reads the guard's private attributes)

def test_guard_fingerprint_is_public_and_follows_the_checks():
    def make(cap):
        g = Guard()

        @g.tool
        def pay(amount: float) -> str:
            """Pay."""
            return "ok"

        @g.policy("pay")
        def under_cap(amount: float) -> bool:
            return amount <= cap
        return g
    a, b, c = make(10), make(10), make(20)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    assert fingerprint_of(a) == {"guard": a.fingerprint()}
    spec = Process("g").node("g", a, reads=["call"], writes={"o": "answer"}, end=True).spec()
    Process.from_spec(spec, systems={"g": b})
    with pytest.raises(ProcessSpecError, match="not the recorded one"):
        Process.from_spec(spec, systems={"g": c})


# --- S1 admission: E2 at N=10 — units on archipelagos exchanging facts, no record lost

ACTS = ["move", "trade", "hold"]


def _unit_rule(i, prev):
    """unit i decides from its resource res_i and its neighbour's last act (act_<prev>)."""
    src = (f"def act_{i}(res_{i}: int, act_{prev}: str) -> str:\n"
           f"    if res_{i} < 3:\n        return 'move'\n"
           f"    return 'trade' if act_{prev} == 'move' else 'hold'\n")
    env = {}
    exec(src, env)                                     # noqa: S102 — a test's generated rule with its own fact names
    return env[f"act_{i}"]


def _units(tmp, clk, n=10, islands=2):
    """n units on `islands` archipelagos; each archipelago shares a solvi.Knowledge (its weather) the units were built
    with; a unit reads its neighbour's act (plain value) and writes its own."""
    from solvi import Knowledge
    kms = [Knowledge() for _ in range(islands)]
    for k, km in enumerate(kms):
        km.tell({"s": f"island{k}", "r": "weather", "o": "calm"}, source="person", by="test")
    rng = random.Random(0)
    systems = {}
    for i in range(n):
        prev = (i - 1) % n
        rule = _unit_rule(i, prev)
        q = Question(f"act_{i}", "Move, trade or hold?", Answer.choice(ACTS))
        ex = []
        for _ in range(120):
            st = {f"res_{i}": rng.randint(0, 9), f"act_{prev}": rng.choice(ACTS)}
            ex.append((st, rule(**st)))
        cat = Catalog()
        cat.rule(f"act_{i}")(rule)
        systems[f"u{i}"] = solvi.build(q, ex, catalog=cat, max_risk=0.05, storage=store(tmp / f"u{i}.jsonl", clk),
                                       knowledge=kms[i * islands // n])
    return systems


def test_e2_ten_units_exchanging_facts_run_end_to_end_with_no_record_lost(tmp_path):
    n, rounds, runs = 10, 3, 6
    clk = clock()
    systems = _units(tmp_path, clk, n)
    pst = store(tmp_path / "process.jsonl", clk)
    p = Process("archipelagos", storage=pst, on_abstain="abstain", on_human="abstain")
    for i in range(n):
        p.node(f"u{i}", systems[f"u{i}"], reads=[f"res_{i}", f"act_{(i - 1) % n}"], writes={f"act_{i}": "answer"},
               end=i == n - 1)
    for i in range(n - 1):
        p.edge(f"u{i}", f"u{i + 1}", when={"status": "ok"})
    p.edge(f"u{n - 1}", "u0", loop=True, max_iter=rounds - 1)
    rng = random.Random(1)
    results = []
    for k in range(runs):
        given = {f"res_{i}": rng.randint(0, 9) for i in range(n)} | {f"act_{n - 1}": "hold"}
        results.append((given, p.run(given, run=f"e2-{k}")))
    stores = {f"u{i}": systems[f"u{i}"].storage for i in range(n)} | {"process": pst}
    for given, r in results:
        assert r.status == "decided" and len(r.path) == n * rounds
        # the facts went round: recompute every act from the rules in the same order
        acts = {f"act_{n - 1}": "hold"}
        for _ in range(rounds):
            for i in range(n):
                acts[f"act_{i}"] = _unit_rule(i, (i - 1) % n)(given[f"res_{i}"], acts[f"act_{(i - 1) % n}"])
        assert {f: r.facts[f] for f in acts} == acts and r.answer == acts[f"act_{n - 1}"]
        assert all(st["knowledge"] is not None for st in r.path)           # each unit was given its island's snapshot
        assert len({st["knowledge"] for st in r.path}) == 2
        # no record lost: every step is in its store, in this run, chained to the step before
        rows = chronology(stores, run=r.run)
        assert len(rows) == n * rounds + 1
        assert [(x["store"], x["id"]) for x in rows[:-1]] == [(st["node"], st["stored_id"]) for st in r.path]
        assert [x["parent"] for x in rows] == [None] + [st["stored_id"] for st in r.path]
        assert rows[-1] == {**rows[-1], "store": "process", "id": r.stored_id, "kind": "process"}
    for name, st in stores.items():
        assert st.verify()["ok"], name
    assert sum(len([s for s in st.iter(None) if s.run]) for st in stores.values()) == runs * (n * rounds + 1)
