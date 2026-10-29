"""solvi.experimental.process, stage 0: the process as data — the spec's JSON round trip, its hash, diff, render, the
predicate language, loops with a limit and the fingerprint check of from_spec (docs/experimental.md#process)."""
import copy
import json
import random
import warnings

import pytest

from solvi import Answer, Catalog, Question, System

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import (Process, ProcessSpecError, canonical, check_predicate, describe, diff,
                                            holds, render, spec_hash)


class Fake:
    """A system with a fingerprint (stands for a solvi.build result: .dispatcher.config() and .system.fingerprint())."""

    class _D:
        def __init__(self, v):
            self.v = v

        def config(self):
            return f"disp-{self.v}"

    class _S:
        def __init__(self, v):
            self.v = v

        def fingerprint(self):
            return {"catalog": f"cat-{self.v}", "questions": "q", "parts": {"score": f"p-{self.v}"}, "models": {}}

    def __init__(self, question, v="1"):
        self.question, self.dispatcher, self.system = question, self._D(v), self._S(v)


class Channel:
    def __init__(self, v="1"):
        self.v = v

    def fingerprint(self):
        return f"chan-{self.v}"


def claim(systems=None, promise=True):
    s = systems or {"triage": Fake("kind"), "fraud": Fake("fraud_risk"), "payout": Fake("payout")}
    p = Process("claim", human=Channel())
    p.node("triage", s["triage"], reads=["claim"], writes={"kind": "answer"})
    p.node("fraud", s["fraud"], reads=["claim", "kind"], writes={"fraud_risk": "answer"})
    p.node("payout", s["payout"], reads=["claim", "kind", "fraud_risk"], writes={"payout": "answer"}, end=True)
    p.node("review", kind="human", question="fraud_risk", writes={"fraud_risk": "answer"})
    p.edge("triage", "fraud", when={"answer": "kind", "in": ["auto", "property"]})
    p.edge("triage", "payout", when={"answer": "kind", "eq": "glass"})
    p.edge("fraud", "payout", when={"status": "ok"})
    p.edge("fraud", "review", when={"status": "abstain"})
    p.edge("review", "payout")
    p.edge("fraud", "triage", when={"drift": True}, loop=True, max_iter=2)
    p.merge("payout", "any")
    if promise:
        p.promise(mode="union", risk=0.06)
    return p, s


def _shuffled(x, rng):
    """The same JSON value with every dict's keys in a random order (lists keep their order)."""
    if isinstance(x, dict):
        ks = list(x)
        rng.shuffle(ks)
        return {k: _shuffled(x[k], rng) for k in ks}
    if isinstance(x, list):
        return [_shuffled(v, rng) for v in x]
    return x


# --- S0 admission: JSON round trip byte for byte

def test_round_trip_is_byte_identical():
    p, s = claim()
    text = canonical(p.spec())
    q = Process.from_spec(json.loads(text), systems=s, human=Channel())
    assert canonical(q.spec()) == text
    assert canonical(Process.from_spec(text).spec()) == text          # from the text, no systems: unbound, same spec
    assert Process.from_spec(text).unbound == ["fraud", "payout", "triage"]


@pytest.mark.parametrize("promise", [
    dict(mode="union", risk=0.05),
    dict(mode="union", risk=0.05, split={"triage": 0.01, "fraud": 0.02, "payout": 0.02}),
    dict(mode="union", risk=0.06, split=lambda spec: {n: 0.02 for n in ("triage", "fraud", "payout")}),
    dict(mode="end_to_end", risk=0.05, delta=0.1, examples=[({"claim": 1}, "pay")]),
    dict(mode="end_to_end", risk=0.05, delta=0.1, family="per_node_fixed"),
    dict(mode="end_to_end", risk=0.05, delta=0.1, family={"kind": "per_node_fixed", "order": ["fraud", "triage"]}),
    dict(mode="per_node"),
])
def test_round_trip_holds_for_every_promise_mode(promise):
    p, s = claim(promise=False)
    p.promise(**promise)
    text = canonical(p.spec())
    assert canonical(Process.from_spec(text, systems=s).spec()) == text


def test_a_custom_threshold_family_is_recorded_by_fingerprint():
    class Mine:
        def fingerprint(self):
            return "mine-1"
    p, _ = claim(promise=False)
    pr = p.promise(mode="end_to_end", risk=0.05, delta=0.1, family=Mine()).spec()["promise"]
    assert pr["family"] == {"kind": "custom", "name": "Mine", "order": None, "fingerprint": "mine-1"}
    assert canonical(Process.from_spec(p.spec()).spec()) == canonical(p.spec())


def test_equal_split_is_r_over_the_automatic_nodes():
    p, _ = claim()
    sp = p.spec()["promise"]["split"]
    assert sp["kind"] == "equal" and sp["per_node"] == {"triage": 0.02, "fraud": 0.02, "payout": 0.02}
    with pytest.raises(ProcessSpecError, match="over the process risk"):
        claim(promise=False)[0].promise(mode="union", risk=0.05, split={"triage": 0.04, "fraud": 0.02}).spec()
    with pytest.raises(ProcessSpecError, match="not automatic"):
        claim(promise=False)[0].promise(mode="union", risk=0.05, split={"review": 0.01}).spec()


def test_a_real_system_round_trips_and_is_checked():
    def make(threshold):
        cat = Catalog()

        @cat.rule("ok")
        def ok(amount) -> bool:
            return amount < threshold
        return System(cat, [Question("ok", "OK?", Answer.yes_no())])
    p = Process("one").node("a", make(10), reads=["amount"], writes={"ok": "answer:ok"}, end=True)
    spec = p.spec()
    assert spec["nodes"]["a"]["kind"] == "system" and "catalog" in spec["nodes"]["a"]["fingerprint"]["system"]
    assert canonical(Process.from_spec(spec, systems={"a": make(10)}).spec()) == canonical(spec)
    with pytest.raises(ProcessSpecError, match="not the recorded one"):
        Process.from_spec(spec, systems={"a": make(20)})


# --- S0 admission: the hash does not depend on the order of keys

def test_hash_is_stable_under_key_reordering():
    p, s = claim()
    spec = p.spec()
    rng = random.Random(0)
    for _ in range(50):
        sh = _shuffled(spec, rng)
        assert spec_hash(sh) == spec["hash"]
        assert canonical(Process.from_spec(sh, systems=s).spec()) == canonical(spec)
    q, _ = claim()                                    # the same process built again: the same hash
    assert q.spec()["hash"] == spec["hash"]
    assert len(spec["hash"]) == 64


def test_hash_changes_with_content_and_a_tampered_spec_is_refused():
    p, _ = claim()
    spec = p.spec()
    bad = copy.deepcopy(spec)
    bad["edges"][0]["when"]["in"] = ["auto"]
    assert spec_hash(bad) != spec["hash"]
    with pytest.raises(ProcessSpecError, match="does not match its content"):
        Process.from_spec(bad)
    Process.from_spec(bad, strict=False)              # loads
    order = copy.deepcopy(spec)                       # the order of edges is the order they are tried: it counts
    order["edges"][0], order["edges"][1] = order["edges"][1], order["edges"][0]
    assert spec_hash(order) != spec["hash"]


# --- S0 admission: diff names every change (20 synthetic pairs)

def _mutations():
    """20 (name, mutate(process builder kwargs → Process), expected {(path, change)}) single or compound changes."""
    def edit_spec(f):
        def go(spec):
            spec = copy.deepcopy(spec)
            f(spec)
            spec.pop("hash")
            return spec
        return go

    def n(spec, node):
        return spec["nodes"][node]

    def e(spec, i):
        return spec["edges"][i]
    return [
        ("predicate value", edit_spec(lambda s: e(s, 0)["when"].update({"in": ["auto"]})),
         {("edges.triage -> fraud.when", "changed")}),
        ("predicate operator", edit_spec(lambda s: e(s, 1).update(when={"answer": "kind", "in": ["glass"]})),
         {("edges.triage -> payout.when", "changed")}),
        ("predicate removed", edit_spec(lambda s: e(s, 2).update(when=None)), {("edges.fraud -> payout.when", "changed")}),
        ("max_iter", edit_spec(lambda s: e(s, 5).update(max_iter=3)), {("edges.fraud -> triage.max_iter", "changed")}),
        ("edge removed", edit_spec(lambda s: s["edges"].pop(4)), {("edges.review -> payout", "removed")}),
        ("edge added", edit_spec(lambda s: s["edges"].append({"from": "triage", "to": "review", "when": {"by": "s2"}})),
         {("edges.triage -> review", "added")}),
        ("second edge between two nodes", edit_spec(lambda s: s["edges"].append(
            {"from": "fraud", "to": "payout", "when": {"confidence_below": 0.3}})),
         {("edges.fraud -> payout #2", "added")}),
        ("edges reordered", edit_spec(lambda s: s["edges"].insert(0, s["edges"].pop(1))),
         {("edges", "reordered")}),
        ("node added", edit_spec(lambda s: s["nodes"].update(audit={"kind": "human", "question": "q", "reads": [],
                                                                    "writes": {}})),
         {("nodes.audit", "added")}),
        ("node removed", edit_spec(lambda s: s["nodes"].pop("review")), {("nodes.review", "removed")}),
        ("reads", edit_spec(lambda s: n(s, "fraud")["reads"].append("history")), {("nodes.fraud.reads", "changed")}),
        ("writes", edit_spec(lambda s: n(s, "fraud")["writes"].update(fraud_conf="confidence")),
         {("nodes.fraud.writes.fraud_conf", "added")}),
        ("question", edit_spec(lambda s: n(s, "review").update(question="fraud")),
         {("nodes.review.question", "changed")}),
        ("fingerprint of one part", edit_spec(lambda s: n(s, "fraud")["fingerprint"]["system"]["parts"].update(score="x")),
         {("nodes.fraud.fingerprint.system.parts.score", "changed")}),
        ("dispatcher and catalog", edit_spec(lambda s: n(s, "triage")["fingerprint"].update(
            dispatcher="d2", system=dict(n(s, "triage")["fingerprint"]["system"], catalog="c2"))),
         {("nodes.triage.fingerprint.dispatcher", "changed"), ("nodes.triage.fingerprint.system.catalog", "changed")}),
        ("merge", edit_spec(lambda s: s["merge"].update(payout="all")), {("merge.payout", "changed")}),
        ("promise risk and split", edit_spec(lambda s: (s["promise"].update(risk=0.03),
                                                        s["promise"]["split"]["per_node"].update(triage=0.01,
                                                                                                 fraud=0.01,
                                                                                                 payout=0.01))),
         {("promise.risk", "changed"), ("promise.split.per_node.triage", "changed"),
          ("promise.split.per_node.fraud", "changed"), ("promise.split.per_node.payout", "changed")}),
        ("promise mode", edit_spec(lambda s: s.update(promise={**s["promise"], "mode": "per_node", "risk": None,
                                                                "split": None})),
         {("promise.mode", "changed"), ("promise.risk", "changed"), ("promise.split", "changed")}),
        ("channel and on_abstain", edit_spec(lambda s: (s["human"].update(fingerprint="chan-2"),
                                                        s.update(on_abstain="abstain"))),
         {("human.fingerprint", "changed"), ("on_abstain", "changed")}),
        ("name, end and store", edit_spec(lambda s: (s.update(name="claim2", end=["payout", "review"]),
                                                     n(s, "review").update(end=True),
                                                     n(s, "payout").update(store="st/p.jsonl"))),
         {("name", "changed"), ("end", "changed"), ("nodes.review.end", "added"), ("nodes.payout.store", "added")}),
    ]


def test_diff_names_every_change_in_20_pairs():
    p, _ = claim()
    base = p.spec()
    muts = _mutations()
    assert len(muts) == 20
    for name, mutate, expected in muts:
        other = mutate(base)
        got = {(c["path"], c["change"]) for c in diff(base, other)}
        assert got == expected, name
        back = {(c["path"], c["change"]) for c in diff(other, base)}
        flip = {"added": "removed", "removed": "added"}
        assert back == {(pth, flip.get(ch, ch)) for pth, ch in expected}, name
        assert all(c["say"] for c in diff(base, other))
    assert diff(base, _shuffled(base, random.Random(1))) == []
    assert p.diff(claim()[0]) == []


def test_diff_of_a_changed_system_names_the_part():
    p, _ = claim()
    q, _ = claim({"triage": Fake("kind"), "fraud": Fake("fraud_risk", v="2"), "payout": Fake("payout")})
    paths = {c["path"] for c in p.diff(q)}
    assert paths == {"nodes.fraud.fingerprint.dispatcher", "nodes.fraud.fingerprint.system.catalog",
                     "nodes.fraud.fingerprint.system.parts.score"}
    say = [c["say"] for c in p.diff(claim(promise=False)[0].promise("union", 0.06).edge("triage", "review"))]
    assert any("edges.triage -> review added" in x for x in say)


# --- S0 admission: a cycle without max_iter is refused

def test_a_cycle_without_max_iter_is_refused():
    p = Process("c").node("a", kind="human", question="x").node("b", kind="human", question="y", end=True)
    p.edge("a", "b")
    with pytest.raises(ProcessSpecError, match="cycle without max_iter: a -> b -> a"):
        p.edge("b", "a")
    assert len(p.edges) == 1                           # the refused edge is not kept
    with pytest.raises(ValueError, match="needs max_iter"):
        p.edge("b", "a", loop=True)
    with pytest.raises(ValueError, match="max_iter is an integer"):
        p.edge("b", "a", loop=True, max_iter=0)
    with pytest.raises(ValueError, match="for a loop edge"):
        p.edge("b", "a", max_iter=2)
    with pytest.raises(ProcessSpecError, match="cycle without max_iter: a -> a"):
        p.edge("a", "a")
    p.edge("b", "a", loop=True, max_iter=2)            # with a limit: fine
    spec = p.spec()
    loopless = copy.deepcopy(spec)                     # a spec whose loop edge lost its limit
    del loopless["edges"][1]["loop"], loopless["edges"][1]["max_iter"]
    loopless.pop("hash")
    with pytest.raises(ProcessSpecError, match="cycle without max_iter"):
        Process.from_spec(loopless)
    nolimit = copy.deepcopy(spec)
    nolimit["edges"][1]["max_iter"] = None
    nolimit.pop("hash")
    with pytest.raises(ProcessSpecError, match="needs max_iter"):
        Process.from_spec(nolimit)
    longer = Process("l")                              # a longer cycle, closed by the third edge
    for x in "abc":
        longer.node(x, kind="human", question=x, end=x == "c")
    longer.edge("a", "b").edge("b", "c")
    with pytest.raises(ProcessSpecError, match="a -> b -> c -> a"):
        longer.edge("c", "a")


# --- the predicate language

def test_predicates_are_checked_data():
    for good in [None, {"answer": "kind", "eq": "glass"}, {"answer": "x", "in": [1, 2]}, {"answer": "x", "lt": 3},
                 {"status": "abstain"}, {"by": "s2"}, {"confidence_below": 0.4}, {"budget_over": True},
                 {"drift": False}, {"and": [{"status": "ok"}, {"not": {"by": "human"}}]},
                 {"or": [{"answer": "x", "gt": 1}, {"drift": True}]}]:
        assert check_predicate(good) is good
    for bad, msg in [(lambda o: True, "non-empty dict"), ({}, "non-empty dict"), ({"answer": "x"}, "exactly one"),
                     ({"answer": "x", "eq": 1, "in": [1]}, "exactly one"), ({"answer": "x", "in": "ab"}, "takes a list"),
                     ({"answer": "x", "lt": "3"}, "a number"), ({"status": "maybe"}, "status is one of"),
                     ({"by": "s3"}, "by is one of"), ({"confidence_below": 2}, r"in \[0, 1\]"),
                     ({"drift": "yes"}, "true or false"), ({"and": []}, "non-empty list"),
                     ({"or": [{"status": "ok"}, {"foo": 1}]}, r"when.or\[1\]: unknown predicate"),
                     ({"status": "ok", "by": "s1"}, "one form"), ({"not": None}, "null")]:
        with pytest.raises(ProcessSpecError, match=msg):
            check_predicate(bad)
    with pytest.raises(ProcessSpecError, match="edge a -> a.when"):
        Process("p").node("a", kind="human", question="q").edge("a", "a", when={"status": "late"})


def test_holds_and_describe():
    out = {"facts": {"kind": "glass", "amount": 120}, "status": "ok", "by": "s1", "confidence": 0.3, "drift": False}
    assert holds({"answer": "kind", "eq": "glass"}, out) and holds({"answer": "kind", "in": ["glass", "auto"]}, out)
    assert holds({"answer": "amount", "gt": 100}, out) and not holds({"answer": "amount", "lt": 100}, out)
    assert not holds({"answer": "missing", "eq": None}, out) and not holds({"answer": "kind", "lt": 3}, out)
    assert holds({"and": [{"status": "ok"}, {"confidence_below": 0.5}, {"not": {"by": "human"}}]}, out)
    assert holds({"or": [{"drift": True}, {"budget_over": False}]}, out) and holds(None, out)
    assert describe({"and": [{"answer": "kind", "in": ["auto", "property"]}, {"not": {"drift": True}}]}) == \
        '(kind in ["auto", "property"] and not drift)'


# --- the rest of the format

def test_spec_validation_names_the_place():
    p = Process("p")
    with pytest.raises(ProcessSpecError, match="needs question"):
        p.node("h", kind="human")
    with pytest.raises(ProcessSpecError, match="writes"):
        p.node("a", Fake("q"), writes={"x": "score"})
    with pytest.raises(ProcessSpecError, match="no node 'b'"):
        p.node("a", Fake("q")).edge("a", "b")
    with pytest.raises(ProcessSpecError, match="no end node"):
        p.spec()
    p.node("b", Fake("q"), end=True).node("c", Fake("q"), end=True)
    with pytest.raises(ProcessSpecError, match="start node is ambiguous"):
        p.spec()
    p.edge("a", "b")
    with pytest.raises(ProcessSpecError, match="start node is ambiguous"):
        p.spec()
    p._start = "a"
    with pytest.raises(ProcessSpecError, match="not reachable"):
        p.spec()
    with pytest.raises(ProcessSpecError, match="promise mode"):
        p.promise("best")
    with pytest.raises(ProcessSpecError, match="delta"):
        p.promise("end_to_end", risk=0.05)
    with pytest.raises(ProcessSpecError, match="family"):
        p.promise("union", risk=0.05, family="shared_rank")
    with pytest.raises(ProcessSpecError, match="major|reads 1"):
        Process.from_spec({"solvi_process": "2.0", "name": "x"})


def test_from_spec_checks_fingerprints_and_the_channel():
    p, s = claim()
    spec = p.spec()
    other = dict(s, fraud=Fake("fraud_risk", v="2"))
    with pytest.raises(ProcessSpecError, match="node 'fraud': the system given is not the recorded one"):
        Process.from_spec(spec, systems=other)
    q = Process.from_spec(spec, systems=other, strict=False)
    assert set(q.mismatched) == {"fraud"} and q.spec()["nodes"]["fraud"]["fingerprint"]["dispatcher"] == "disp-2"
    with pytest.raises(ProcessSpecError, match="channel given"):
        Process.from_spec(spec, systems=s, human=Channel("2"))
    with pytest.raises(ProcessSpecError, match="nodes the spec does not have"):
        Process.from_spec(spec, systems={"nobody": Fake("q")})


def test_render_mermaid_and_text():
    p, _ = claim()
    m = p.render("mermaid")
    assert m.startswith("flowchart TD")
    assert 'triage -->|kind in ["auto", "property"]| fraud'.replace('"', "'") in m
    assert "fraud -.->|drift; at most 2| triage" in m
    assert 'review{{"review<br/>human: fraud_risk"}}' in m and 'payout((("payout' in m
    t = render(p.spec(), "text")
    assert "fraud -> triage when drift (loop, at most 2)" in t and "promise union, risk 0.06" in t
    with pytest.raises(ValueError):
        p.render("dot")
