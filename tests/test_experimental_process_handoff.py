"""solvi.experimental.process and the stream certificate: the fixes of the 2026-10-30 external handoff (S270, S273,
S275, S279, S280, S281, S282) — a stream-certified node's promise read at its risk level (and the union promise in
force with one), the equal union split weighted by the visits a loop can make, a hand-over satisfying a
confidence_below edge, a node whose fitted head reads a fact it is not given named at build time, a promise on a
signal the answers cannot carry refused at install, a pending result naming its node and question, and the risk
detector's alarm on the promise's own event list.

Small synthetic processes and systems, no LLM, no torch; each test runs in seconds."""
import math
import warnings

import pytest

import solvi
from solvi import Answer, Catalog, Question, System

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import MemoryQueue, Process, holds
    from solvi.experimental.stream import RiskDetector, StreamPromise
    from solvi.experimental.stream.sampling import StreamDriver


def _build(tmp_path, name, examples, max_risk=0.05):
    """A solvi.build result on a small separable choice question, with its own store."""
    return solvi.build(Question(name, f"{name}?", Answer.choice(sorted({y for _, y in examples}))), examples,
                       catalog=Catalog(), max_risk=max_risk, novel=False,
                       storage=tmp_path / f"{name}.jsonl")


# --- S282: a stream certificate's level lives under "risk" (kind "joint")

def test_a_stream_certified_node_is_said_at_its_risk_level_and_bears_a_union_promise(tmp_path):
    cat = Catalog()

    @cat.rule("q")
    def q(x: int) -> str:
        return "a" if x % 2 else "b"

    s = System(cat, [Question("q", "?", Answer.choice(["a", "b"]))], storage=tmp_path / "n.jsonl")
    rec = StreamPromise(max_risk=0.05, delta=0.05).record()      # {"method": "stream-ltt", "kind": "joint",
    # ... "risk": 0.05} — the level is a risk (a share of the inputs), not an error among the answers given alone
    p = Process("pp")
    p.node("d7_a", s, reads=["x"], writes={"q": "answer"}, end=True, promise=rec)
    p.promise("per_node")
    g = p.guarantee()
    assert "d7_a (stream-ltt risk ≤ 0.05, δ 0.05)" in g["statement"] and "≤ None" not in g["statement"]
    p.promise("union", risk=0.10)                                # r_i = 0.10 ≥ the node's 0.05
    g = p.guarantee()
    assert g["in_force"] and g["problems"] == [] and g["nodes"]["d7_a"]["ok"]
    assert g["delta"] == pytest.approx(0.05)                     # counted with Bonferroni over its δ, like an LTT node
    assert any("Bonferroni" in w and "stream-ltt" in w for w in g["warnings"])
    p.promise("union", risk=0.04)                                # under its level: honestly out of force
    g = p.guarantee()
    assert not g["in_force"] and any("risk level 0.05 is over its share" in x for x in g["problems"])


# --- S273: the equal union split is weighted by the visits a loop can make

def test_an_equal_split_is_in_force_on_a_process_with_a_loop(tmp_path):
    ex = [({"x": i % 3}, "a" if i % 3 else "b") for i in range(200)]
    s = _build(tmp_path, "q", ex, max_risk=0.025)
    p = Process("loop", start="n1")
    p.node("n1", s, reads=["x"], writes={"q": "answer"})
    p.node("n2", s, reads=["x"], writes={"r": "answer"}, end=True)
    p.node("h", kind="human", question="q", reads=["x"], writes={"q": "answer"})
    p.edge("n1", "n2")
    p.edge("n2", "h", when={"confidence_below": 0.9}, loop=True, max_iter=1)
    p.edge("h", "n2")
    p.promise("union", risk=0.075)                               # equal, visit-weighted: r_i = 0.075 / (1 + 2) = 0.025
    g = p.guarantee()
    assert g["nodes"]["n2"]["visits"] == 2 and g["nodes"]["n1"]["visits"] == 1
    assert {n: v["r_i"] for n, v in g["nodes"].items()} == pytest.approx({"n1": 0.025, "n2": 0.025})
    assert g["bound"] == pytest.approx(0.075)                    # Σ r_i·m_i = risk by construction
    assert not any("m_i times" in x for x in g["problems"])      # the loop no longer puts the split over
    assert g["in_force"] and g["problems"] == []                 # each node keeps its 0.025 at its 0.025 share


# --- S275: a hand-over satisfies a confidence_below edge

def test_a_decision_sent_to_a_person_takes_a_confidence_below_edge(tmp_path):
    hand = {"facts": {}, "status": "human", "by": "human", "confidence": None}
    assert all(holds({"confidence_below": c}, hand) for c in (1e-9, 0.5, 0.9999))
    ok = {"facts": {}, "status": "ok", "by": "s1", "confidence": 0.7}
    assert holds({"confidence_below": 0.8}, ok) and not holds({"confidence_below": 0.5}, ok)
    s2 = {"facts": {}, "status": "ok", "by": "s2", "confidence": None}    # the slow path: no confidence to compare
    assert not holds({"confidence_below": 0.9999}, s2)

    ex = [({"x0": float(a), "x1": float(b)}, "a" if a + b > 1 else "b")
          for a, b in ((i / 99.0, (i % 7) / 9.0) for i in range(99))]
    s = _build(tmp_path, "hand", ex, max_risk=1e-9)              # nothing answers alone: the dispatcher hands over
    p = Process("edge", storage=tmp_path / "p.jsonl", human=MemoryQueue())
    p.node("n", s, reads=["x0", "x1"], writes={"q": "answer"})
    p.node("h", kind="human", question="q", end=True)
    p.edge("n", "h", when={"confidence_below": 0.9999})          # the unsure ones go to a person: it fires now
    r = p.run({"x0": 0.1, "x1": 0.2})
    assert r.status == "pending" and r.path[0]["status"] == "human"
    assert r.path[0]["edge"]["to"] == "h" and p.pending()[0]["node"] == "h"


# --- S270: a head reading a fact the node is not given, named at build time

def test_a_node_whose_head_reads_a_fact_it_is_not_given_warns(tmp_path):
    ex = [({"a": i, "b": (i * 7) % 3}, "x" if (i * 7) % 3 == 0 else "y") for i in range(300)]
    s = _build(tmp_path, "miss", ex, max_risk=0.1)
    p = Process("one", storage=tmp_path / "p.jsonl", human=MemoryQueue())
    with pytest.warns(UserWarning, match="node 'n': its fitted head reads \\['b'\\].*cannot compute: b"):
        p.node("n", s, reads=["a"], writes={"q": "answer"}, end=True)
    quiet = Process("quiet")                                     # a node that gives the head its facts stays quiet
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet.node("m", s, reads=["a", "b"], writes={"q": "answer"}, end=True)
    with pytest.warns(UserWarning, match="its fitted head reads"):
        Process.from_spec(p.spec(), systems={"n": s})             # a spec loaded with the system says it too


# --- S279: a promise on a signal the answers cannot carry is refused at install

def test_a_promise_gated_on_an_unreadable_signal_is_refused_at_install(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    ex = [({f"x{j}": float(v) for j, v in enumerate(x)}, f"o{int(np.argmax(W @ x))}")
          for W, x in [(rng.normal(size=(3, 4)), row) for row in rng.normal(size=(60, 4))]]
    s = _build(tmp_path, "act", ex)
    sp = StreamPromise(max_risk=0.05, delta=0.05, grid_from=50)
    was = s.system.guards["act"].promise                        # the build's own guarantee stays behind a refusal
    with pytest.raises(ValueError, match="the signal could not be read.*no act probability.*check signal="):
        s.system.guarantee("act", promise=sp, signal="act")      # no examples: the structural probe
    with pytest.raises(ValueError, match="the signal could not be read"):
        s.system.guarantee("act", promise=sp, signal="act", examples=ex[:1])   # an example: asked and read
    assert s.system.guards["act"].promise is was                 # nothing half-installed stays behind
    s.system.guarantee("act", promise=sp)                        # the confidence signal reads fine and installs
    assert s.system.guards["act"].promise is sp


# --- S280: a pending result names the node that holds the question

def test_a_pending_result_names_the_node_and_the_question(tmp_path):
    cat = Catalog()

    @cat.rule("q")
    def q(x: int) -> str:
        return "a"

    @cat.rule("v")
    def v(zz: str) -> str:
        return "ok"

    s = System(cat, [Question("q", "?", Answer.choice(["a", "b"]))], storage=tmp_path / "n1.jsonl")
    sv = System(cat, [Question("v", "?", Answer.choice(["ok"]))], storage=tmp_path / "n2.jsonl")
    p = Process("who", storage=tmp_path / "proc.jsonl", human=MemoryQueue())
    p.node("l0", s, reads=["x"], writes={"q": "answer"})
    p.node("lighthouse", kind="human", question="q", end=True)
    p.node("abstainer", sv, reads=["q", "zz"], writes={"v": "answer"}, question="v")
    p.edge("l0", "lighthouse", when={"answer": "zz", "eq": "a"})    # zz = a: to the person; else the abstainer
    p.edge("l0", "abstainer")
    r = p.run({"x": 1})                                             # no zz: the abstainer abstains and hands over
    assert r.status == "pending" and (r.node, r.question) == ("abstainer", "v")
    assert {k: p.pending()[0][k] for k in ("node", "question")} == {"node": r.node, "question": r.question}
    still = p.resume(r.run)                                         # no answer yet: the fields stay
    assert still.status == "pending" and (still.node, still.question) == ("abstainer", "v")
    r2 = p.run({"x": 2, "zz": "a"})                                 # a human node asked directly
    assert r2.status == "pending" and (r2.node, r2.question) == ("lighthouse", "q")


# --- S281: the risk detector's alarm is on the promise's own event list

def test_an_alarm_is_on_the_promise_events_with_its_statistic():
    sp = StreamPromise(max_risk=0.05, delta=0.05, grid_from=30)
    det = RiskDetector(alpha=0.05, arl=100)
    drv = StreamDriver(sp, det, on_drift="recertify", probation=10)
    i = 0
    while math.isinf(sp.threshold):
        sp.observe(0.9)
        sp.learn(0.9, "a", "a", sampling_policy="complete")                                  # clean labels until a threshold certifies
        i += 1
        assert i < 5000
    for _ in range(20):                                          # then wrong labelled samples: the detector fires
        thr = None if math.isinf(sp.threshold) else sp.threshold
        _, events, _ = drv.after(score=0.9, would="a", alone=True, label="b", label_source="human",
                                 sampled=True, free=True, rho=1.0, in_force=thr)
        if any(e.get("event") == "alarm" for e in events):
            break
    names = [e["event"] for e in sp.events]
    alarm = next(e for e in sp.events if e["event"] == "alarm")
    assert "alarm" in names and names.index("alarm") < names.index("recertify")   # next to the restart, in order
    assert alarm["R"] >= 100 and alarm["detector_n"] >= 1 and "Shiryaev" in alarm["why"]
    assert det.alarms == 1 and det.state()["n"] == 0             # the detector itself reset: the event is the record
