"""solvi.experimental.process, stage 3: the path promise — union (split "equal", {node: r_i} or a callable, checked
against each node's own promise and against the loops), per_node ("the end error is not bounded"), and the watch on
the mix of paths (docs/experimental.md#process), and end_to_end (stage 4): calibration on end-labelled runs over a
threshold family, the gate at run time and its replay step.

Admission (PROCESS_DESIGN §F, S3) is the E1 stand at K=10 (benchmarks/process_e1.py); these tests check the
statements, the checks behind "in force" and the watch."""
import random
import warnings

import numpy as np
import pytest

from test_experimental_process_human import review_process
from test_experimental_process_run import _claims, _glass, _kind, _pay, _risk, claim_process, claim_systems, clock

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.process import (MemoryQueue, PathWatch, PerNodeFixed, Process, ProcessRunError,
                                            ProcessSpecError, SharedRank, ThresholdFamily, _visits)


@pytest.fixture(scope="module")
def ps(tmp_path_factory):
    return claim_systems(tmp_path_factory.mktemp("promise"), "p", clock())      # four nodes, each at max_risk 0.05


def test_union_equal_split_in_force_when_every_node_keeps_its_share(ps):
    p = claim_process(ps).promise(mode="union", risk=0.2)                       # r_i = 0.05 = each node's risk
    g = p.guarantee()
    assert g["in_force"] and g["problems"] == [] and g["bound"] == pytest.approx(0.2)
    assert {n: v["r_i"] for n, v in g["nodes"].items()} == pytest.approx({n: 0.05 for n in ps})
    assert all(v["promise"]["method"] == "crc" and v["promise"]["risk"] == 0.05 for v in g["nodes"].values())
    assert "P(a run ends automatically and its end answer is wrong) ≤ 0.2" in g["statement"]
    assert "not over the path" in g["statement"] and any(a.startswith("A2") for a in g["assumptions"])
    r = p.run({"item": "car", "amount": 300.0})
    assert r.status == "decided" and r.automatic is True and r.risk == 0.2
    assert "; in force" in p.explain(r) and "(automatically)" in p.explain(r)


def test_union_is_not_in_force_when_a_node_is_calibrated_over_its_share(ps):
    p = claim_process(ps).promise(mode="union", risk=0.06)                      # r_i = 0.015 < 0.05
    g = p.guarantee()
    assert not g["in_force"] and len(g["problems"]) == 4
    assert "calibrated at risk 0.05, over its share r_i = 0.015: rebuild it with max_risk=0.015" in g["problems"][0]
    r = p.run({"item": "car", "amount": 300.0})
    assert r.automatic is True and r.risk is None
    assert "NOT in force" in p.explain(r)


def test_a_given_or_computed_split_is_checked_against_the_risk_and_the_nodes(ps):
    ok = {"triage": 0.05, "fraud": 0.05, "payout": 0.05, "glass": 0.05}
    assert claim_process(ps).promise(mode="union", risk=0.2, split=ok).guarantee()["in_force"]
    with pytest.raises(ProcessSpecError, match="over the process risk"):
        claim_process(ps).promise(mode="union", risk=0.15, split=ok).spec()
    g = claim_process(ps).promise(mode="union", risk=0.2, split={"triage": 0.1, "fraud": 0.05, "payout": 0.05}).guarantee()
    assert not g["in_force"] and g["nodes"]["glass"]["r_i"] == 0.0           # a node left out has no share
    assert any("glass: calibrated at risk 0.05, over its share r_i = 0" in x for x in g["problems"])

    def by_cost(spec):                       # a payout's error costs more: a smaller share — here all fit at 0.05
        return {n: (0.05 if n in ("payout", "glass") else 0.06) for n in spec["nodes"]}
    g = claim_process(ps).promise(mode="union", risk=0.22, split=by_cost).guarantee()
    assert g["in_force"] and g["bound"] == pytest.approx(0.22)
    assert claim_process(ps).promise(mode="union", risk=0.22, split=by_cost).spec()["promise"]["split"]["kind"] == \
        "callable"


def test_a_loop_counts_a_node_once_per_run_it_can_make(ps):
    nodes = {n: {"kind": "decision"} for n in "abcd"}
    edges = [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}, {"from": "c", "to": "d"},
             {"from": "c", "to": "b", "loop": True, "max_iter": 2}, {"from": "d", "to": "a", "loop": True, "max_iter": 1}]
    assert _visits(nodes, edges) == {"a": 2, "b": 4, "c": 4, "d": 4}          # an upper bound (d runs twice at most)
    p = claim_process(ps)
    p.edge("fraud", "triage", when={"drift": True}, loop=True, max_iter=2)
    g = p.promise(mode="union", risk=0.2).guarantee()                         # equal: 0.05 each, but triage, fraud ×3
    assert g["nodes"]["triage"]["visits"] == 3 and g["nodes"]["glass"]["visits"] == 1
    assert not g["in_force"] and g["bound"] == pytest.approx(0.05 * (3 + 3 + 1 + 1))
    assert any("m_i times" in x and "triage ×3" in x for x in g["problems"])
    roomy = {"triage": 0.05, "fraud": 0.05, "payout": 0.05, "glass": 0.05}
    assert claim_process(ps).edge("fraud", "triage", when={"drift": True}, loop=True, max_iter=2).promise(
        mode="union", risk=0.4, split=roomy).guarantee()["in_force"]


def test_per_node_says_the_end_error_is_not_bounded(ps):
    p = claim_process(ps).promise(mode="per_node")
    g = p.guarantee()
    assert g["in_force"] and g["bound"] is None and "The end error is not bounded" in g["statement"]
    assert "triage (risk ≤ 0.05)" in g["statement"]
    r = p.run({"item": "car", "amount": 300.0})
    assert r.risk is None and "end error is not bounded" in p.explain(r)


def _end_label(c):
    k = _kind(c["item"])
    return _glass(c["amount"]) if k == "glass" else _pay(c["amount"], k, _risk(c["amount"], k))


def test_end_to_end_is_not_in_force_until_calibrated(ps):
    p = claim_process(ps).promise(mode="end_to_end", risk=0.05, delta=0.1)
    g = p.guarantee()
    assert not g["in_force"] and "not calibrated" in g["problems"][0] and g["bound"] is None
    r = p.run({"item": "car", "amount": 300.0})
    assert r.status == "decided" and r.risk is None and "promise" not in r.path[0]   # no threshold, no gate


def test_end_to_end_calibrated_on_right_labels_lets_runs_through_and_replays(ps, tmp_path):
    ex = [(c, _end_label(c)) for c in _claims(300, 7)]
    p = claim_process(ps, storage=str(tmp_path / "proc.jsonl")).promise(mode="end_to_end", risk=0.05, delta=0.1,
                                                                         examples=ex)
    h0 = p.spec()["hash"]
    th = p.calibrate()
    assert th["procedure"] == "bonferroni" and th["grid"] == 64 and th["n"] == 300 and th["wrong"] == 0
    assert th["automatic"] == 300 and th["t"] is not None and p.spec()["hash"] != h0
    pr = p.spec()["promise"]
    assert pr["calibrated_on"]["n"] == 300 and pr["calibrated_on"]["family"].startswith("solvi.shared_rank")
    g = p.guarantee()
    assert g["in_force"] and g["bound"] == 0.05 and g["delta"] == 0.1 and "with probability ≥ 0.9" in g["statement"]
    assert any(x.startswith("A3") for x in g["assumptions"])
    r = p.run({"item": "car", "amount": 300.0})
    assert r.status == "decided" and r.automatic and r.risk == 0.05
    assert all(st["promise"] == {"signal": st_conf, "automatic": True}
               for st, st_conf in zip(r.path, [r.nodes[st["node"]].confidence for st in r.path]))
    rep = p.replay(r)
    assert rep["ok"] and all("promise" in s["checked"] for s in rep["steps"])
    q = Process.from_spec(p.spec(), systems=ps, storage=str(tmp_path / "proc.jsonl"))
    assert q.guarantee()["in_force"] and q.replay(r.run)["ok"]           # the artifact round-trips with the spec


def test_end_to_end_with_wrong_labels_certifies_nothing_and_hands_every_node_over(ps, tmp_path):
    ex = [(c, "pay" if _end_label(c) == "refuse" else "refuse") for c in _claims(300, 8)]   # every label wrong
    p = claim_process(ps, storage=str(tmp_path / "proc.jsonl")).promise(mode="end_to_end", risk=0.05, delta=0.1)
    th = p.calibrate(ex)
    assert th["t"] is None and th["certified"] == 0 and th["automatic"] == 0
    g = p.guarantee()
    assert g["in_force"] and "no threshold was certified" in g["statement"]
    r = p.run({"item": "car", "amount": 300.0})                    # on_human="abstain": the first node stops it
    assert r.status == "abstain" and r.path[0]["by"] == "human" and r.path[0]["promise"]["held"] is True
    assert "below the process threshold" in p.explain(r)
    assert p.replay(r)["ok"]


def test_end_to_end_promise_step_names_a_forged_decision_of_the_threshold(ps, tmp_path):
    import json
    ex = [(c, _end_label(c)) for c in _claims(200, 9)]
    path = tmp_path / "proc.jsonl"
    p = claim_process(ps, storage=str(path)).promise(mode="end_to_end", risk=0.05, delta=0.1)
    p.calibrate(ex)
    r = p.run({"item": "car", "amount": 300.0})
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    for d in recs:
        if d.get("id") == r.stored_id:
            d["path"][1]["promise"]["signal"] = 0.5
    path.write_text("".join(json.dumps(d) + "\n" for d in recs))
    q = Process.from_spec(p.spec(), systems=ps, storage=str(path))
    rep = q.replay(r.run)
    assert not rep["ok"] and rep["at"]["node"] == "fraud" and rep["at"]["step"] == "promise"
    assert rep["kind"] == "integrity"


def test_end_to_end_is_out_of_force_when_the_process_changes(ps):
    ex = [(c, _end_label(c)) for c in _claims(200, 10)]
    p = claim_process(ps).promise(mode="end_to_end", risk=0.05, delta=0.1)
    p.calibrate(ex)
    p.edge("glass", "payout", when={"status": "abstain"})
    g = p.guarantee()
    assert not g["in_force"] and any("changed after its calibration" in x for x in g["problems"])


def test_built_in_families_grids_are_monotone_and_fixed_sequence_opens_end_first():
    rng = np.random.default_rng(0)
    sig = {n: list(rng.random(50)) for n in ("a", "b", "c")}
    grid = SharedRank().grid({}, sig)
    assert len(grid) == 64
    for n in sig:
        thr = [t[n] for t in grid]
        assert thr == sorted(thr) and thr[0] == min(sig[n])
    counts = [sum(SharedRank().automatic(n, s, t) for n in sig for s in sig[n]) for t in grid]
    assert counts == sorted(counts, reverse=True) and counts[0] == 150
    spec = {"promise": {"family": {"kind": "per_node_fixed", "order": ["c", "b", "a"]}}}
    chain = PerNodeFixed().grid(spec, sig)
    assert len(chain) == 3 * 64 and chain[0]["a"] is None and chain[0]["b"] is None and chain[0]["c"] is not None
    assert chain[63]["c"] == min(sig["c"]) and chain[64]["b"] is not None and chain[64]["a"] is None
    assert all(sum(x[k] != y[k] for k in sig) <= 1 for x, y in zip(chain, chain[1:]))   # one node per step


def test_per_node_fixed_defaults_to_the_end_first_and_calibrates(ps):
    ex = [(c, _end_label(c)) for c in _claims(200, 11)]
    p = claim_process(ps).promise(mode="end_to_end", risk=0.05, delta=0.1, family="per_node_fixed")
    order = p.spec()["promise"]["family"]["order"]
    assert order.index("payout") < order.index("fraud") < order.index("triage")
    th = p.calibrate(ex)
    assert th["procedure"] == "fixed" and th["grid"] == 4 * 64 and th["t"] is not None and p.guarantee()["in_force"]


def test_a_custom_family_is_used_and_refused_when_another(ps):
    class Open(ThresholdFamily):
        def grid(self, spec, signals):
            return [0.5, 0.9]

        def automatic(self, node, signal, t):
            return signal is not None and signal >= t

        def fingerprint(self):
            return "open-1"

    class Other(Open):
        def fingerprint(self):
            return "other-1"
    ex = [(c, _end_label(c)) for c in _claims(200, 12)]
    p = claim_process(ps).promise(mode="end_to_end", risk=0.05, delta=0.1, family=Open())
    th = p.calibrate(ex)
    assert th["t"] == 0.5 and p.guarantee()["in_force"]
    with pytest.raises(ProcessSpecError, match="threshold family"):
        Process.from_spec(p.spec(), systems=ps, family=Other())
    q = Process.from_spec(p.spec(), systems=ps)
    with pytest.raises(ProcessRunError, match="needs its object"):
        q.run({"item": "car", "amount": 300.0})
    assert Process.from_spec(p.spec(), systems=ps, family=Open()).run({"item": "car", "amount": 300.0}).risk == 0.05


def test_a_node_without_a_promise_an_ltt_node_and_a_mismatch(ps):
    p = claim_process(ps)
    p.nodes["glass"].pop("promise", None)
    p.promise(mode="union", risk=0.2)

    class Bare:                                                                 # keeps no promise
        guards = {}
    p.systems["glass"] = Bare()
    g = p.guarantee()
    assert not g["in_force"] and any("glass: keeps no promise" in x for x in g["problems"])
    p.systems["glass"] = ps["glass"]
    p.nodes["glass"]["promise"] = {"method": "ltt", "error": 0.04, "delta": 0.1, "n": 400}
    g = p.guarantee()
    assert g["in_force"] and g["delta"] == pytest.approx(0.1) and "with probability ≥ 0.9" in g["statement"]
    assert any("Bonferroni" in w for w in g["warnings"])
    p.mismatched["fraud"] = {"recorded": "x", "actual": "y"}
    assert not p.guarantee()["in_force"]


def test_a_run_with_a_person_on_its_path_is_not_automatic(tmp_path, ps):
    queue = MemoryQueue()
    p = review_process(ps, tmp_path / "process.jsonl", queue).promise(mode="union", risk=0.1)
    r = p.run({"item": "car", "amount": 300.0})
    assert r.status == "pending" and r.automatic is None
    queue.answer(r.ticket, "low", reviewer="anna")
    r2 = p.resume(r.run)
    assert r2.status == "decided" and r2.automatic is False
    assert "(with a person on the path)" in p.explain(r2)


def _routed(ps, seed, share_glass, n):
    rng = random.Random(seed)
    items = ["car", "bumper", "roof", "door"]
    return [{"item": "windscreen" if rng.random() < share_glass else rng.choice(items),
             "amount": float(rng.randint(10, 400))} for _ in range(n)]


def test_the_path_watch_stays_quiet_on_the_same_mix_and_flags_a_new_one(ps):
    p = claim_process(ps).promise(mode="union", risk=0.2)
    ref = [p.run(c) for c in _routed(ps, 1, 0.2, 300)]
    w = PathWatch(alpha=0.01, horizon=500, sims=2000, seed=3)
    p.monitor(w, reference=ref)
    assert w.n_ref == 300 and w.p0[1] == pytest.approx((sum("glass" in [s["node"] for s in r.path] for r in ref) + 0.5)
                                                      / 301)
    same = [p.run(c) for c in _routed(ps, 2, 0.2, 400)]
    assert not w.flagged and not same[-1].path_drift["drift"]
    assert p.guarantee()["in_force"]
    shifted = [p.run(c) for c in _routed(ps, 3, 0.6, 400)]
    assert w.flagged and shifted[-1].path_drift["drift"] and "triage -> glass" in w.why[0]
    assert "more runs take triage -> glass" in w.why[0]
    g = p.guarantee()
    assert not g["in_force"] and any("mix of paths drifted" in x for x in g["problems"])
    assert "paths: the mix drifted" in p.explain(shifted[-1])
    w.reset()
    assert p.guarantee()["in_force"]


def test_the_path_watch_false_flag_rate_is_near_alpha():
    rng = np.random.default_rng(0)

    def run():
        return [0, 2] if rng.random() < 0.7 else [1]
    flags = 0
    for k in range(60):                                   # a fresh reference each time: alpha is over both
        w = PathWatch(alpha=0.05, horizon=200, sims=1000, seed=k).set_reference([run() for _ in range(300)])
        for _ in range(200):
            w.observe(run())
        flags += w.flagged
    assert flags <= 8                                     # alpha 5 % of 60 = 3 (+ the binomial spread)
    with pytest.raises(ValueError, match="at least 20"):
        PathWatch().set_reference([[0]] * 5)
    with pytest.raises(TypeError):
        Process("x").monitor(object())


def test_a_build_node_that_sends_its_decision_to_a_person_is_handed_over(tmp_path):
    """A solvi.build node whose dispatcher sends the decision to a person (by "human", status "human") is handed over to
    the channel when no edge holds for it — before stage 3 the run stopped "abstain" there (found by the E1 harness)."""
    import solvi
    from test_experimental_process_run import KIND, _catalog, _claims, _kind
    ex = [({"item": c["item"]}, _kind(c["item"])) for c in _claims(40, 2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shy = solvi.build(KIND, ex, catalog=_catalog("kind", _kind), max_risk=0.001,     # too few examples: a person
                          storage=str(tmp_path / "shy.jsonl"))
    assert shy.ask({"item": "car"}).by == "human"
    queue = MemoryQueue()
    p = Process("shy", storage=tmp_path / "p.jsonl", human=queue)
    p.node("triage", shy, reads=["item"], writes={"kind": "answer"}, end=True)
    r = p.run({"item": "car"})
    assert r.status == "pending" and r.path[0]["by"] == "human"
    queue.answer(r.ticket, "auto", reviewer="anna")
    r2 = p.resume(r.run)
    assert r2.status == "decided" and r2.answer == "auto" and r2.automatic is False
    rep = p.replay(r2.run)
    assert rep["ok"], (rep["at"], rep["kind"], rep["reason"])
