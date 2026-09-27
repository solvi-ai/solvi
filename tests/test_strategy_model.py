"""The model strategist (solvi.strategy) and name matching (solvi.aliases): code plans, dead ends, mandatory checks, verified
model proposals, the plan record in the trace, alias acceptance — with stub models (no torch needed)."""
import time
from typing import Literal

import numpy as np
import pytest

from solvi import Catalog, Question, System
from solvi.strategist import plan as det_plan
from solvi.strategy import ModelStrategist, check_segment, search, segments, mandatory_checks, validate


def chain():
    """a2 has a dead-end producer (reads a feed nobody gives) and a costly shortcut; a3 a cheap and a costly producer."""
    cat = Catalog()

    @cat.fn(cost=1)
    def a1(g1: float) -> float:
        return g1 + 1

    @cat.fn(provides="a2", cost=1)
    def a2_feed(partner_feed: float) -> float:
        """Reads the partner feed. Fast."""
        return partner_feed

    @cat.fn(provides="a2", cost=3)
    def a2_std(a1: float) -> float:
        """Standard step."""
        return a1 * 2

    @cat.fn(provides="a3", cost=20)
    def a3_slow(a1: float, g2: float) -> float:
        """Slow: rebuilds it from scratch."""
        return a1 * 2 + g2

    @cat.fn(provides="a3")
    def a3_std(a2: float, g2: float) -> float:
        """Standard step."""
        return a2 + g2

    @cat.check(hard=True, then={"ok": "rejected"})
    def policy(a2: float) -> bool:
        return a2 < 100

    @cat.check
    def audit(a3: float) -> bool:
        return a3 > 0

    @cat.rule("ok")
    def ok(a3: float) -> Literal["approved", "review", "rejected"]:
        return "approved" if a3 > 5 else "review"
    return cat, [Question("ok", "approve?", None)]


ST = {"g1": 3.0, "g2": 1.0}


def test_dead_end_breaks_the_deterministic_strategist_not_this_one():
    cat, qs = chain()
    assert System(cat, qs).ask(ST)["ok"].status == "abstain"          # union of inputs: partner_feed is never given
    s = System(cat, qs, strategist=ModelStrategist())
    r = s.ask(ST)
    assert r["ok"].answer == "approved"
    assert r.trace.replay(s, r.flow)["ok"]
    assert r.trace.records[-1].kind == "plan" and r.trace.records[-1].provenance == "computed"


def test_declared_mode_equals_deterministic_without_dead_ends():
    cat = Catalog()

    @cat.fn
    def x(a: int) -> int:
        return a + 1

    @cat.fn(provides="y")
    def y1(x: int) -> int:
        return x * 2

    @cat.fn(provides="y")
    def y2(a: int) -> int:
        return a * 3

    @cat.rule("q")
    def q(y: int) -> bool:
        return y > 4
    qs = [Question("q", "?", None)]
    for a in (1, 2, 5):
        d = System(cat, qs).ask({"a": a})["q"]
        m = System(cat, qs, strategist=ModelStrategist()).ask({"a": a})["q"]
        assert (d.answer, d.status) == (m.answer, m.status)


def test_equivalent_mode_is_cheapest_and_keeps_mandatory_checks():
    cat, qs = chain()
    ms = ModelStrategist(producers="equivalent", costs={"a3_std": 1.0})
    flow = ms.plan(cat, qs, set(ST))
    assert ms.last["choice"]["a3"] == "a3_std" and ms.last["choice"]["a2"] == "a2_std"
    assert ms.last["mandatory"] == {"ok": ["policy"]}
    assert "policy" in flow.per_question["ok"]
    assert validate(cat, flow, qs, set(ST), ms.last["mandatory"]) == []
    # a flow without the mandatory check fails validation
    bad = det_plan(cat, [Question("ok", "approve?", None)], set(ST))
    assert validate(cat, bad, qs, set(ST), {"ok": ["policy"]})


def test_search_milp_and_branch_and_bound_agree():
    cat, qs = chain()
    gov = mandatory_checks(cat, qs, set(ST))
    must = [c for cs in gov.values() for c in cs]
    for costs in (None, {"a3_std": 1.0}, {"a3_std": 50.0}):
        a = search(cat, qs, set(ST), costs, extra=must, method="milp")
        b = search(cat, qs, set(ST), costs, extra=must, method="bnb")
        assert a.cost == pytest.approx(b.cost) and a.choice == b.choice


class Stub:
    """A segment model that proposes given node lists per fact."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.fingerprint = "stubfp0123456789"

    def info(self):
        return {"type": "ModelStrategist", "id": "stub", "fp": self.fingerprint}

    def propose(self, segs):
        return [{"nodes": list(self.nodes.get(s["fact"], [])), "score": -0.1, "second": None} for s in segs]


def test_model_proposal_verified_recorded_and_replayed():
    cat, qs = chain()
    ms = ModelStrategist(Stub({"a3": ["a3_std"]}), producers="equivalent")
    s = System(cat, qs, strategist=ms)
    r = s.ask(ST)
    seg = [x for x in ms.last["segments"] if x["fact"] == "a3"][0]
    assert seg["by"] == "model" and "accepted" in seg
    rec = r.trace.records[-1]
    assert rec.kind == "plan" and rec.provenance == "proposed" and rec.model["fp"] == "stubfp0123456789"
    assert r.trace.replay(s, r.flow)["ok"]
    assert r["ok"].answer == "approved"
    rec.value["choice"]["a3"] = "a3_nope"                 # tampering: hash broken and the plan no longer verifies
    rep = r.trace.replay(s, r.flow)
    assert not rep["ok"]


def test_invalid_proposal_falls_back_to_code():
    cat, qs = chain()
    for nodes in (["a2_feed"], ["nonexistent"], ["a2_std", "a3_std"], []):
        ms = ModelStrategist(Stub({"a3": nodes}), producers="equivalent")
        r = System(cat, qs, strategist=ms).ask(ST)
        seg = [x for x in ms.last["segments"] if x["fact"] == "a3"][0]
        assert seg["by"] == "code" and seg["rejected"]
        assert r["ok"].answer == "approved"                  # a model error costs cost / coverage, never the answer


def test_check_segment_rules():
    cat, qs = chain()
    init = set(ST)
    gov = mandatory_checks(cat, qs, init)
    code = search(cat, qs, init, extra=[c for cs in gov.values() for c in cs])
    seg = [s for s in segments(cat, qs, init, code, all_facts=True) if s["fact"] == "a3"][0]
    assert check_segment(cat, seg, ["a3_std"]) is None or "a2" not in {x for x, _, _ in seg["available"]}
    assert "not a candidate" in check_segment(cat, seg, ["policy"])
    assert "at most" in check_segment(cat, seg, ["a1"] * 5)
    assert check_segment(cat, seg, []) == "empty segment"


def test_deterministic_strategist_is_linear_on_diamonds():
    cat = Catalog()
    prev, prev2 = "g", "g"
    for i in range(60):                                    # every fact reachable by two routes: exponential without memo
        name = f"m{i}"
        cat.fn(_two(name, prev, prev2 if prev2 != prev else "h"))
        prev2, prev = prev, name

    @cat.rule("q")
    def q(m59) -> bool:
        return m59 > 0
    t0 = time.perf_counter()
    det_plan(cat, [Question("q", "?", None)], {"g", "h"})
    assert time.perf_counter() - t0 < 2.0


def _two(name, a, b):
    import inspect

    def f(**kw):
        return kw[a] + kw[b]
    f.__name__ = name
    f.__signature__ = inspect.Signature([inspect.Parameter(a, inspect.Parameter.POSITIONAL_OR_KEYWORD),
                                         inspect.Parameter(b, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    return f


# ---------------------------------------------------------------------------------------------------------------- aliases
from solvi.aliases import Proposal, accept, apply, propose, unresolved  # noqa: E402


def team_catalog():
    """Two teams: the rule's team calls the facts `INV_TOTAL` and `TAX_AMT`."""
    cat = Catalog()

    @cat.fn
    def invoice_total(net: float, tax: float) -> float:
        """Gross invoice amount."""
        return net + tax

    @cat.fn
    def tax_amount(tax: float) -> float:
        """Tax on the invoice."""
        return tax

    @cat.rule("big")
    def big(INV_TOTAL: float, TAX_AMT: float) -> bool:
        return INV_TOTAL > 100 and TAX_AMT < 30
    return cat, [Question("big", "large invoice?", None)]


class StubMatcher:
    """Embeds a name as a bag of letters (enough for SCREAMING vs snake names in a test)."""
    T = 0.05

    def embed(self, texts, names):
        out = []
        for t, n in zip(texts, names):
            v = np.zeros(27)
            for ch in (n or t.split("|")[0]).lower():
                if ch.isalpha():
                    v[ord(ch) - 97] += 1
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(out)


def test_unresolved_and_apply():
    cat, qs = team_catalog()
    assert set(unresolved(cat, {"net", "tax"})) == {"INV_TOTAL", "TAX_AMT"}
    cat2 = apply(cat, {"INV_TOTAL": "invoice_total", "TAX_AMT": "tax_amount"})
    assert cat2.aliases == {"INV_TOTAL": "invoice_total", "TAX_AMT": "tax_amount"}
    assert not unresolved(cat2, {"net", "tax"})
    s = System(cat2, qs)
    r = s.ask({"net": 100.0, "tax": 20.0})
    assert r["big"].answer == "yes" and r.trace.replay(s, r.flow)["ok"]


def test_accept_by_examples_and_active_mode():
    cat, qs = team_catalog()
    init = {"net", "tax"}
    props = propose(cat, qs, init, StubMatcher())
    assert props
    states = [{"net": n, "tax": t} for n, t in [(100, 20), (50, 10), (90, 40), (120, 5), (10, 1), (95, 29), (200, 35)]]

    def truth(s):
        return {"big": "yes" if s["net"] + s["tax"] > 100 and s["tax"] < 30 else "no"}
    ex = [(s, truth(s)) for s in states]
    ok = ({"INV_TOTAL": "invoice_total", "TAX_AMT": "tax_amount"}, {"INV_TOTAL": "invoice_total", "TAX_AMT": "tax"})
    got = accept(cat, qs, props, ex, probes=states, k=5)     # tax_amount is the given tax: both wirings are the same fact
    assert got.aliases in ok
    act = accept(cat, qs, props, ex, probes=states, oracle=truth, active=(2, 4))
    assert act.aliases in ok + (None,) and act.labels <= 6
    # a wrong-only proposal list is never accepted
    wrong = [Proposal({"INV_TOTAL": "tax_amount", "TAX_AMT": "invoice_total"}, -1.0, {})]
    assert accept(cat, qs, wrong, ex, probes=states, k=5).aliases is None


def test_example_17_runs(capsys, monkeypatch):
    import runpy
    from pathlib import Path
    monkeypatch.delenv("SOLVI_STRATEGIST", raising=False)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "17_model_strategist.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "abstain" in out and "approve = 'pay'" in out and "plan record provenance proposed" in out
    assert "accepted: {'INV_TOTAL': 'invoice_amount', 'FX': 'fx_rate'}" in out


def test_segment_model_save_load_propose(tmp_path):
    """A tiny random segment network: save → load (torch) → propose; ONNX parity when onnxruntime is there."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tok_src = pytest.importorskip("huggingface_hub").snapshot_download
    import json
    from solvi import strategy_model as SM
    try:
        base = tok_src("answerdotai/ModernBERT-base", allow_patterns=["*.json"])
    except Exception:  # noqa: BLE001
        pytest.skip("ModernBERT tokenizer not available offline")
    cfg = json.load(open(f"{base}/config.json"))
    cfg.update(num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=2, vocab_size=8193,
               global_rope_theta=160000.0)
    net = SM.build_net(cfg, d=64, heads=2, layers=1, passes=2)
    SM.save(net, str(tmp_path), net.enc.config, base, {"name": "tiny", "n_nodes": 8, "cell_tok": 48,
                                                      "arch": {"d": 64, "heads": 2, "layers": 1, "passes": 2}})
    m = SM.SegmentModel.load(str(tmp_path), backend="torch")
    cat, qs = chain()
    ms = ModelStrategist(m, producers="equivalent")
    r = System(cat, qs, strategist=ms).ask(ST)
    assert r["ok"].answer == "approved" and len(m.fingerprint) == 16
    if __import__("importlib").util.find_spec("onnxruntime") and __import__("importlib").util.find_spec("onnx"):
        SM.export_onnx(str(tmp_path))
        mo = SM.SegmentModel.load(str(tmp_path), backend="onnx")
        segs = segments(cat, qs, set(ST), search(cat, qs, set(ST)), all_facts=True)
        assert [p["nodes"] for p in mo.propose(segs)] == [p["nodes"] for p in m.propose(segs)]
