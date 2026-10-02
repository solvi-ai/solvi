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
    assert "a2" in {x for x, _, _ in seg["available"]} and check_segment(cat, seg, ["a3_std"]) is None
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


def test_link_table_ranks_the_sources_of_each_unresolved_name_and_match_names_wires_them():
    from solvi.aliases import link_table, match_names
    cat, qs = team_catalog()
    table = link_table(cat, qs, {"net", "tax"}, StubMatcher(), k=3)
    assert set(table) == {"INV_TOTAL", "TAX_AMT"} and table["INV_TOTAL"][0][0] == "invoice_total"
    for links in table.values():
        lp = [x for _, x in links]
        assert len(links) == 3 and lp == sorted(lp, reverse=True) and all(x <= 0 for x in lp)
    states = [{"net": n, "tax": t} for n, t in [(100, 20), (50, 10), (90, 40), (120, 5), (10, 1), (95, 29), (200, 35)]]
    ex = [(s, {"big": "yes" if s["net"] + s["tax"] > 100 and s["tax"] < 30 else "no"}) for s in states]
    wired, got = match_names(cat, qs, {"net", "tax"}, StubMatcher(), ex, probes=states)
    assert got.aliases["INV_TOTAL"] == "invoice_total" and wired is not cat and not unresolved(wired, {"net", "tax"})
    assert all(System(wired, qs).ask(st)["big"].answer == want["big"] for st, want in ex)
    same, none = match_names(wired, qs, {"net", "tax"}, StubMatcher(), [])
    assert same is wired and none.why == "every name resolves" and none.aliases == {}


def test_model_strategist_record_false_keeps_the_plan_out_of_the_trace_and_fallbacks_false_keeps_one_producer():
    cat, qs = chain()
    rec = System(cat, qs, strategist=ModelStrategist(producers="equivalent")).ask(ST)
    assert rec.trace.records[-1].kind == "plan"
    off = System(cat, qs, strategist=ModelStrategist(producers="equivalent", record=False)).ask(ST)
    assert all(r.kind != "plan" for r in off.trace.records) and off["ok"].answer == rec["ok"].answer
    one = System(cat, qs, strategist=ModelStrategist(producers="equivalent", fallbacks=False)).ask(ST)
    assert all(len(st.part.alternatives or [st.part]) == 1 for st in one.flow.steps) and one["ok"].answer == "approved"


def test_plan_raises_plan_error_for_a_checkpoint_that_is_not_in_the_catalog():
    from solvi.strategist import PlanError
    cat, _ = chain()
    with pytest.raises(PlanError, match="requires .nope., which is not in the catalog"):
        det_plan(cat, [Question("ok", "", None, requires=["nope"])], set(ST))


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
    assert act.aliases in ok and act.labels <= 6                     # something was accepted, with few labels asked
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


@pytest.mark.model
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


# ---------------------------------------------------------------- facts derivable from each other
def _net_gross():
    cat = Catalog()

    @cat.fn(provides="net", cost=1)
    def net_given(net_in: float) -> float:
        return net_in

    @cat.fn(provides="net", cost=2)
    def net_from_gross(gross: float) -> float:
        return gross / 1.25

    @cat.fn(provides="gross", cost=1)
    def gross_given(gross_in: float) -> float:
        return gross_in

    @cat.fn(provides="gross", cost=2)
    def gross_from_net(net: float) -> float:
        return net * 1.25

    @cat.rule("big")
    def big(gross: float, net: float) -> bool:
        return gross > 100 and net > 0
    return cat, [Question("big", "", None)]


@pytest.mark.parametrize("strategist", [
    lambda: ModelStrategist(), lambda: ModelStrategist(producers="equivalent"),
    lambda: ModelStrategist(producers="equivalent", fallbacks=False)], ids=["declared", "equivalent", "no_fallbacks"])
@pytest.mark.parametrize("state, order, answer", [({"net_in": 100.0}, ["net", "gross"], "yes"),
                                                  ({"gross_in": 100.0}, ["gross", "net"], "no")])
def test_facts_derivable_from_each_other_are_planned_run_and_replayed(strategist, state, order, answer):
    cat, qs = _net_gross()
    st = strategist()
    s = System(cat, qs, strategist=st)
    res = s.ask(dict(state))
    assert st.last["fallback"] is None
    assert [x.part.name for x in res.flow.steps] == order + ["answer:big"]
    assert (res["big"].answer, res["big"].status, res["big"].confidence) == (answer, "ok", 1.0)
    for step in res.flow.steps[:2]:                                # no producer kept in the flow reads its own fact back
        assert len(step.part.alternatives) == 1
    assert res.trace.replay(s, res.flow)["ok"]


def test_a_fallback_producer_is_dropped_only_when_it_would_read_its_own_fact():
    cat, qs = _net_gross()
    st = ModelStrategist(producers="equivalent")
    res = System(cat, qs, strategist=st).ask({"net_in": 100.0, "gross_in": 130.0})
    alts = {s.part.name: [a.name for a in s.part.alternatives] for s in res.flow.steps[:2]}
    assert alts["net"][0] == "net_given" and alts["gross"][0] == "gross_given"   # both given: the cheap ones first,
    assert sorted(len(v) for v in alts.values()) == [1, 2]                       # and one derived fallback, not both
    assert res["big"].answer == "yes"


def test_path_confidence_follows_the_producer_that_ran_and_survives_a_ring_of_facts():
    from solvi.runtime import path_confidence
    cat, qs = _net_gross()
    res = System(cat, qs, strategist=ModelStrategist(producers="equivalent", fallbacks=False)).ask({"net_in": 100.0})
    assert path_confidence(cat, res.trace, ["gross", "net"]) == 1.0            # the full catalog: net ⇄ gross


def test_solvi_check_calls_facts_derived_from_each_other_a_cycle_only_for_the_deterministic_strategist():
    from solvi.check import lint
    cat, qs = _net_gross()
    rep = lint(System(cat, qs, strategist=ModelStrategist()))
    assert rep.ok and rep.codes() == ["mutual_producers"]
    rep = lint(System(cat, qs))                                     # the deterministic strategist cannot plan it
    assert "cycle" in rep.codes("error") and "ModelStrategist" in str(rep)
    cat = Catalog()                                                 # a loop with no way in stays an error whatever plans

    @cat.fn
    def a(b): return b

    @cat.fn
    def b(a): return a

    @cat.rule("q")
    def q(a) -> bool: return True
    assert "cycle" in lint(System(cat, [Question("q", "", None)], strategist=ModelStrategist())).codes("error")


def test_fit_serve_and_check_plan_with_the_systems_strategist_as_ask_does():
    from solvi.check import lint
    from solvi.serve import question_inputs
    cat, qs = chain()
    s = System(cat, qs, strategist=ModelStrategist())
    assert s.ask(ST)["ok"].answer == "approved"
    assert {"a2", "a3"} <= set(s.facts_for(ST))                     # computed around the dead-end producer, as ask does
    info = question_inputs(s, "ok")
    assert "partner_feed" in info["properties"] and "partner_feed" not in info["required"]   # the dead end's input is
    assert lint(s).ok                                                                         # optional, not required
    det = System(*chain())
    assert "a2" not in det.facts_for(ST)                            # the deterministic strategist: unchanged
    assert "partner_feed" in question_inputs(det, "ok")["required"]


# ---------------------------------------------------------------- aliases.apply keeps every declaration
def _declared_catalog():
    from solvi import Quote
    cat = Catalog()

    @cat.extract(timeout=2.0, blocking=True, cost=5, source="mail", min_confidence=0.3, exact=False)
    def total(mail):
        "the total"
        return Quote(mail[:3], 0, 3, confidence=0.9)

    @cat.fn(timeout=1.5, blocking=True, cost=7, validate=lambda v: v > 0)
    def amount(total) -> float:
        return 1.0

    @cat.fn(provides="vat", timeout=0.5, blocking=True)
    def vat_fast(amount: float) -> float:
        return amount / 5

    @cat.fn(provides="vat", cost=3)
    def vat_slow(amount: float) -> float:
        return amount / 5

    @cat.check(hard=True, then={"ok": "no"}, timeout=3.0, blocking=True, cost=2)
    def small(amount) -> bool:
        return amount < 10

    @cat.rule("ok", timeout=4.0, blocking=True)
    def ok(amount, vat) -> bool:
        return True
    return cat


@pytest.mark.parametrize("aliases", [{}, {"mail": "message"}], ids=["no_aliases", "one_alias"])
def test_apply_keeps_every_declaration_of_every_part_and_rule(aliases):
    import dataclasses
    from solvi.aliases import apply
    cat = _declared_catalog()
    new = apply(cat, aliases)
    rewired = {"func", "tin", "tout", "alternatives", "validate", "doc", "inputs", "source"}   # compared below or by behaviour
    pairs = [(p, new.parts[n]) for n, p in cat.parts.items()] + [(r, new.rules[q]) for q, r in cat.rules.items()]
    pairs += [(a, b) for n, p in cat.parts.items() if p.alternatives
              for a, b in zip(p.alternatives, new.parts[n].alternatives)]
    assert len(pairs) == 7
    for was, now in pairs:
        diff = {f.name: (getattr(was, f.name), getattr(now, f.name)) for f in dataclasses.fields(was)
                if f.name not in rewired and getattr(was, f.name) != getattr(now, f.name)}
        assert not diff, (was.name, diff)
        assert [aliases.get(x, x) for x in was.inputs] == now.inputs
        assert (was.validate is None) == (now.validate is None)
    assert new.parts["total"].source == aliases.get("mail", "mail")
    assert (new.parts["amount"].timeout, new.parts["amount"].blocking) == (1.5, True)
    assert (new.rules["ok"].timeout, new.rules["ok"].blocking) == (4.0, True)


def test_a_part_that_times_out_still_times_out_after_apply():
    import asyncio
    from solvi.aliases import apply
    cat = Catalog()

    @cat.fn(timeout=0.05, blocking=True)
    def slow(x):
        time.sleep(0.3)
        return x

    @cat.rule("q")
    def q(slow) -> bool:
        return True
    for c in (cat, apply(cat, {})):
        r = asyncio.run(System(c, [Question("q", "", None)]).aask({"x": 1}))["q"]
        assert (r.answer, r.status, r.guard) == (None, "abstain", "timeout")
