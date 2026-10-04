"""solvi.core.knowledge.actions and .risk: the conservative action model over an explicit vocabulary (accept / refuse /
unknown with risk and support, hard rules, effects, MISSING conditions, journal and replay, commit as action items) and
the risk policies (Protect: the verdict is final; RiskBudget: justified risk within a budget, hard rules never traded;
LearnedGate: bounded by an expiry and a floor, reopened by evidence)."""
import pytest

from solvi.core.knowledge import (ActionModel, ConservativeActionModel, KnowledgeStore, LearnedGate, Prediction, Protect,
                                  RiskBudget, RiskPolicy, Vocabulary)


def shop():
    """A toy shop: cancel needs a pending order; refund goes only to the original method; lava kills (hard)."""
    vocab = Vocabulary({"status": lambda s, a: s["orders"][a["order"]],
                        "original_method": lambda s, a: a.get("method") == s["paid_with"][a["order"]],
                        "lava": lambda s, a: bool(s.get("lava"))}, hard={"lava"})
    return vocab


def state(status="pending", paid="card", lava=False):
    return {"orders": {"A": status}, "paid_with": {"A": paid}, "lava": lava}


def env(st, action, args):
    """The environment's own checks → (accepted, effect)."""
    if st["lava"]:
        return False, None
    if action == "cancel":
        return st["orders"][args["order"]] == "pending", {"status": "cancelled"}
    if action == "refund":
        return args.get("method") == st["paid_with"][args["order"]], {"refunded": True}
    return True, None


def trained():
    am = ConservativeActionModel(shop())
    for st, act, args in [(state(), "cancel", {"order": "A"}), (state("delivered"), "cancel", {"order": "A"}),
                          (state("shipped"), "cancel", {"order": "A"}),
                          (state(), "refund", {"order": "A", "method": "card"}),
                          (state(), "refund", {"order": "A", "method": "gift"}),
                          (state(lava=True), "cancel", {"order": "A"}), (state("pending"), "cancel", {"order": "A"})]:
        ok, eff = env(st, act, args)
        am.observe(st, act, args, ok, eff if ok else None)
    return am


def test_the_protocols():
    assert isinstance(trained(), ActionModel) and isinstance(Protect(), RiskPolicy) and isinstance(RiskBudget(), RiskPolicy)
    with pytest.raises(TypeError, match="vocabulary is an explicit"):
        ConservativeActionModel({"status": lambda s, a: 1})                     # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Vocabulary({"a": lambda s, a: 1}, hard={"b"})
    with pytest.raises(ValueError):
        Prediction("maybe")


def test_accept_refuse_unknown_with_risk_support_and_effects():
    am = trained()
    p = am.predict(state(), "cancel", {"order": "A"})
    assert (p.verdict, p.support, p.effects) == ("accept", 2, {"status": "cancelled"}) and 0 < p.risk < 0.5
    r = am.predict(state("delivered"), "cancel", {"order": "A"})
    assert r.verdict == "refuse" and r.support == 1 and "status='delivered'" in r.reason and not r.hard
    u = am.predict(state("returned"), "cancel", {"order": "A"})                 # a status never seen: no evidence
    assert u.verdict == "unknown" and "no evidence" in u.reason
    assert am.predict(state(), "exchange", {"order": "A"}).reason == "action never observed"
    rf = am.predict(state(paid="gift"), "refund", {"order": "A", "method": "card"})
    assert rf.verdict == "refuse" and "original_method=False" in rf.reason
    lava = am.predict(state(lava=True), "cancel", {"order": "A"})
    assert lava.verdict == "refuse" and lava.hard                               # a hard rule stands behind it


def test_an_unreadable_condition_is_unknown_not_a_guess():
    am = trained()
    p = am.predict({"orders": {}, "paid_with": {}}, "cancel", {"order": "A"})   # the order was not looked up
    assert p.verdict == "unknown"


def test_refusals_the_vocabulary_cannot_explain_answer_only_seen_vectors():
    vocab = Vocabulary({"status": lambda s, a: s["status"]})
    am = ConservativeActionModel(vocab)
    am.observe({"status": "pending"}, "cancel", {}, True)
    am.observe({"status": "pending"}, "cancel", {"amount": 900}, False)        # refused for a reason not in the vocabulary
    assert am.predict({"status": "pending"}, "cancel", {}).verdict == "unknown"
    assert am.summary("cancel")["unexplained"] == 1


def test_a_surprise_is_counted_and_the_prediction_before_returned():
    am = trained()
    before = am.observe(state(), "cancel", {"order": "A"}, accepted=False)     # predicted accept, refused: a surprise
    assert before.verdict == "accept" and am.surprises == 1


def test_journal_replay_commit_and_fingerprint():
    ks = KnowledgeStore()
    am = ConservativeActionModel(shop(), store=ks)
    for st in (state(), state("delivered")):
        ok, eff = env(st, "cancel", {"order": "A"})
        am.observe(st, "cancel", {"order": "A"}, ok, eff if ok else None)
    again = ConservativeActionModel.replay(ks, shop())
    assert again.fingerprint() == am.fingerprint()
    ids = am.commit()
    rec = ks.item(ids["cancel"])
    assert rec["kind"] == "action" and rec["status"] == "active" and rec["source"] == "outcome"
    assert set(rec["body"]["o"]["predicates"]) == {"status", "original_method", "lava"}
    am.observe(state("shipped"), "cancel", {"order": "A"}, False)
    newer = am.commit()["cancel"]
    assert ks.status(ids["cancel"]) == "refuted" and ks.status(newer) == "active"
    other = Vocabulary({"status": lambda s, a: s["orders"][a["order"]].upper()})
    assert other.fingerprint() != shop().fingerprint()


# --- risk policies
def test_protect_keeps_the_verdict():
    p = Protect()
    assert p.decide("x", Prediction("refuse", 0.01, 50, "r"), gain=100).choice == "avoid"
    assert p.decide("x", Prediction("unknown")).choice == "ask_s2"
    assert p.decide("x", Prediction("accept")).choice == "take"


def test_risk_budget_takes_justified_risk_and_never_trades_a_hard_rule():
    rb = RiskBudget(max_risk_per_episode=0.5, min_gain_ratio=1.0, min_support=3)
    assert rb.decide("melee", Prediction("refuse", 0.2, 10, "eye"), gain=0.5).choice == "take"
    assert rb.left == pytest.approx(0.3)
    assert rb.decide("melee", Prediction("refuse", 0.2, 10, "eye"), gain=0.1).choice == "avoid"      # gain too small
    assert rb.decide("eat", Prediction("refuse", 0.4, 10, "x"), gain=1.0).reason.startswith("risk 0.4 > budget")
    assert rb.decide("eat", Prediction("refuse", 0.01, 1, "x"), gain=1.0).choice == "ask_s2"        # ambiguous
    assert rb.decide("eat", Prediction("refuse", 0.01, -1, "spec rate"), gain=1.0).choice == "take"  # a spec's rate
    hard = rb.decide("eat", Prediction("refuse", 0.0, 99, "cockatrice", hard=True), gain=100)
    assert hard.choice == "avoid" and "never traded" in hard.reason
    d = rb.decide("fight", Prediction("refuse", 0.1, 5, "x"), gain=1, key=("fight", 3))
    assert d.choice == "take" and rb.decide("fight", Prediction("refuse", 0.1, 5, "x"), gain=0, key=("fight", 3)).choice == "take"
    assert set(d.to_dict()) == {"choice", "reason", "risk", "support", "gain", "budget_left", "action"}
    rb.new_episode()
    assert rb.left == 0.5 and rb.taken == set()


def test_a_learned_gate_has_an_expiry_a_floor_and_is_reopened_by_evidence():
    with pytest.raises(ValueError):
        LearnedGate("depth", expiry=0)
    g = LearnedGate("depth", expiry=3, floor=lambda xl: xl + 1, near=1, min_support=3, margin=1)
    for d in (3, 3, 3):                       # three deaths at depth 3 at level 4: the median − 1 = 2, but floor 5
        g.failed(4, d)
        g.end_episode()
    assert g.limit(4) == 5                    # never below the floor: no unbounded self-tightening
    assert g.predict(4, 5).verdict == "accept" and g.predict(4, 6).verdict == "refuse"
    assert g.predict(4, 6).risk == 0.5 and g.predict(4, 6).support == 0          # the prior, no evidence yet
    for _ in range(3):
        g.arrived(4, 9, failed=False)         # evidence: three arrivals at depth 9, none died there
        g.end_episode()                       # the deaths expire after 3 episodes
    assert g.limit(4) is None
    for d in (8, 8, 8):
        g.failed(4, d)
    g.end_episode()
    p = g.predict(4, 9)
    assert p.verdict == "refuse" and p.support == 3 and p.risk == pytest.approx(1 / 5)   # reopened by evidence
    assert RiskBudget(1.0, 1.0, 3).decide("descend", p, gain=0.3).choice == "take"
    assert Protect().decide("descend", p, gain=0.3).choice == "avoid"
