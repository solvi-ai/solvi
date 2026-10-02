"""The learned strategist: a learned order of hard checks gives the same answers as the default order; alternative producers
of one fact fall back, are recorded in the trace and replay; single-producer catalogs are unchanged."""
import copy
import inspect
import random
import time

from solvi import Answer, Catalog, Question, Quote, System
from solvi.costs import CostBook
from solvi.runtime import vhash


def _fn(name, inputs, body):
    def f(**kw):
        return body(*[kw[x] for x in inputs])
    f.__name__ = name
    f.__signature__ = inspect.Signature([inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD) for x in inputs])
    return f


def random_system(rng):
    """A random catalog: numeric inputs, computed facts, 2-5 hard checks (some with `then`, some checkpoints, some failing
    often), soft checks, 2-4 questions with rules."""
    cat = Catalog()
    xs = [f"x{i}" for i in range(5)]
    facts = list(xs)
    for k in range(rng.randint(3, 6)):
        ins = rng.sample(facts, rng.randint(1, 2))
        a, b = rng.randint(1, 5), rng.randint(0, 9)
        cat.fn(_fn(f"f{k}", ins, lambda *v, a=a, b=b: (a * sum(v) + b) % 10))
        facts.append(f"f{k}")
    qnames = [f"q{j}" for j in range(rng.randint(2, 4))]
    options = ["a", "b", "c"]
    hard = []
    for k in range(rng.randint(2, 5)):
        ins = rng.sample(facts, rng.randint(1, 2))
        m = rng.choice([2, 3, 4, 7])
        then = {} if rng.random() < 0.3 else {q: rng.choice(options) for q in rng.sample(qnames, rng.randint(1, len(qnames)))}
        cat.check(hard=True, then=then)(_fn(f"h{k}", ins, lambda *v, m=m: sum(v) % m != 0))
        hard.append(f"h{k}")
    for k in range(rng.randint(0, 2)):
        ins = rng.sample(facts[5:] or facts, 1)
        cat.check(_fn(f"s{k}", ins, lambda v: v < 7))
    qs = []
    for q in qnames:
        ins = rng.sample(facts[5:] or facts, rng.randint(1, 2))
        cat.rule(q)(_fn(q, ins, lambda *v: options[sum(v) % 3]))
        qs.append(Question(q, q, Answer.choice(options), requires=rng.sample(hard, rng.randint(0, 2))))
    return cat, qs


class RandomOrder:
    """Any P(fail) must give the same answers: a random order model."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.features = []

    def row(self, vals, init_keys):
        return {}

    def p_fail(self, check, row):
        return self.rng.random()


def _key(res):
    return {q: (r.answer, r.status, r.why, round(r.confidence, 9)) for q, r in res.results.items()}


def test_learned_order_gives_identical_answers_on_random_catalogs():
    rng = random.Random(0)
    n_cases = n_multi = 0
    for c in range(40):
        cat, qs = random_system(rng)
        sys_default = System(cat, qs)
        sys_learned = System(cat, qs, order="learned")
        for i in range(40):
            state = {f"x{j}": rng.randint(0, 9) for j in range(5)}
            names = None if rng.random() < 0.6 else rng.sample([q.name for q in qs], rng.randint(1, len(qs)))
            d = sys_default.ask(state, names)
            runs = [sys_learned.ask(state, names), sys_default.ask(state, names, order=RandomOrder(c * 100 + i))]
            if i % 8 == 0:
                runs.append(sys_learned.ask(state, names, workers=3))           # parallel steps inside the learned order
            for res in runs:
                assert _key(res) == _key(d), (c, i)
                assert res.trace.replay(cat, res.flow)["ok"]
                steps = [r.step for r in res.trace.records]
                assert steps == sorted(steps)                      # records stay in flow order
            failed = [r.name for r in d.trace.records if r.name.startswith("h") and r.value is False]
            n_cases += 1
            n_multi += len(failed) >= 2
    assert n_multi > 100                                          # several failing hard checks were exercised


def _claim_catalog(calls):
    cat = Catalog()

    @cat.fn
    def slow_lookup(customer):
        calls.append("slow_lookup")
        return customer.startswith("bad")

    @cat.fn
    def big_model(amount):
        calls.append("big_model")
        return amount * 2

    @cat.check(hard=True, then={"decision": "deny"})
    def expensive_ok(slow_lookup):
        return not slow_lookup

    @cat.check(hard=True, then={"decision": "deny"})
    def cheap_ok(amount):
        return amount < 1000

    @cat.rule("decision")
    def decision(big_model):
        return "pay" if big_model < 1500 else "review"
    return cat, [Question("decision", "?", Answer.choice(["pay", "deny", "review"]), requires=["expensive_ok", "cheap_ok"])]


def test_learned_order_runs_likely_failure_first_and_explains_it():
    calls = []
    cat, qs = _claim_catalog(calls)
    s = System(cat, qs)
    s.cost_book.ms.update({"slow_lookup": 100.0, "expensive_ok": 0.01, "cheap_ok": 0.01, "big_model": 50.0, "answer:decision": 0.01})
    s.learn = False
    train = [{"customer": "ok", "amount": a} for a in range(0, 2000, 20)]
    s.learn_order(train)
    assert s.order == "learned"
    calls.clear()
    # "cheap_ok" is declared after "expensive_ok": when it fails, expensive_ok must still be checked (it could decide)
    r = s.ask({"customer": "ok", "amount": 1500})
    assert r["decision"].answer == "deny" and r.trace.schedule[0]["check"] == "cheap_ok"
    assert "slow_lookup" in calls and "big_model" not in calls
    assert "P(fail)" in r.trace.explain_order()
    # when the earlier-declared check fails too, it decides — same as the default order
    d = System(cat, qs).ask({"customer": "bad", "amount": 1500})
    assert _key(s.ask({"customer": "bad", "amount": 1500})) == _key(d)


def test_costs_are_tracked():
    calls = []
    cat, qs = _claim_catalog(calls)
    s = System(cat, qs)
    s.ask({"customer": "ok", "amount": 10})
    assert {"slow_lookup", "big_model", "cheap_ok"} <= set(s.cost_book.ms)
    assert isinstance(s.cost_book, CostBook)


# ---------- alternative producers

DOC = "SHOP\nTotal 12.50\nCash 20.00\n"
DOC_NO_KEYWORD = "SHOP\nsum due: 7.25\n"


def _receipts(model_calls, regex_validate=True):
    cat = Catalog()
    import re

    @cat.fn(provides="total", cost=0.1, validate=(lambda v: v > 0) if regex_validate else None)
    def total_regex(doc):
        m = re.search(r"Total\s+(\d+\.\d\d)", doc)
        return None if m is None else float(m.group(1))

    @cat.extract(provides="total", cost=50, min_confidence=0.5)
    def total_model(doc):
        model_calls.append(doc)
        m = re.search(r"(\d+\.\d\d)", doc)
        return Quote(float(m.group(1)), m.start(1), m.end(1), confidence=0.9 if "due" in doc or "Total" in doc else 0.2)

    @cat.rule("reimburse")
    def reimburse(total):
        return total <= 10
    return cat, [Question("reimburse", "?", Answer.yes_no())]


def test_fallback_chain_and_trace():
    calls = []
    cat, qs = _receipts(calls)
    s = System(cat, qs)
    r = s.ask({"doc": DOC})
    rec = next(x for x in r.trace.records if x.name == "total")
    assert r["reimburse"].answer == "no" and rec.producer == "total_regex" and calls == []
    assert rec.tried == [["total_regex", "accepted"]]
    r = s.ask({"doc": DOC_NO_KEYWORD})
    rec = next(x for x in r.trace.records if x.name == "total")
    assert rec.producer == "total_model" and rec.value == 7.25 and rec.quote is not None and rec.kind == "extract"
    assert rec.tried == [["total_regex", "no value"], ["total_model", "accepted"]]
    assert r["reimburse"].answer == "yes"
    for doc in (DOC, DOC_NO_KEYWORD):
        res = s.ask({"doc": doc})
        assert res.trace.replay(cat, res.flow)["ok"]
    r = s.ask({"doc": "nothing 1.00 here"})                       # model below min_confidence, regex finds nothing
    rec = next(x for x in r.trace.records if x.name == "total")
    assert rec.producer is None and rec.error.startswith("no producer accepted") and r["reimburse"].status == "abstain"
    assert r.trace.replay(cat, r.flow)["ok"]


def test_replay_catches_a_faked_producer():
    cat, qs = _receipts([])
    r = System(cat, qs).ask({"doc": DOC_NO_KEYWORD})
    t = copy.deepcopy(r.trace)
    rec = next(x for x in t.records if x.name == "total")
    rec.producer, rec.tried = "total_regex", [["total_regex", "accepted"]]         # claim the cheap producer found it
    prev = t.init_hash
    for x in t.records:
        x.prev = prev
        x.hash = vhash(x.body())
        prev = x.hash
    rep = t.replay(cat)
    assert not rep["ok"] and "not accepted on replay" in rep["mismatches"][0][2]
    t = copy.deepcopy(r.trace)
    next(x for x in t.records if x.name == "total").tried[0][1] = "shadow: agrees"
    assert not t.replay(cat)["ok"]                                  # edited tried list: the hash chain breaks


def test_single_producer_catalog_unchanged():
    cat = Catalog()

    @cat.fn
    def double(x):
        return 2 * x

    @cat.rule("big")
    def big(double):
        return double > 10
    r = System(cat, [Question("big", "?", Answer.yes_no())]).ask({"x": 7})
    rec = r.trace.records[0]
    assert rec.tried is None and "producer" not in rec.body() and cat.parts["double"].alternatives is None
    assert r.trace.replay(cat, r.flow)["ok"]


def test_plain_part_can_gain_an_alternative_and_group_func_works():
    cat = Catalog()

    @cat.fn
    def total(doc):
        return None if "x" in doc else 1.0

    @cat.fn(provides="total")
    def total_backup(doc):
        return 2.0
    g = cat.parts["total"]
    assert [a.name for a in g.alternatives] == ["total__declared", "total_backup"]
    assert g.func(doc="a") == 1.0 and g.func(doc="x") == 2.0          # plain call: declaration order


def test_learned_producer_policy_avoids_a_cheap_producer_where_it_is_wrong():
    import re
    cat = Catalog()
    calls = []

    @cat.fn(provides="total", cost=0.1)
    def total_regex(doc):
        m = re.search(r"(\d+\.\d\d)", doc)                            # first amount: wrong on "long" receipts
        return None if m is None else float(m.group(1))

    @cat.fn(provides="total", cost=50)
    def total_model(doc):
        calls.append(1)
        time.sleep(0.002)                                               # a model is slower than a regex
        return float(re.findall(r"(\d+\.\d\d)", doc)[-1])

    @cat.features("total")
    def total_features(doc):
        return {"long": doc.count("\n") > 2}

    @cat.rule("amount")
    def amount(total):
        return "high" if total > 50 else "low"
    s = System(cat, [Question("amount", "?", Answer.choice(["low", "high"]))], producers="learned")
    s.producer_policy.explore = 0.3
    rng = random.Random(1)

    def doc(long):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return f"A {a}.00\nB {b}.00\nC 1.00\nT {a + b}.00" if long else f"T {a}.00"
    for i in range(300):
        s.ask({"doc": doc(i % 2 == 0)})
    s.producer_policy.explore = 0.0
    short = s.ask({"doc": doc(False)})
    long = s.ask({"doc": doc(True)})
    assert next(r for r in short.trace.records if r.name == "total").producer == "total_regex"
    assert next(r for r in long.trace.records if r.name == "total").producer == "total_model"
    assert long.trace.replay(cat, long.flow)["ok"] and short.trace.replay(cat, short.flow)["ok"]


def test_learn_order_turns_online_learning_on_and_a_default_system_does_not_learn_from_asks():
    """The docs said every ask feeds the order model; on a default System (learn=False) none did, and learn_order()
    without examples switched to an empty model that then stayed empty."""
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.check(hard=True, then={"ship": "no"})
    def paid(status):
        return status == "paid"

    @cat.rule("ship")
    def ship(status):
        return "yes"
    qs = [Question("ship", "Ship?", Answer.yes_no(), requires=["paid"])]
    states = [{"status": "paid" if i % 4 else "unpaid"} for i in range(40)]
    s = System(cat, qs)
    for st in states:
        s.ask(st)
    assert s.learn is False and s.order_model.n("paid") == 0
    s.learn_order()
    assert s.learn is True and s.order == "learned"
    for st in states:
        s.ask(st)
    assert s.order_model.n("paid") == 40
    assert System(cat, qs, order="learned").learn is True and System(cat, qs, learn=True).learn is True
