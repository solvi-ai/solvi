"""diff(incremental=True) re-runs only the stored decisions something changed under, and reports what the full diff
reports; diff(facts=True) lists the computed facts that changed."""
import random

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.store import JSONLStorage
from solvi.core.store.diff import diff


class Scorer:
    model_id = "scorer"

    def __init__(self, version="1"):
        self.version = version


def build(threshold=100, version="1", min_confidence=None, cutoff=3, extra_rule=False, words_upper=False):
    cat = Catalog()
    scorer = Scorer(version)

    @cat.fn
    def words(text):
        return set((text.upper() if words_upper else text.lower()).split())

    @cat.fn
    def size(amount):
        return "big" if amount >= cutoff * 100 else "small"

    @cat.check(hard=True, then={"approve": False})
    def known_customer(customer):
        return customer != "blocked"

    @cat.fn(model=scorer, options=["low", "high"])
    def risk(words):
        high = "urgent" in words or (scorer.version == "2" and "please" in words)
        return Decision("high", {"high": 0.9, "low": 0.1}) if high else Decision("low", {"high": 0.2, "low": 0.8})

    @cat.rule("approve")
    def approve(amount, risk) -> bool:
        return amount < threshold and risk == "low"

    @cat.rule("urgent")
    def urgent(words) -> bool:
        return "urgent" in words

    @cat.rule("big")
    def big(size) -> bool:
        return size == "big"

    qs = [Question("approve", "Approve?", Answer.yes_no(), requires=["known_customer"], min_confidence=min_confidence),
          Question("urgent", "Urgent?", Answer.yes_no()),
          Question("big", "Big?", Answer.yes_no())]
    if extra_rule:
        @cat.rule("vip")
        def vip(customer) -> bool:
            return customer == "ann"
        qs.append(Question("vip", "VIP?", Answer.yes_no()))
    return System(cat, qs)


def _states(seed, n=24):
    rng = random.Random(seed)
    texts = ["please pay", "urgent pay now", "pay", "PAY please", "hello"]
    return [{"text": rng.choice(texts), "amount": rng.choice([10, 50, 250, 500, 900]),
             "customer": rng.choice(["ann", "bob", "blocked"])} for _ in range(n)]


def _store(tmp_path, seed):
    store = JSONLStorage(tmp_path / f"d{seed}.jsonl")
    s = build()
    s.storage = store
    store.catalog = s
    rng = random.Random(seed + 1)
    for st in _states(seed):
        q = rng.choice([None, ["approve"], ["urgent", "big"], ["big"]])
        s.ask(st, q)
    return store


VARIANTS = [{}, {"threshold": 40}, {"threshold": 600}, {"version": "2"}, {"min_confidence": 0.9}, {"cutoff": 6},
            {"words_upper": True}, {"extra_rule": True}, {"threshold": 600, "cutoff": 1}]


def _same(full, inc):
    a, b = full.to_dict(), inc.to_dict()
    b.pop("reused")
    assert a == b


def test_incremental_diff_reports_what_the_full_diff_reports(tmp_path):
    for seed in range(4):
        store = _store(tmp_path, seed)
        for v in VARIANTS:
            new = build(**v)
            full, inc = diff(store, new), diff(store, build(**v), incremental=True)
            _same(full, inc)
            assert inc.total == full.total and 0 <= inc.reused <= inc.total
            _same(diff(store, build(**v), facts=True), diff(store, build(**v), incremental=True, facts=True))


def test_incremental_diff_skips_decisions_the_change_cannot_reach(tmp_path):
    store = _store(tmp_path, 7)
    n = len(list(store.iter()))
    assert diff(store, build(), incremental=True).reused == n          # nothing changed: nothing re-run
    rep = diff(store, build(cutoff=6), incremental=True)               # only `size` changed: decisions without it reused
    with_size = sum("size" in [st["name"] for st in s.data["response"]["flow"]["steps"]] for s in store.iter())
    assert 0 < with_size < n and rep.reused == n - with_size
    assert diff(store, build(version="2"), incremental=True).reused < n   # a new model: decisions with `risk` re-run
    assert "not re-run" in str(rep)


def test_diff_facts_lists_the_changed_facts(tmp_path):
    store = _store(tmp_path, 3)
    rep = diff(store, build(cutoff=6), facts=True)
    assert rep.changed and all("facts" in c for c in rep.changed)
    only_facts = [c for c in rep.changed if not c["questions"]]
    for c in rep.changed:
        assert set(c["facts"]) <= {"size"}
    assert all(c["facts"] for c in only_facts)
    assert "fact size:" in str(rep)
    assert all("facts" not in c for c in diff(store, build(cutoff=6)).changed)   # default: answers only, as before
    # a fact that changed without changing any answer is listed only with facts=True
    plain_ids = {c["id"] for c in diff(store, build(cutoff=6)).changed}
    assert plain_ids <= {c["id"] for c in rep.changed}


def _module(tmp_path, name, src):
    import importlib.util
    p = tmp_path / f"{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CALLER = """
from solvi import Answer, Catalog, Question, System

def helper(x):
    return x + {step}

def system():
    cat = Catalog()

    @cat.fn
    def bumped(amount):
        return helper(amount) {op} 0

    @cat.rule("big")
    def big(bumped) -> bool:
        return bumped > 100

    return System(cat, [Question("big", "Big?", Answer.yes_no())])
"""


def test_diff_says_which_called_function_changed(tmp_path):
    store = JSONLStorage(tmp_path / "c.jsonl")
    s = _module(tmp_path, "m0", CALLER.format(step=1, op="+")).system()
    s.storage = store
    store.catalog = s
    for a in (50, 99, 150):
        s.ask({"amount": a})
    assert s.fingerprint() and store.get(next(iter(store.iter())).id).trace.fingerprint["calls"]["bumped"]
    callee = diff(store, _module(tmp_path, "m1", CALLER.format(step=2, op="+")).system())
    whys = {c["questions"]["big"]["first_step"]["why"] for c in callee.changed}
    assert whys == {"a function it calls changed: helper"}
    own = diff(store, _module(tmp_path, "m2", CALLER.format(step=1, op="* 2 +")).system())
    whys = {c["questions"]["big"]["first_step"]["why"] for c in own.changed}
    assert whys == {"its own code or declarations changed"}
    both = diff(store, _module(tmp_path, "m3", CALLER.format(step=2, op="* 2 +")).system())
    whys = {c["questions"]["big"]["first_step"]["why"] for c in both.changed}
    assert whys == {"its own code or declarations changed, and a function it calls changed: helper"}


def _two_checks(combine):
    cat = Catalog(combine=combine)

    @cat.check(hard=True, then={"route": "review"})
    def small_enough(amount):
        return amount < 300

    @cat.check(hard=True, then={"route": "reject"})
    def known(customer):
        return customer != "blocked"

    @cat.rule("route")
    def route(amount) -> str:
        return "approve"

    return System(cat, [Question("route", "Route?", Answer.choice(["approve", "review", "reject"]))])


def test_incremental_diff_sees_a_change_of_the_combine_mode(tmp_path):
    import warnings
    for seed in range(4):
        for was, now in (("first", "strictest"), ("strictest", "first"), ("first", "first"), ("strictest", "strictest")):
            store = JSONLStorage(tmp_path / f"c{seed}{was}{now}.jsonl")
            s = _two_checks(was)
            s.storage = store
            store.catalog = s
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for st in _states(seed):
                    s.ask(st)
                full, inc = diff(store, _two_checks(now)), diff(store, _two_checks(now), incremental=True)
            _same(full, inc)
            assert bool(full.changed) == (was != now)        # both checks fail on some states: the mode decides
            if was == now:
                assert inc.reused == inc.total
            for c in full.changed:                           # the why names the mode, not a step
                assert "Catalog(combine=) changed" in c["questions"]["route"]["first_step"]["why"]
