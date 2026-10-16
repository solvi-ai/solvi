"""solvi.core.explain.search(exhaustive=n): on small enumerated domains every combination is re-run in one batch
without hashing, and every counterfactual among them is returned (`complete`); and laws.split(per_cost=True): the
probe with the most information per unit of cost."""
import time
import warnings
from typing import Literal

from solvi import Answer, Catalog, Question, System
from solvi.core.explain import Rerun, search

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import split


def loan():
    cat = Catalog()

    @cat.fn
    def score(job: Literal["none", "part", "full"], housing: Literal["rent", "own", "family"],
              history: Literal["bad", "thin", "good"]) -> int:
        return {"none": 0, "part": 1, "full": 2}[job] + {"rent": 0, "family": 0, "own": 1}[housing] \
            + {"bad": -1, "thin": 0, "good": 1}[history]

    @cat.rule("approve")
    def approve(score: int) -> bool:
        return score >= 3
    return System(cat, [Question("approve", "Approve?", Answer.yes_no())])


D = {"job": ["none", "part", "full"], "housing": ["rent", "own", "family"], "history": ["bad", "thin", "good"]}


def test_exhaustive_finds_every_single_change_and_the_search_finds_a_subset():
    s = loan()
    res = s.ask({"job": "part", "housing": "rent", "history": "good"})
    assert res["approve"].answer in ("no", False)
    full = search(res, "approve", domains=D, exhaustive=100)
    assert full.complete and full.evals == 6                     # 2 + 2 + 2 other values, re-run once each
    got = {(c.changes[0].fact, c.changes[0].to) for c in full.found}
    assert got == {("job", "full"), ("housing", "own")}
    probed = search(res, "approve", domains=D)
    assert not probed.complete
    assert {(c.changes[0].fact, c.changes[0].to) for c in probed.found} <= got


def test_exhaustive_goes_to_pairs_only_when_no_single_change_suffices_and_respects_its_limit():
    s = loan()
    res = s.ask({"job": "none", "housing": "rent", "history": "thin"})
    full = search(res, "approve", domains=D, exhaustive=100)
    assert full.complete and full.found and all(len(c.changes) == 2 for c in full.found)
    assert {frozenset((c.fact, c.to) for c in cf.changes) for cf in full.found} == {
        frozenset({("job", "full"), ("housing", "own")}), frozenset({("job", "full"), ("history", "good")})}
    small = search(res, "approve", domains=D, exhaustive=5)                 # 6 singles + 12 pairs > 5: the search as before
    assert not small.complete


def test_the_batch_rerun_gives_the_same_results_and_its_speed_is_measured_in_one_run():
    s = loan()
    res = s.ask({"job": "part", "housing": "rent", "history": "good"})
    rr = Rerun(res, s, "approve")
    changes = [{"job": j, "housing": h, "history": x} for j in ("none", "part", "full")
               for h in ("rent", "own", "family") for x in ("bad", "thin", "good")] * 4
    t0 = time.perf_counter()
    one = [rr(c) for c in changes]
    t1 = time.perf_counter()
    many = rr.many(changes)
    t2 = time.perf_counter()
    assert [(r.answer, r.status) for r in one] == [(r.answer, r.status) for r in many]
    print(f"per-call {1e3 * (t1 - t0) / len(changes):.3f} ms, batch {1e3 * (t2 - t1) / len(changes):.3f} ms, "
          f"ratio {(t1 - t0) / max(t2 - t1, 1e-9):.2f}")


def test_split_per_cost_prefers_cheap_information():
    hyps = [lambda s, a, k=k: (s * k) % 4 for k in range(4)]
    probes = [(1, "cheap"), (3, "dear")]
    always = lambda s, a: True                                                     # noqa: E731
    cost = {"cheap": 1.0, "dear": 10.0}
    default = split(hyps, always, probes, cost=lambda s, a: cost[a])
    per = split(hyps, always, probes, cost=lambda s, a: cost[a], per_cost=True)
    assert default.action == per.action == "cheap" and per.info > 0 and per.cost == 1.0
    hyps2 = [lambda s, a: 0, lambda s, a: 0, lambda s, a: 1, lambda s, a: 2]
    probes2 = [(0, "dear"), (0, "cheap")]
    pick = {"dear": [0, 1, 2, 3], "cheap": [0, 0, 1, 1]}
    hyps2 = [lambda s, a, i=i: pick[a][i] for i in range(4)]
    assert split(hyps2, always, probes2, cost=lambda s, a: cost[a]).action == "dear"          # larger split first
    assert split(hyps2, always, probes2, cost=lambda s, a: cost[a], per_cost=True).action == "cheap"   # 1 bit / 1
    import pytest
    with pytest.raises(ValueError):
        split(hyps2, always, probes2, per_cost=True)
