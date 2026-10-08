"""Replay determinism: the same input gives the same hashes and answers; a fresh, a reloaded and a stored decision
(full or compact) replay ok, every time, with the same result. docs/trace_spec.md §5."""
import pytest
from semantics_helpers import SEEDS, random_input, rng_for, shuffled_keys, system

from solvi import Response
from solvi.core.store import JSONLStorage, entry
from solvi.core.provenance import KINDS

SUMMARY = ("init_hash", "answers", "flow", "records", "guards", "safeguards", "models", "catalog")


def _summary(res):
    e = entry(res)
    return {k: e.get(k) for k in SUMMARY}


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_input_gives_the_same_hashes_and_answers(seed):
    rng = rng_for(seed)
    state = random_input(rng)
    a, b = system().ask(state), system().ask(dict(state))           # two Systems built alike, two asks
    assert _summary(a) == _summary(b)
    c = system().ask(shuffled_keys(state, rng))                      # the input's key order does not matter
    assert _summary(c) == _summary(a)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_decision_replays_ok_fresh_reloaded_and_twice(seed):
    s = system()
    res = s.ask(random_input(rng_for(seed)))
    first = res.trace.replay(s, res.flow)
    assert first["ok"] and first["answers"] == "same" and first["catalog"] == "same"
    assert res.trace.replay(s, res.flow) == first                    # replay changes nothing it reads
    back = Response.from_json(res.to_json())
    rep = back.trace.replay(s, back.flow)
    assert rep["ok"] and rep["answers"] == "same"
    assert {r.origin for r in res.trace.records} <= set(KINDS)


@pytest.mark.parametrize("record", ["full", "compact", "sample:2"])
@pytest.mark.parametrize("seed", range(4))
def test_every_stored_decision_replays_and_rederives_to_its_answers(tmp_path, seed, record):
    s = system()
    rng = rng_for(seed)
    st = JSONLStorage(tmp_path / "s.jsonl", system=s, record=record)
    asked = [s.ask(random_input(rng)) for _ in range(5)]
    ids = [st.save(r) for r in asked]
    assert st.verify()["ok"]
    assert st.replay_all(s) == [] and st.replay_all(s) == []
    for i, r in zip(ids, asked):
        back = st.rederive(i, s)
        assert {q: (x.answer, x.status) for q, x in back.results.items()} == \
               {q: (x.answer, x.status) for q, x in r.results.items()}
        assert [x.hash for x in back.trace.records] == [x.hash for x in r.trace.records]
    reopened = JSONLStorage(tmp_path / "s.jsonl", system=s)
    assert reopened.replay_all(s) == []


@pytest.mark.parametrize("seed", range(4))
def test_a_changed_rule_is_told_from_damaged_data(seed):
    """Replay against another catalog: the verdict says the catalog changed, never that the data is damaged."""
    s = system()
    res = s.ask(random_input(rng_for(seed)))
    rep = res.trace.replay(system(big=-1), res.flow)              # every amount is "big" now
    assert rep["catalog"] == "changed" and "data damaged" not in (rep.get("summary") or "")
    if res["size"].answer == "small":
        assert not rep["ok"] and rep["kinds"].keys() <= {"recompute"}
