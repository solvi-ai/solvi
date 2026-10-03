"""One decider protocol: a decision part and a Cascade / Vote / Route have the same public methods with the same
signatures and result keys. What only a combination takes is keyword-only (scale=); what cannot mean anything for a
combination raises with the reason and where to call it instead."""
import inspect
import math

import pytest

from solvi.decide import DecisionPart
from solvi.multi import Cascade, Combination, Route, Vote
from test_guarantees import _labelled
from test_multi import CLEAR, _parts

# options a combination takes besides the part's: keyword-only, with a default
COMBINATION_ONLY_OPTIONS = {"act_guard": {"scale"}, "calibrate_for": {"scale"}}
# the part's own thresholds, by the question-level names; a combination's is `threshold`, on every part's signal
PART_ONLY_PROPERTIES = {"min_confidence", "min_act", "long_len"}
# a combination's structure and replay, not a decider's: the members, their states, the rule over the proposals
STRUCTURE = {"leaves", "parts", "state", "vec", "final", "entry", "check", "check_record", "same_question", "usage",
             "pick", "option_order"}
# what a part cannot do in a combination's place: raises NotImplementedError naming the part
ONE_PART_ONLY = {"adapt_lora": ([],), "save_lora": ("x",), "load_lora": ("x",), "budget": (), "sections_k": (),
                 "long_key": (), "long_input": ("x",), "in_pass": ([], {})}


def _public(cls):
    methods, props = {}, set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name)
        if isinstance(attr, property):
            props.add(name)
        elif callable(getattr(cls, name)):
            methods[name] = inspect.signature(getattr(cls, name))
    return methods, props


def _params(sig):
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


@pytest.mark.parametrize("comb", [Combination, Cascade, Vote, Route])
def test_every_public_method_of_a_part_is_a_combinations_with_the_same_signature(comb):
    part, part_props = _public(DecisionPart)
    mine, my_props = _public(comb)
    assert sorted(set(part) - set(mine)) == []
    for name, sig in part.items():
        extra = COMBINATION_ONLY_OPTIONS.get(name, set())
        theirs = [p for p in _params(mine[name]) if p[0] not in extra]
        assert theirs == _params(sig), name
        for p in mine[name].parameters.values():
            if p.name in extra:
                assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty, (name, p)
    assert sorted(part_props - my_props - PART_ONLY_PROPERTIES) == []
    # the other way round: what a combination has, a part has too, except its structure
    assert sorted(set(mine) - set(part) - STRUCTURE) == []
    assert _params(mine["calls"]) == _params(part["calls"])


def test_calibrate_for_on_a_combination_returns_the_parts_keys_and_sets_its_shared_threshold():
    s, l_, _, _ = _parts()
    cal = _labelled()
    want = set(s.calibrate_for(cal, max_error=0.05))
    for c in (Cascade([s, l_]), Vote([s, l_])):
        fp = c.fingerprint()
        info = c.calibrate_for(cal, max_error=0.05)
        assert want <= set(info) and set(info) - want == {"calls_per_question", "scale"}
        assert info["signal"] == "shared" and info["method"] == "empirical" and info["n"] == len(cal)
        assert info["error"] <= 0.05 + 1e-12 and c.threshold == info["threshold"] and c.fingerprint() != fp
        assert c.guarantee["method"] == "empirical"
        answered = [d for d in c.decide([x for x, _ in cal]) if d.escalate is None]
        assert len(answered) / len(cal) == pytest.approx(info["answered"])
        lt = c.calibrate_for(cal, max_error=0.05, method="ltt", delta=0.1, scale="rank")
        assert lt["signal"] == "shared-rank" and lt["method"] == "ltt" and c.guarantee["delta"] == 0.1
        assert lt["error"] <= 0.05 or not math.isfinite(lt["threshold"])
    with pytest.raises(ValueError, match="each part's own"):
        Cascade([s, l_]).calibrate_for(cal, signal="act")


def test_calibrate_for_and_score_and_memory_route_to_every_part_of_a_combination():
    s, l_, _, _ = _parts()
    c = Cascade([s, l_])
    p = c.score(CLEAR)
    assert p == c.decide(CLEAR).probs and c.score([CLEAR, CLEAR]) == [p, p]
    mems = c.memory(k=3)
    assert [m.part for m in mems] == [s, l_] and s.correction_memory is mems[0] and mems[1].k == 3
    with pytest.raises(ValueError, match="belongs to one part"):
        c.memory(mems[0])
    assert c.memory(False) is None and s.correction_memory is None and l_.correction_memory is None
    assert c.remove_lora() == [None, None]
    assert c.labels == s.labels and c.task == s.task and c.multi is False and c.adaptation == [None, None]


@pytest.mark.parametrize("name", sorted(ONE_PART_ONLY))
def test_what_belongs_to_one_part_raises_on_a_combination_and_names_the_part(name):
    s, l_, _, _ = _parts()
    c = Cascade([s, l_], name="team")
    with pytest.raises(NotImplementedError, match=rf"Cascade.{name}\(\): .* call it on the part \(team.parts\[i\]"):
        getattr(c, name)(*ONE_PART_ONLY[name])


def test_a_part_counts_its_own_decisions_as_a_combination_does():
    s, l_, _, _ = _parts()
    assert s.calls() == {"asked": 0, "calls": {"0:team": 0}, "calls_per_question": 0.0}
    s.decide(CLEAR)
    s.decide([CLEAR, CLEAR])
    s(email=CLEAR)
    assert s.calls() == {"asked": 4, "calls": {"0:team": 4}, "calls_per_question": 1.0}
    Cascade([s, l_]).decide(CLEAR)                     # inside a combination it counts there, not here
    assert s.calls()["asked"] == 4
