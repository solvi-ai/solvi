"""The five holes an audit found in the guarantees, each pinned as a property over seeded random cases
(docs/trace_spec.md §8). A hole still open is a test marked xfail(strict=True) with reason "open: ..." — it starts
failing the suite (XPASS) the day the hole is closed, so the mark is removed with the fix.

1  an input key named like a catalog part replaced the part (a hard check too; over HTTP; in Guard via facts=)
2  an untyped hard check / guard policy returning a falsy non-bool (numpy False, 0, None, []) counted as passed
3  cat.check(f, hard=True, then=...) with f passed directly registered a soft check
4  an answer edited in the stored JSON still replayed ok
5  a redacted record's kept answers could be edited and verify(anchor, signature) still passed"""
import copy
import json

import pytest
from semantics_helpers import (OPTIONS, SEEDS, STATUSES, MemoryStorage, fill, random_input, read_lines, rechain,
                               rng_for, system, write_lines)

from solvi import Answer, Catalog, Question, Response, System
from solvi.core.store import JSONLStorage, record_body, record_hash

OVER = {"amount": 9000, "limit": 100}                     # the cap fails: "pay" may never be answered "yes"


def _cap_system(body, typed=False, direct=False, then=None):
    then = then if then is not None else {"pay": "no"}
    cat = Catalog()
    if direct:
        def cap(amount, limit):
            return body(amount, limit)
        cat.check(cap, hard=True, then=then)
    elif typed:
        @cat.check(hard=True, then=then)
        def cap(amount, limit) -> bool:
            return body(amount, limit)
    else:
        @cat.check(hard=True, then=then)
        def cap(amount, limit):
            return body(amount, limit)

    @cat.fn
    def share(amount, limit) -> float:
        return amount / limit

    @cat.rule("pay")
    def pay(amount):
        return "yes"

    return cat, System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]), requires=["cap"])])


# --- hole 1: input keys cannot shadow parts
@pytest.mark.parametrize("seed", SEEDS)
def test_hole1_an_input_key_named_like_a_part_is_refused(seed):
    rng = rng_for(seed)
    cat, s = _cap_system(lambda a, l: a <= l, typed=True)
    name = rng.choice(["cap", "share"])
    value = rng.choice([True, False, None, 0, 1, "yes", [], 0.0])
    with pytest.raises(ValueError, match=f"'{name}' .*cannot stand in"):
        s.ask({**OVER, name: value})


@pytest.mark.parametrize("seed", SEEDS)
def test_hole1_any_other_extra_key_leaves_the_answers_alone(seed):
    rng = rng_for(seed)
    _, s = _cap_system(lambda a, l: a <= l, typed=True)
    state = rng.choice([OVER, {"amount": 5, "limit": 100}])
    key = rng.choice(["answer:pay", "pay", "then:pay", "guard:pay", "plan", "note", "Cap", "cap "])
    want = {q: (r.answer, r.status) for q, r in s.ask(state).results.items()}
    got = s.ask({**state, key: rng.choice(["yes", True, 1, None])})
    assert {q: (r.answer, r.status) for q, r in got.results.items()} == want
    assert got.trace.replay(s, got.flow)["ok"]


@pytest.mark.parametrize("seed", range(4))
def test_hole1_over_http_the_key_is_refused(seed):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    rng = rng_for(seed)
    _, s = _cap_system(lambda a, l: a <= l, typed=True)
    c = TestClient(create_app(system=s))
    r = c.post("/ask", json={"state": {**OVER, rng.choice(["cap", "share"]): rng.choice([True, None, 1])}})
    assert r.status_code == 422 and '"yes"' not in r.text


@pytest.mark.parametrize("seed", range(4))
def test_hole1_a_guard_fact_named_like_a_policy_is_refused(seed):
    from solvi.solutions.guard import Guard
    rng = rng_for(seed)
    paid = []
    g = Guard(fact_names={"role": str})

    @g.tool
    def send_payment(amount: float) -> str:
        """Pay."""
        paid.append(amount)
        return "paid"

    @g.policy("send_payment")
    def under_cap(amount) -> bool:
        return amount <= 100

    call = {"name": "send_payment", "arguments": {"amount": float(rng.randint(101, 10 ** 6))}}
    ctx = [{"role": "user", "content": f"pay {call['arguments']['amount']}"}]
    with pytest.raises(ValueError, match="under_cap"):
        g.call(call, context=ctx, facts={"role": "finance", "under_cap": rng.choice([True, None, 1])})
    assert paid == []


# --- hole 2: a hard check passes only on True
NON_BOOLS = [0, 0.0, None, [], "", {}, (), "no", "False", 1, "yes", [False], "ok", 2.5]


@pytest.mark.parametrize("seed", SEEDS)
def test_hole2_an_untyped_hard_check_passes_only_on_true(seed):
    """Anything but True (or numpy's True) never lets the guarded answer through: a non-bool rejects the step (the
    question abstains); False forces the `then` answer."""
    rng = rng_for(seed)
    v = rng.choice(NON_BOOLS)
    r = _cap_system(lambda a, l: v)[1].ask(OVER)["pay"]
    assert (r.answer, r.status) == (None, "abstain")
    r = _cap_system(lambda a, l: False)[1].ask({"amount": 5, "limit": 100})["pay"]
    assert (r.answer, r.status) == ("no", "forced")


def test_hole2_numpy_bools_are_bools():
    np = pytest.importorskip("numpy")
    for v, want in ((np.bool_(False), ("no", "forced")), (np.bool_(True), ("yes", "ok"))):
        for typed in (False, True):
            r = _cap_system(lambda a, l: v, typed=typed)[1].ask(OVER)["pay"]
            assert (r.answer, r.status) == want


@pytest.mark.parametrize("v", [1, "yes", "true", "1", 1.0, "True", 2, [True], "ok"])
def test_hole2_a_typed_check_passes_only_on_true_too(v):
    """A check declared `-> bool` is as strict as an untyped one: anything but True / False (or numpy's bools) rejects
    the step and the question abstains — its type no longer reads 1, "yes", "true" as True."""
    s = _cap_system(lambda a, l: v, typed=True)[1]
    res = s.ask(OVER)
    r = res["pay"]
    assert (r.answer, r.status, r.guard) == (None, "abstain", "hard_check")
    (c,) = res.checks
    assert c.status == "error" and "declared `-> bool` returns True or False" in c.reason
    assert res.trace.replay(s, res.flow)["ok"]


@pytest.mark.parametrize("v", [0, "no", "false", "0", 0.0, None, [], ""])
def test_hole2_a_typed_check_does_not_read_falsy_values_as_false(v):
    """...nor 0, "no" as False: they do not force the `then` answer either, the question abstains."""
    r = _cap_system(lambda a, l: v, typed=True)[1].ask({"amount": 5, "limit": 100})["pay"]
    assert (r.answer, r.status, r.guard) == (None, "abstain", "hard_check")


def test_hole2_a_check_typed_otherwise_keeps_its_type():
    """Only `-> bool` is made strict: a check declared with another return type is validated by that type."""
    from typing import Literal
    cat = Catalog()

    @cat.check(hard=True, then={"pay": "no"})
    def cap(amount, limit) -> Literal[True, False]:
        return amount <= limit

    @cat.rule("pay")
    def pay(amount):
        return "yes"

    s = System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]), requires=["cap"])])
    assert s.ask(OVER)["pay"].answer == "no" and s.ask({"amount": 5, "limit": 100})["pay"].answer == "yes"


@pytest.mark.parametrize("seed", range(6))
def test_hole2_an_untyped_guard_policy_allows_only_on_true(seed):
    from solvi.solutions.guard import Guard
    v = rng_for(seed).choice(NON_BOOLS)
    paid = []
    g = Guard()

    @g.tool
    def send_payment(amount: float) -> str:
        """Pay."""
        paid.append(amount)
        return "paid"

    @g.policy("send_payment")
    def under_cap(amount):
        return v

    d = g.call({"name": "send_payment", "arguments": {"amount": 5.0}}, context=[{"role": "user", "content": "pay 5"}])
    assert d.outcome != "allow" and paid == []


# --- hole 3: check(f, hard=True, then=...) passed directly keeps its options
@pytest.mark.parametrize("seed", SEEDS)
def test_hole3_check_with_the_function_passed_directly_is_the_decorator(seed):
    rng = rng_for(seed)
    then = rng.choice([{"pay": "no"}, {}, None])
    deco, s1 = _cap_system(lambda a, l: a <= l, then=then)
    direct, s2 = _cap_system(lambda a, l: a <= l, direct=True, then=then)
    for k in ("hard", "then", "kind", "inputs"):
        assert getattr(direct.parts["cap"], k) == getattr(deco.parts["cap"], k), k
    assert direct.parts["cap"].hard is True
    state = random_input(rng)
    assert {q: (r.answer, r.status) for q, r in s1.ask(state).results.items()} == \
           {q: (r.answer, r.status) for q, r in s2.ask(state).results.items()}


# --- hole 4: answers are bound to the trace
def _other(q, a, rng):
    return rng.choice([o for o in OPTIONS[q] if o != a])


@pytest.mark.parametrize("seed", SEEDS)
def test_hole4_an_answer_edited_in_the_response_json_does_not_replay(seed):
    rng = rng_for(seed)
    s = system()
    res = s.ask(random_input(rng))
    q = rng.choice(sorted(res.results))
    d = json.loads(res.to_json())
    r = d["results"][q]
    if rng.random() < 0.5 or r["answer"] is None:
        r["answer"] = _other(q, r["answer"], rng)
    else:
        r["status"] = rng.choice([x for x in STATUSES if x != r["status"]])
    forged = Response.from_json(json.dumps(d))
    rep = forged.trace.replay(s, forged.flow)
    assert not rep["ok"] and rep["answers"] == "differ" and rep["kinds"] == {"answer": 1}
    assert rep["summary"].startswith("data damaged")
    assert forged.trace.replay(s.catalog)["answers"] == "unchecked"          # a bare catalog cannot check answers


@pytest.mark.parametrize("record", ["full", "compact"])
@pytest.mark.parametrize("seed", SEEDS)
def test_hole4_an_answer_edited_in_the_store_with_every_hash_recomputed_does_not_replay(tmp_path, seed, record):
    rng = rng_for(seed)
    s = system()
    st = JSONLStorage(tmp_path / "s.jsonl", system=s, record=record)
    for _ in range(3):
        st.save(s.ask(random_input(rng)))
    recs = read_lines(st.path)
    i = rng.randrange(len(recs))
    q = rng.choice(sorted(recs[i]["answers"]))
    new = _other(q, recs[i]["answers"][q][0], rng)
    recs[i]["answers"][q][0] = new
    (recs[i].get("response") or recs[i].get("compact"))["results"][q]["answer"] = new
    write_lines(st.path, recs, rechain(recs))
    st = JSONLStorage(st.path, system=s)
    assert st.verify()["ok"]                               # a consistent rewrite: the chain alone cannot tell
    bad = st.replay_all(s)
    assert [b["seq"] for b in bad] == [i] and bad[0]["kinds"] == {"answer": 1}


@pytest.mark.parametrize("field,value", [("confidence", 0.37), ("why", "forged"), ("source", "cap"), ("guard", "hard_check"),
                                         ("confidence", 1.0 - 1e-6)])
def test_hole4_a_results_other_fields_are_bound_to_the_trace(field, value):
    """Every field a result is read for — confidence, why, guard, source, not only the answer and its status — is
    compared on replay; an edit is a mismatch of kind `answer` (docs/trace_spec.md §5)."""
    s = system()
    res = s.ask({"amount": 5, "limit": 100})
    d = json.loads(res.to_json())
    assert d["results"]["pay"][field] != value
    d["results"]["pay"][field] = value
    forged = Response.from_json(json.dumps(d))
    rep = forged.trace.replay(s, forged.flow)
    assert not rep["ok"] and rep["answers"] == "differ" and rep["kinds"] == {"answer": 1}
    assert field in rep["mismatches"][0][2] and rep["summary"].startswith("data damaged")


def test_hole4_a_confidence_within_float_rounding_still_replays():
    """Confidence is compared within runtime.CONFIDENCE_TOL (1e-9, relative): float rounding is not an edit."""
    s = system()
    res = s.ask({"amount": 5, "limit": 100})
    d = json.loads(res.to_json())
    d["results"]["pay"]["confidence"] *= 1 + 1e-12
    again = Response.from_json(json.dumps(d))
    assert again.trace.replay(s, again.flow)["ok"]


@pytest.mark.parametrize("seed", SEEDS)
def test_hole4_a_why_edited_in_the_store_with_every_hash_recomputed_does_not_replay(tmp_path, seed):
    rng = rng_for(seed)
    s = system()
    st = JSONLStorage(tmp_path / "s.jsonl", system=s, record=rng.choice(["full", "compact"]))
    for _ in range(3):
        st.save(s.ask(random_input(rng)))
    recs = read_lines(st.path)
    i = rng.randrange(len(recs))
    q = rng.choice(sorted(recs[i]["answers"]))
    (recs[i].get("response") or recs[i].get("compact"))["results"][q]["why"] = "approved by the manager"
    write_lines(st.path, recs, rechain(recs))
    st = JSONLStorage(st.path, system=s)
    bad = st.replay_all(s)
    assert [b["seq"] for b in bad] == [i] and bad[0]["kinds"] == {"answer": 1}


def _edit_index(rec, field):
    """Edit one index field of a stored record the way a forger who wants query() to find something would."""
    if field == "safeguards":
        rec["safeguards"] = rec["safeguards"] + [["hard_check", "answer:pay", ["pay"]]]
    elif field == "guards":
        rec["guards"] = {**rec["guards"], "size": "low_confidence"}
    elif field == "models":
        rec["models"] = rec["models"] + [{"type": "Forged", "id": "gpt", "fp": "0" * 16}]
    elif field == "flow":
        rec["flow"] = list(reversed(rec["flow"]))
    elif field == "confidence":
        q = sorted(rec["answers"])[0]
        rec["answers"][q][1] = 0.5 if rec["answers"][q][1] != 0.5 else 0.25


@pytest.mark.parametrize("record", ["full", "compact"])
@pytest.mark.parametrize("field", ["safeguards", "guards", "models", "flow", "confidence"])
def test_hole6_a_records_index_fields_agree_with_its_response(tmp_path, field, record):
    """The record's summary — guards, safeguards, models, flow and the confidence in answers, what query() and the SQL
    index tables read — is compared with its stored response by verify() (docs/trace_spec.md §4.3): a consistent rewrite
    of an index field alone is a problem. A compact record keeps no flow: there the edited flow is caught by replay."""
    s = system()
    st = JSONLStorage(tmp_path / "s.jsonl", system=s, record=record)
    st.save(s.ask({"amount": 5, "limit": 100}))
    assert st.verify()["ok"]
    recs = read_lines(st.path)
    _edit_index(recs[0], field)
    write_lines(st.path, recs, rechain(recs))
    st = JSONLStorage(st.path, system=s)
    out = st.verify()
    if record == "compact" and field == "flow":
        assert out["ok"]
        (bad,) = st.replay_all(s)
        assert bad["kinds"] == {"integrity": 1} and bad["summary"].startswith("data damaged")
    else:
        assert not out["ok"] and any(field.rstrip("s") in p[2] for p in out["problems"]), out["problems"]


# --- hole 5: what is left of a redacted record stays verified
def _redacted(seed, keep_answers):
    rng = rng_for(seed)
    s = system()
    st = MemoryStorage(s)
    for _ in range(4):
        st.save(s.ask(random_input(rng)))
    anchor, sig = st.head(), st.signature()
    i = rng.randrange(4)
    st.redact(st.rows[i]["id"], by="dpo", note="request", keep_answers=keep_answers)
    return st, i, anchor, sig, rng


@pytest.mark.parametrize("keep_answers", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
def test_hole5_a_legitimate_redaction_keeps_anchor_and_signature_valid(seed, keep_answers):
    st, i, anchor, sig, _ = _redacted(seed, keep_answers)
    out = st.verify(anchor=anchor, signature=sig)
    assert out["ok"] and "unverified" not in out
    assert ("answers" in st.rows[i]) is keep_answers


@pytest.mark.parametrize("seed", SEEDS)
def test_hole5_a_kept_answer_edited_after_redaction_fails_verify(seed):
    st, i, anchor, sig, rng = _redacted(seed, True)
    rec = st.rows[i]
    q = rng.choice(sorted(rec["answers"]))
    rec["answers"][q][0] = _other(q, rec["answers"][q][0], rng)
    assert not st.verify()["ok"]
    assert not st.verify(anchor=anchor, signature=sig)["ok"]
    rec["hash"] = record_hash(rec)                        # and with its hash recomputed: the chain breaks after it
    assert not st.verify()["ok"]


@pytest.mark.parametrize("seed", SEEDS)
def test_hole5_a_record_passed_off_as_redacted_with_other_answers_is_caught(seed):
    """The forgery the audit built: drop a decision's content, write other answers, add a mark with the real content
    digest and a redaction record, recompute every hash and the head — the anchor and the signature taken before
    catch it, and the signature names the record."""
    rng = rng_for(seed)
    s = system()
    st = MemoryStorage(s)
    for _ in range(4):
        st.save(s.ask(random_input(rng)))
    anchor, sig = st.head(), st.signature()
    i = rng.randrange(4)
    rec = st.rows[i]
    body = record_body(rec)
    q = rng.choice(sorted(rec["answers"]))
    forged = {k: v for k, v in rec.items() if k not in ("response", "meta")}
    forged["answers"] = copy.deepcopy(rec["answers"])
    forged["answers"][q][0] = _other(q, rec["answers"][q][0], rng)
    forged["redacted"] = {"by": "dpo", "note": None, "content": body["#content"]}
    rows = st.rows[:i] + [forged] + st.rows[i + 1:]
    rows.append({"v": 2, "kind": "redaction", "of": "", "of_hash": "", "by": "dpo", "note": None,
                 "seq": len(rows), "time": rows[-1]["time"]})
    rechain(rows)                                          # the forged record's final hash, then the record naming it
    rows[-1]["of"], rows[-1]["of_hash"] = rows[i]["id"], rows[i]["hash"]
    st.rows, st.stored_head = rows, rechain(rows)
    assert st.verify()["ok"]                               # a consistent rewrite: the chain alone cannot tell
    assert not st.verify(anchor=anchor)["ok"]
    out = st.verify(signature=sig)
    assert not out["ok"] and out["signature"]["index"] == i


@pytest.mark.parametrize("seed", range(4))
def test_hole5_on_a_real_backend_too(tmp_path, seed):
    s = system()
    st = JSONLStorage(tmp_path / "s.jsonl", system=s)
    fill(st, s, rng_for(seed), n=3)
    recs = read_lines(st.path)
    for r in recs:
        if r.get("redacted") and "answers" in r:
            q = sorted(r["answers"])[0]
            r["answers"][q][0] = _other(q, r["answers"][q][0], rng_for(seed))
            write_lines(st.path, recs)
            assert not JSONLStorage(st.path, system=s).verify()["ok"]
            return
    assert any(r.get("redacted") for r in recs)            # the answers were erased with it: nothing to edit
