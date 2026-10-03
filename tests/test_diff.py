"""Catalog fingerprints in every trace, solvi diff (stored decisions re-run with a new catalog or model) and shadow mode."""
import importlib.util
import json

from solvi import Answer, Catalog, Decision, JSONLStorage, Question, SQLiteStorage, System
from solvi.cli import main
from solvi.diff import Shadow, compare, diff
from solvi.core.provenance import catalog_fingerprint, code_fingerprint


class Scorer:
    model_id = "scorer"

    def __init__(self, version="1"):
        self.version = version


def build(threshold=100, version="1", min_confidence=None):
    cat = Catalog()
    scorer = Scorer(version)

    @cat.fn
    def words(text):
        return set(text.lower().split())

    @cat.check(hard=True, then={"approve": False})
    def known_customer(customer):
        return customer != "blocked"

    @cat.fn(model=scorer, options=["low", "high"])
    def risk(words):
        high = "urgent" in words or (scorer.version == "2" and "please" in words)      # version 2 flags "please"
        return Decision("high", {"high": 0.9, "low": 0.1}) if high else Decision("low", {"high": 0.2, "low": 0.8})

    @cat.rule("approve")
    def approve(amount, risk) -> bool:
        return amount < threshold and risk == "low"

    @cat.rule("urgent")
    def urgent(words) -> bool:
        return "urgent" in words

    return System(cat, [Question("approve", "Approve?", Answer.yes_no(), requires=["known_customer"],
                                 min_confidence=min_confidence),
                        Question("urgent", "Urgent?", Answer.yes_no())])


STATES = [{"text": "please pay", "amount": 50, "customer": "ann"},
          {"text": "urgent pay now", "amount": 50, "customer": "bob"},
          {"text": "pay", "amount": 500, "customer": "ann"},
          {"text": "pay", "amount": 10, "customer": "blocked"}]


def stored(tmp_path, kind="jsonl"):
    store = JSONLStorage(tmp_path / "d.jsonl") if kind == "jsonl" else SQLiteStorage(tmp_path / "d.db")
    s = build()
    s.storage = store
    store.catalog = s
    return store, [s.ask(st) for st in STATES]


# --- fingerprints
def test_catalog_fingerprint_follows_the_code():
    a, b = catalog_fingerprint(build().catalog), catalog_fingerprint(build().catalog)
    assert a == b                                            # the same code: the same fingerprint
    c = catalog_fingerprint(build(threshold=40).catalog)
    assert c["fp"] != a["fp"]
    assert [n for n in a["parts"] if a["parts"][n] != c["parts"][n]] == ["answer:approve"]   # the rule's closure changed
    assert catalog_fingerprint(build(version="2").catalog) == a   # a model's weights are its own fingerprint, not the code


def _module(tmp_path, name, src):
    p = tmp_path / f"{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_code_fingerprint_ignores_formatting_and_sees_constants(tmp_path):
    a = _module(tmp_path, "fa", "LIMIT = 30\n\ndef late(days):\n    return days > LIMIT\n")
    b = _module(tmp_path, "fb", 'LIMIT = 30\n\n\ndef late(days):   # a comment\n    """Docs."""\n    return (days >\n'
                                "            LIMIT)\n")
    c = _module(tmp_path, "fc", "LIMIT = 31\n\ndef late(days):\n    return days > LIMIT\n")
    d = _module(tmp_path, "fd", "LIMIT = 30\n\ndef late(days):\n    return days >= LIMIT\n")
    e = _module(tmp_path, "fe", "LIMIT = 30\n\ndef helper(x):\n    return x + 1\n\ndef late(days):\n"
                                "    return helper(days) > LIMIT\n")
    f = _module(tmp_path, "ff", "LIMIT = 30\n\ndef helper(x):\n    return x + 2\n\ndef late(days):\n"
                                "    return helper(days) > LIMIT\n")
    assert code_fingerprint(a.late) == code_fingerprint(b.late)
    assert code_fingerprint(a.late) != code_fingerprint(c.late)          # a module constant it reads
    assert code_fingerprint(a.late) != code_fingerprint(d.late)
    assert code_fingerprint(e.late) != code_fingerprint(f.late)          # a helper of the same module it calls


def test_the_code_text_behind_a_fingerprint_is_the_same_on_every_python(tmp_path):
    """ast.dump changed with the interpreter (3.12 added `type_params=[]`, 3.13 drops empty fields), so a catalog's
    fingerprint — and a stored dispatch's configuration — differed between Pythons: the text is now pinned."""
    from solvi.core.provenance import _source_ast
    m = _module(tmp_path, "fg", "def late(days: int, limit=None) -> bool:\n    return [d for d in days if d != limit]\n")
    assert _source_ast(m.late) == (
        "FunctionDef(name='late', args=arguments(args=[arg(arg='days', annotation=Name(id='int', ctx=Load())), "
        "arg(arg='limit')], defaults=[Constant()]), body=[Return(value=ListComp(elt=Name(id='d', ctx=Load()), "
        "generators=[comprehension(target=Name(id='d', ctx=Store()), iter=Name(id='days', ctx=Load()), "
        "ifs=[Compare(left=Name(id='d', ctx=Load()), ops=[NotEq()], comparators=[Name(id='limit', ctx=Load())])], "
        "is_async=0)]))], returns=Name(id='bool', ctx=Load()))")


def test_declared_types_are_in_the_fingerprint():
    def cat_with(t):
        cat = Catalog()
        ns = {}
        exec("def days(n: T) -> int:\n    return int(n)", {"T": t}, ns)   # noqa: S102
        cat.fn(ns["days"])
        return catalog_fingerprint(cat)["parts"]["days"]
    assert cat_with(int) == cat_with(int) and cat_with(int) != cat_with(float)


def test_every_trace_records_its_catalog(tmp_path):
    s = build()
    r = s.ask(STATES[0])
    fp = r.trace.fingerprint
    assert fp["catalog"] == s.fingerprint()["catalog"] and fp["questions"] == s.fingerprint()["questions"]
    assert set(fp["parts"]) == {x.part.name for x in r.flow.steps}
    assert s.fingerprint()["models"]["risk"] == next(x.model["fp"] for x in r.trace.records if x.name == "risk")
    back = type(r).from_json(r.to_json(), catalog=s)
    assert back.trace.fingerprint == fp
    assert r.trace.replay(s)["catalog"] == "same"
    rep = s.ask(STATES[2]).trace.replay(build(threshold=40))   # 500 is refused under either threshold
    assert rep["catalog"] == "changed" and rep["changed_parts"] == ["answer:approve"]
    assert rep["ok"]                                         # a changed part that re-computes the same value is no mismatch
    store, resps = stored(tmp_path)
    assert [x.id for x in store.query(catalog_fp=fp["catalog"])] == [x.stored_id for x in resps]
    assert store.query(catalog_fp="other") == []


# --- solvi diff
def test_diff_after_a_rule_change(tmp_path):
    store, resps = stored(tmp_path)
    assert diff(store, build()).ok                          # the same system: nothing changes
    rep = diff(store, build(threshold=40))
    assert rep.total == 4 and not rep.errors
    assert [c["id"] for c in rep.changed] == [resps[0].stored_id]
    ch = rep.changed[0]["questions"]["approve"]
    assert ch["old"]["answer"] == "yes" and ch["new"]["answer"] == "no" and ch["changed"] == ["answer"]
    fs = ch["first_step"]
    assert fs["name"] == "answer:approve" and fs["old"] == "True" and fs["new"] == "False"
    assert "code or declarations changed" in fs["why"]
    assert rep.catalog["changed_parts"] == ["answer:approve"]
    assert rep.by_question()["approve"]["transitions"] == {"'yes' → 'no'": 1}
    assert "approve: 'yes' → 'no'" in str(rep)
    assert len(store) == 4                                  # diff writes nothing


def test_diff_after_a_model_change(tmp_path):
    store, resps = stored(tmp_path, "sqlite")
    rep = diff(store, build(version="2"))
    assert [c["id"] for c in rep.changed] == [resps[0].stored_id]
    fs = rep.changed[0]["questions"]["approve"]["first_step"]
    assert fs["name"] == "risk" and fs["old"] == "'low'" and fs["new"] == "'high'"
    assert "model changed" in fs["why"] and "code" not in fs["why"]


def build_fallback(version="1", limit=100, storage=None):
    """A fact with a model-backed producer whose proposal a validator turns down above `limit`, and a rule after it."""
    cat = Catalog()
    scorer = Scorer(version)

    @cat.fn(provides="pick", model=scorer, options=["a", "b"], validate=lambda v, amount: amount < limit)
    def pick_by_model(amount):
        return Decision("a", {"a": 0.7, "b": 0.3}) if scorer.version == "1" else Decision("b", {"a": 0.4, "b": 0.6})

    @cat.fn(provides="pick")
    def pick_by_rule(amount):
        return "b"

    @cat.rule("choice")
    def choice(pick) -> str:
        return pick

    return System(cat, [Question("choice", "Which?", Answer.choice(["a", "b"]))], storage=storage)


def test_a_rejected_model_is_in_the_trace_the_index_and_the_diff(tmp_path):
    """The decision came from the rule after the model's proposal was turned down: the trace still names the model that
    ran, the store finds the decision by it, replay checks it, and diff says what happened to it."""
    store = SQLiteStorage(tmp_path / "d.db")
    s = build_fallback(storage=store)
    used, fell = s.ask({"amount": 50}), s.ask({"amount": 500})
    ru = next(r for r in used.trace.records if r.name == "pick")
    rf = next(r for r in fell.trace.records if r.name == "pick")
    assert ru.producer == "pick_by_model" and ru.model["id"] == "scorer" and ru.tried_models is None
    assert "tried_models" not in ru.body() and "tried_models" not in used.trace.to_json()      # as before when it answers
    assert "tried_models" in fell.trace.to_json()
    assert rf.producer == "pick_by_rule" and rf.model is None
    assert rf.tried_models == {"pick_by_model": {"type": "Scorer", "id": "scorer", "fp": ru.model["fp"],
                                                 "probs": {"a": 0.7, "b": 0.3}}}
    assert len(store.query(model="scorer")) == 2                    # the fallback decision too (it was 1 of 2)
    back = store.get(store.query(model="scorer")[-1].id)
    rb = next(r for r in back.trace.records if r.name == "pick")
    assert rb.tried_models == rf.tried_models and back.trace.replay(s)["ok"] and store.replay_all(s) == []
    v2 = build_fallback(version="2")                                # the rejected model changed: a model_changed mismatch
    rep = fell.trace.replay(v2)
    assert rep["kinds"] == {"model_changed": 1} and "pick_by_model (rejected)" in rep["mismatches"][0][2]
    assert rep["models"] == [(rf.step, "pick (pick_by_model)", "changed")]
    assert fell.trace.replay(v2, trust_models=True)["kinds"] == {"model_changed": 1}
    rf.tried_models["pick_by_model"]["fp"] = "0" * 16                # edited after the run: the record's hash tells
    assert fell.trace.replay(s)["mismatches"][0].kind == "integrity"
    rf.tried_models["pick_by_model"]["fp"] = ru.model["fp"]
    whys = [c["questions"]["choice"]["first_step"]["why"] for c in diff(store, v2).changed]
    assert whys and all(f"its model changed (#{ru.model['fp']} → #" in w and "#—" not in w for w in whys)
    ch = diff(store, build_fallback(limit=1000)).changed            # same model, now accepted where it was rejected
    assert len(ch) == 1
    assert f"its model (#{ru.model['fp']}) was rejected and is now used" in ch[0]["questions"]["choice"]["first_step"]["why"]


def test_diff_after_a_question_change(tmp_path):
    store, resps = stored(tmp_path)
    rep = diff(store, build(min_confidence=0.9))            # approve answers at 0.8 now abstain
    ids = [c["id"] for c in rep.changed]
    assert ids == [resps[0].stored_id, resps[2].stored_id]          # the urgent one was at 0.9
    ch = rep.changed[0]["questions"]["approve"]
    assert ch["new"]["status"] == "abstain" and ch["new"]["guard"] == "low_confidence"
    assert "questions changed" in ch["first_step"]["why"]
    only = diff(store, build(min_confidence=0.9), question="approve", answer="yes")    # query filters pick decisions
    assert only.total == 1


def test_compare_confidence_tolerance():
    a, b = build().ask(STATES[0]), build().ask(STATES[0])
    assert compare(a, b) == {}
    b.results["approve"].confidence = 0.5
    assert compare(a, b)["approve"]["changed"] == ["confidence"]
    assert compare(a, b, confidence=None) == {}


SYSTEM_FILE = """
import sys
sys.path.insert(0, {tests!r})
from test_diff import build
system = build(threshold={threshold})
"""


def test_cli(tmp_path, capsys):
    import os
    tests = os.path.dirname(__file__)
    (tmp_path / "v1.py").write_text(SYSTEM_FILE.format(tests=tests, threshold=100))
    (tmp_path / "v2.py").write_text(SYSTEM_FILE.format(tests=tests, threshold=40))
    store, resps = stored(tmp_path, "sqlite")
    path = str(tmp_path / "d.db")
    assert main(["verify", path]) == 0
    assert "chain verified" in capsys.readouterr().out
    assert main(["replay", path, "--system", str(tmp_path / "v1.py") + ":system"]) == 0
    assert "every stored trace replays" in capsys.readouterr().out
    assert main(["diff", path, "--system", str(tmp_path / "v1.py") + ":system"]) == 0
    capsys.readouterr()
    assert main(["diff", path, "--system", str(tmp_path / "v2.py") + ":system"]) == 1
    out = capsys.readouterr().out
    assert "1 changed" in out and "first difference: answer:approve" in out
    assert main(["diff", path, "--system", str(tmp_path / "v2.py") + ":system", "--json"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert d["changed"][0]["id"] == resps[0].stored_id
    assert main(["replay", path, "--system", str(tmp_path / "v2.py") + ":system"]) == 1
    head = store.head()
    assert main(["verify", path, "--anchor", f"{head['count'] + 1}:{head['hash']}"]) == 1


# --- shadow mode
def test_shadow_answers_with_the_current_system(tmp_path):
    current = build()
    current.storage = JSONLStorage(tmp_path / "current.jsonl")
    current.storage.catalog = current
    shadow = Shadow(current, build(threshold=40), storage=SQLiteStorage(tmp_path / "shadow.db"))
    plain = build()
    for st in STATES:
        r = shadow.ask(st)
        assert {q: x.answer for q, x in r.results.items()} == {q: x.answer for q, x in plain.ask(st).results.items()}
    assert shadow.stats == {"asks": 4, "agree": 3, "differ": 1, "errors": 0}
    assert len(current.storage) == 4 and len(shadow.storage) == 4
    first = next(iter(current.storage)).id
    rec = shadow.storage.query(question="approve", answer="no")[0]
    assert rec.meta["shadow_of"] == first and rec.meta["diff"]["approve"]["new"]["answer"] == "no"
    assert shadow.changed[0]["stored_id"] == first and shadow.changed[0]["shadow_id"] == rec.id
    assert "approve: 'yes' → 'no'  ×1" in shadow.summary()
    assert shadow.storage.verify()["ok"] and current.storage.verify()["ok"]


def test_shadow_survives_a_failing_candidate():
    class Broken:
        def ask(self, *a, **kw):
            raise RuntimeError("candidate down")
    shadow = Shadow(build(), Broken())
    r = shadow.ask(STATES[0])
    assert r["approve"].answer == "yes"
    assert shadow.stats["errors"] == 1 and "candidate down" in shadow.errors[0]["error"]


# --- which step changed the answer
def scorecard(version=1, refuse_at=5):
    """v2 adds a rule (guarantor_points: 0 for most applicants) and lowers the threshold."""
    cat = Catalog()

    @cat.fn
    def duration_points(months):
        return 3 if months > 24 else 1

    @cat.fn
    def amount_points(amount):
        return 2 if amount > 5000 else 0

    if version == 1:
        @cat.fn
        def points(duration_points, amount_points):
            return duration_points + amount_points
    else:
        @cat.fn
        def guarantor_points(guarantor):
            return 0 if guarantor else 2

        @cat.fn
        def points(duration_points, amount_points, guarantor_points):
            return duration_points + amount_points + guarantor_points

    @cat.rule("decision")
    def decision(points):
        return "refuse" if points >= refuse_at else "approve"
    return System(cat, [Question("decision", "Decide", Answer.choice(["approve", "refuse"]))])


def test_diff_names_the_step_that_changed_the_answer_not_an_added_step_that_changed_nothing(tmp_path):
    """Every change used to be blamed on the first differing step in flow order, and an added step always differs: a new
    rule that scored 0 was named for decisions that changed because of the threshold alone."""
    store = JSONLStorage(tmp_path / "d.jsonl")
    v1 = scorecard(1)
    v1.storage, store.catalog = store, v1
    apps = [{"months": 12, "amount": 9000, "guarantor": True},      # 3 points under both versions
            {"months": 36, "amount": 100, "guarantor": True},       # 3 points under both versions
            {"months": 12, "amount": 9000, "guarantor": False},     # 3 points, 5 with the new rule
            {"months": 36, "amount": 9000, "guarantor": True}]      # 5 points: refused under both
    for a in apps:
        v1.ask(a)
    rep = diff(store, scorecard(2, refuse_at=3))
    by = {c["seq"]: c["questions"]["decision"] for c in rep.changed}
    assert sorted(by) == [0, 1, 2]
    for seq in (0, 1):                                              # the total is the same: only the rule's threshold
        assert [c["name"] for c in by[seq]["causes"]] == ["answer:decision"]
        assert by[seq]["first_step"]["why"] == "its code or declarations changed"
    assert [c["name"] for c in by[2]["causes"]] == ["guarantor_points", "points", "answer:decision"]
    assert by[2]["first_step"]["name"] == "guarantor_points" and "new step" in by[2]["first_step"]["why"]
    text = str(rep)
    assert "parts that ran now and not then: guarantor_points" in text and "and: points: 3 → 5" in text
    only_rule = diff(store, scorecard(2, refuse_at=5))             # the new rule alone: one decision changes, by it
    (c,) = only_rule.changed
    assert c["seq"] == 2 and [x["name"] for x in c["questions"]["decision"]["causes"]] == ["guarantor_points", "points"]
