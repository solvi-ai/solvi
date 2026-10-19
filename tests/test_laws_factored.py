"""solvi.experimental.laws.factored: several law sets predicting one step together — channels, the joint error budget,
row provenance (voiding), dedup, version, report, admission by probes, export; the push-world known answer and the
joint false-accept simulation (benchmarks/laws_factored_toy.py)."""
from __future__ import annotations

import importlib.util
import pickle
import random
import sys
import warnings
from pathlib import Path

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Candidate, Grammar, Unknown, check_compiled
    from solvi.experimental.laws.factored import Channel, FactoredModel, FactoredStep, Row

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    spec = importlib.util.spec_from_file_location("laws_factored_toy", ROOT / "benchmarks" / "laws_factored_toy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["laws_factored_toy"] = mod             # its atoms pickle by module name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


T = _toy()


class In:
    def __init__(self, name):
        self.name = name

    def __call__(self, s):
        return self.name in s


def g(*names, k=2, min_k=0):
    return Grammar([Atom(n, In(n), code=f"{n!r} in s") for n in names], k=k, min_k=min_k)


def base_no(s, a):
    return "no"


def two_channels(**kw):
    return FactoredModel([Channel("A", g("x", "y"), base=base_no), Channel("B", g("u", "v"), base=base_no)], **kw)


# ───────────────────────────────────────────── the joint budget
def test_delta_is_split_equally_and_the_bound_is_the_sum():
    m = two_channels(delta=0.1)
    assert m.deltas == {"A": 0.05, "B": 0.05} and m.laws["A"].delta == 0.05 and m.laws["A"].sized
    assert m.report()["bound"] == pytest.approx(0.1)
    m2 = FactoredModel([Channel("A", g("x"), base=base_no), Channel("B", g("u"), base=base_no)],
                       share={"A": 0.08, "B": 0.02})
    assert m2.laws["A"].delta == 0.08 and m2.report()["bound"] == pytest.approx(0.1)
    with pytest.warns(UserWarning, match="sum to"):
        FactoredModel([Channel("A", g("x")), Channel("B", g("u"))], share={"A": 0.4, "B": 0.3})
    with pytest.raises(ValueError):
        Channel("A", g("x"), delta=0.1)
    with pytest.raises(ValueError):
        Channel("A", g("x"), sized=False)
    with pytest.raises(ValueError):
        FactoredModel([Channel("A", g("x")), Channel("A", g("u"))])
    with pytest.raises(ValueError):
        FactoredModel([Channel("A", g("x"))], share={"B": 0.1})


# ───────────────────────────────────────────── dedup
def test_dedup_logs_one_row_per_key_per_group_and_counts_across_groups():
    rows = [Row("A", frozenset({"x"}), "yes"), Row("A", frozenset({"y"}), "no")]
    m = two_channels()
    m.new_group("w1")
    for _ in range(3):                                   # the same world replayed three times
        m.log(rows)
    assert len(m.buffers["A"]) == 2 and m.report()["A"]["dropped"] == 4
    m2 = two_channels()
    for w in ("w1", "w2", "w3"):
        st = m2.log(rows, group=w)
        assert isinstance(st, FactoredStep) and not st.dropped
    assert len(m2.buffers["A"]) == 6
    m3 = two_channels(dedup=False)
    m3.new_group("w1")
    for _ in range(3):
        m3.log(rows)
    assert len(m3.buffers["A"]) == 6


def test_log_needs_a_group():
    with pytest.raises(ValueError, match="new_group"):
        two_channels().log([Row("A", frozenset(), "no")])


# ───────────────────────────────────────────── provenance
def learn_B_on_rows_depending_on(m, law, groups=15):
    """B: u -> yes, every row naming `law` of A."""
    for w in range(groups):
        m.log([Row("B", frozenset({"u"}), "yes", depends=(law,)), Row("B", frozenset({"v"}), "no", depends=(law,)),
               Row("B", frozenset(), "no", depends=(law,)),
               Row("B", frozenset({"u", "v"}), "yes", depends=(("A", law.id),))], group=w)


def test_retiring_a_law_voids_the_rows_that_depend_on_it_and_flags_the_law_learned_on_them():
    m = two_channels()
    a = m.laws["A"].add(Candidate("A", ("x",), "yes"))
    learn_B_on_rows_depending_on(m, a)
    lb = m.laws["B"].laws
    assert [x.condition for x in lb] == [("u",)]
    buf = m.buffers["B"]
    n, N, allowed = len(buf), buf.tested, buf.allowed("B").bit_count()
    assert allowed == n == 60
    v0 = m.version
    voided = m.retire("A", a, why="test")
    assert voided == 60 and m.version > v0
    assert buf.allowed("B") == 0 and buf.rows("B") == [] and len(buf) == n and buf.tested == N   # N unchanged
    rep = m.report()
    assert rep["B"]["voided"] == 60 and rep["B"]["stale"] == {lb[0].id: lb[0].support}
    # a row naming the fallen law is refused; the situation, freed from the dedup set, is logged again re-labelled
    st = m.log([Row("B", frozenset({"u"}), "yes", depends=(a,))])
    assert st.refused == {"B": 1} and m.report()["B"]["refused"] == 1
    st = m.log([Row("B", frozenset({"u"}), "yes")])
    assert len(st.logged["B"]) == 1 and buf.allowed("B").bit_count() == 1


def test_a_voided_row_leaves_the_running_precision_it_was_counted_in():
    m = two_channels()
    a = m.laws["A"].add(Candidate("A", ("x",), "yes"))
    b = m.laws["B"].add(Candidate("B", ("u",), "yes"))
    m.new_group("w")
    m.log([Row("B", frozenset({"u"}), "no", depends=(a,))])   # a wrong-looking row, counted against b
    assert b.running() == (0, 1)
    m.retire("A", a)
    assert b.running() == (0, 0)


def test_a_law_that_falls_by_its_running_precision_voids_its_dependents_automatically():
    m = two_channels()
    a = m.laws["A"].add(Candidate("A", ("x",), "yes"))
    m.log([Row("B", frozenset({"u"}), "yes", depends=(a,))], group="w0")
    st = None
    for w in range(1, 7):                                # A's law is wrong on every row: it falls after retire_min
        st = m.log([Row("A", frozenset({"x", "y" if w % 2 else "x"}), "no")], group=w)
        if a.status != "active":
            break
    assert a.status == "retired" and st.voided == 1 and m.buffers["B"].allowed("B") == 0


def test_depends_rejects_laws_of_no_channel():
    m = two_channels()
    other = two_channels()
    stray = other.laws["A"].add(Candidate("A", ("x",), "yes"))
    stray.id = "law-99"
    with pytest.raises(ValueError):
        m.log([Row("B", frozenset({"u"}), "yes", depends=(stray,))], group="w")


# ───────────────────────────────────────────── version, memo, pickle, export
def test_version_bumps_on_accept_retire_suspend_and_the_memo_follows_it():
    m = FactoredModel([Channel("A", g("x", "y"), base=base_no, on_fall="suspend"), Channel("B", g("u"))])
    s = frozenset({"x"})
    assert m.predict("A", s) == "no"
    v = m.version
    for w in range(12):                                  # x -> yes in every group: accepted by the test
        m.log([Row("A", frozenset({"x"}), "yes"), Row("A", frozenset({"y"}), "no")], group=w)
    assert m.version > v and m.predict("A", s) == "yes"
    law = m.laws["A"].laws[0]
    v = m.version
    m.laws["A"].suspend(law)
    assert m.version == v + 1 and m.predict("A", s) == "no"
    m.retire("A", law)
    assert m.version == v + 2


def test_pickle_round_trip_and_compiled_modules_per_channel():
    model = T.learn_factored(seed=1, worlds=150)
    got = pickle.loads(pickle.dumps(model))
    assert got.version == model.version and got.report()["bound"] == model.report()["bound"]
    s = frozenset({"p:M"})
    assert got.predict("self", s) == model.predict("self", s) == "move"
    mods = model.to_module()
    assert set(mods) == {"self", "enter", "gone"}
    for ch, src in mods.items():
        chk = check_compiled(src, model.buffers[ch], model.laws[ch])
        assert chk.rows == len(model.buffers[ch]) and chk.differences == 0


def test_to_knowledge_writes_each_channels_laws_with_the_channel_in_scope():
    from solvi.core.knowledge import KnowledgeStore
    model = T.learn_factored(seed=1, worlds=150)
    ks = KnowledgeStore()
    out = model.to_knowledge(ks, {"env": "push"})
    assert set(out) == {"self", "enter", "gone"} and sum(len(v) for v in out.values()) == len(model.all_laws())


# ───────────────────────────────────────────── admission by probes
def test_admit_by_probes_with_the_library_guards_counts_outside_delta_and_is_tried_once():
    m = FactoredModel([Channel("A", g("x", "y", "z", k=1), base=base_no)])
    m.new_group("w")
    row = Row("A", frozenset({"x", "y"}), "yes")
    m.log([row])

    def ask(s):
        return "yes" if "x" in s else "no"
    pool = [frozenset(c) for c in ({"x"}, {"y"}, {"z"}, {"x", "z"}, {"y", "z"}, set())]
    adm = m.admit("A", row, ask, pool)
    assert adm.status == "accepted" and adm.law.condition == ("x",)
    assert m.report()["A"]["probe_admitted"] == 1 and m.report()["probe_admitted"] == 1
    assert m.admit("A", row, ask, pool) is None          # tried, and no longer a mismatch


def test_admit_refuses_when_the_pool_cannot_tell_siblings_apart_and_unknown_is_spent():
    m = FactoredModel([Channel("A", g("x", "y", k=1, min_k=1), base=base_no)])
    m.new_group("w")
    row = Row("A", frozenset({"x", "y"}), "yes")
    m.log([row])
    adm = m.admit("A", row, lambda s: "yes", [frozenset({"x", "y"})])   # x and y always together on the pool
    assert adm.status == "undetermined" and not m.laws["A"].laws
    m2 = FactoredModel([Channel("A", g("x", "y", k=1), base=base_no)])
    m2.new_group("w")
    m2.log([row])
    adm = m2.admit("A", row, lambda s: Unknown, [frozenset({"x"}), frozenset({"y"})])
    assert adm.status == "undetermined" and adm.unknown


def test_admit_does_no_held_out_check():
    """admit judges by the probes and the current group's rows only: rows of other groups that contradict the admitted
    law do not refuse it (a held-out check on probe admissions is not shipped; see ROADMAP). Retirement by the
    held-out test happens later, on step / log."""
    m = FactoredModel([Channel("A", g("x", "y", k=1), base=base_no)])
    for w in range(4):                                   # other groups: x is "no" on 12 rows
        m.log([Row("A", frozenset({"x"}), "no", key=i) for i in range(3)], group=w)
    m.new_group("now")
    row = Row("A", frozenset({"x", "y"}), "yes")
    m.log([row])
    adm = m.admit("A", row, lambda s: "yes" if "x" in s else "no", [frozenset({"x"}), frozenset({"y"})])
    assert adm.status == "accepted" and m.laws["A"].laws


# ───────────────────────────────────────────── the push world (known answer)
WRITTEN = {"self": [("p:M",)], "enter": [("q:S",), ("q:P",)], "gone": [("p:D",), ("m:D",)]}


def reachable(ch, s):
    """States the world can produce: the mover always has M (enter: a:M; gone: one of the pair is the mover)."""
    return ch == "self" or (ch == "enter" and "a:M" in s) or (ch == "gone" and ("p:M" in s or "m:M" in s))


def test_push_world_the_written_laws_are_learned_with_no_false_law_and_exact_next_states():
    literal = 0
    for seed in range(5):
        r = T.push_world(seed=seed)
        model = T.learn_factored(seed=seed)
        for ch in WRITTEN:                               # the laws say what the written rules say on every reachable state
            wrong = [s for s in T.STATES[ch] if reachable(ch, s) and model.predict(ch, s) != T.WRITTEN[ch](s)]
            assert wrong == [], (seed, ch, wrong[:3])
        literal += all(c in {law.condition for law in model.laws[ch].laws} for ch, cs in WRITTEN.items() for c in cs)
        assert r["false_laws"] == [], (seed, r["false_laws"])
        assert r["accuracy"] == 1.0 and r["accuracy_clean"] == 1.0, (seed, r["accuracy"])
        assert r["joint_accuracy_clean"] < r["accuracy_clean"]          # measured: 0.947-0.952
        assert r["bound"] == pytest.approx(0.1)
    assert literal >= 4          # measured 4 of 5: seed 0 learns m:D as m:D & m:M and m:D & p:M (same predictions)


def test_push_world_a_planted_wrong_self_law_falls_and_its_dependent_rows_are_voided():
    rng = random.Random(3)
    model = FactoredModel(T.channels(), T.PushRows(), delta=0.1)
    wrong = model.laws["self"].add(Candidate("self", ("p:P",), "move"))   # P objects do not move unless M
    for w, props, pos, a, nxt in T.stream(rng, 150, 60):
        if model.group != ("train", w):
            model.new_group(("train", w))
        model.step((props, pos), a, (props, nxt))
    rep = model.report()
    assert wrong.status == "retired"
    assert rep["enter"]["voided"] + rep["gone"]["voided"] > 0
    assert T.false_laws(model) == []
    assert T.evaluate(model, None)["accuracy"] == 1.0


# ───────────────────────────────────────────── the joint false-accept simulation
def test_null_channels_any_false_law_across_three_channels_at_most_delta():
    out = T.null_run(trials=100, rows=2000)
    assert out["any_false_law_factored"] <= 10           # ≤ δ = 0.1 jointly
    assert out["any_false_law_fixed"] >= 50              # the control: fixed thresholds
