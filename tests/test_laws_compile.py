"""solvi.experimental.laws: a law set compiled into plain code (`Atom.code`, `Law.to_code`, `LawSet.to_module`) and the
differential check (`check_compiled`) that runs the compiled module against `LawSet.predict` on every buffer row.

The door world of tests/test_laws.py, with atoms that carry their code."""
from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, Buffer, Candidate, Grammar, LawSet, Transition, check_compiled,
                                         load_module)


def door_grammar(code=True, wrong=False):
    c = (lambda x: x) if code else (lambda x: None)
    return Grammar([
        Atom("locked", lambda s: s["locked"], feature="lock", code=c("s['locked']")),
        Atom("unlocked", lambda s: not s["locked"], feature="lock", code=c("not s['locked']")),
        Atom("key", lambda s: s["key"], feature="key", code=c("s['light']" if wrong else "s['key']")),
        Atom("no_key", lambda s: not s["key"], feature="key", code=c("not s['key']")),
        Atom("light", lambda s: s["light"], feature="light", code=c("s['light']")),
        Atom("dark", lambda s, a: not s["light"] and a != "never", feature="light",
             code=c("not s['light'] and a != 'never'")),
    ], k=3)


def door(state, action):
    if action == "push" and (not state["locked"] or state["key"]):
        return "opens"
    return "same"


def states():
    return [{"locked": lo, "key": k, "light": li} for lo in (False, True) for k in (False, True) for li in (False, True)]


def buffer_of(g, groups=("w1", "w2", "w3")):
    b = Buffer(g)
    for gr in groups:
        for s in states():
            for a in ("push", "wait"):
                b.add(Transition(s, a, door(s, a), group=gr))
    return b


def door_laws(g):
    ls = LawSet(g, base=lambda s, a: "same", min_support=2)
    ls.add(Candidate("push", ("unlocked",), "opens"))
    ls.add(Candidate("push", ("key", "locked"), "opens"))
    ls.add(Candidate("push", ("dark", "key", "locked"), "opens"))      # more specific: first in the dispatch
    return ls


def test_an_atom_carries_its_code_and_a_law_compiles_to_a_pure_function():
    g = door_grammar()
    assert g.by_name["key"].code == "s['key']" and Atom("x", lambda s: 1).code is None
    ls = door_laws(g)
    src = ls.laws[0].to_code(g)
    assert src.startswith("def law_3(s, a):") and "s['key']" in src and "'push'" in src
    ns: dict = {}
    exec(src, ns)
    assert ns["law_3"]({"locked": True, "key": True, "light": False}, "push") == (True, "opens")
    assert ns["law_3"]({"locked": True, "key": True, "light": True}, "push") == (False, None)
    assert ns["law_3"]({"locked": True, "key": True, "light": False}, "wait") == (False, None)


def test_the_module_dispatches_most_specific_first_with_the_base_as_fallback():
    g = door_grammar()
    ls = door_laws(g)
    mod = load_module(ls.to_module())
    s = {"locked": True, "key": True, "light": False}
    assert mod.applied(s, "push") == ls.applied(s, "push").id == "law-3"
    assert mod.applied({"locked": False, "key": False, "light": True}, "push") == "law-1"
    assert mod.applied(s, "wait") is None
    assert mod.predict(s, "wait") is None                                # no base given: None, like LawSet
    assert mod.predict(s, "wait", base=lambda s, a: "same") == "same"
    assert mod.predict(s, "push", apply=lambda e, s: e.upper()) == "OPENS"
    assert [lid for lid, _ in mod.LAWS] == [law.id for law in ls.laws]


def test_the_differential_check_finds_zero_differences_on_every_row():
    g = door_grammar()
    ls = door_laws(g)
    b = buffer_of(g)
    chk = check_compiled(ls.to_module(), b, ls)
    assert chk.rows == len(b) == 48 and chk.differences == 0 and chk.ok
    chk2 = check_compiled(load_module(ls.to_module()), b, ls)          # a module object works too
    assert chk2.ok


def test_the_differential_check_catches_code_that_does_not_equal_its_atom():
    g = door_grammar(wrong=True)                                         # "key" compiled as s['light']
    ls = door_laws(g)
    chk = check_compiled(ls.to_module(), buffer_of(g), ls)
    assert chk.differences > 0 and not chk.ok
    row = chk.examples[0]
    assert set(row) >= {"id", "action", "compiled", "interpreted"}


def test_an_atom_without_code_is_refused_by_name():
    g = door_grammar()
    g2 = Grammar([a if a.name != "key" else Atom("key", a.fn, feature="key") for a in g.atoms], k=3)
    ls = door_laws(g2)
    with pytest.raises(ValueError, match="key"):
        ls.to_module()
    with pytest.raises(ValueError, match="without code"):
        ls.laws[0].to_code(g2)
    ls.laws[-1].to_code(g2)                                              # ("unlocked",) needs no "key": fine


def test_an_effect_or_action_that_is_not_a_literal_is_refused():
    g = door_grammar()
    ls = LawSet(g)
    ls.add(Candidate("push", ("unlocked",), object()))
    with pytest.raises(ValueError, match="literal"):
        ls.to_module()


def test_defs_are_written_once_and_the_module_imports_nothing_from_solvi(tmp_path):
    helper = "def _even(x):\n    return x % 2 == 0\n"
    g = Grammar([Atom("even", lambda s: s["n"] % 2 == 0, feature="n", code="_even(s['n'])", defs=helper),
                 Atom("even2", lambda s: s["m"] % 2 == 0, feature="m", code="_even(s['m'])", defs=(helper,))], k=2)
    ls = LawSet(g, base=lambda s, a: 0)
    ls.add(Candidate("go", ("even",), 1))
    ls.add(Candidate("go", ("even", "even2"), 2))
    src = ls.to_module()
    assert src.count("def _even(") == 1
    p = tmp_path / "compiled_laws.py"
    p.write_text(src)
    code = ("import sys; sys.modules['solvi'] = None; sys.path.insert(0, sys.argv[1]); import compiled_laws as m; "
            "print(m.predict({'n': 2, 'm': 4}, 'go'), m.predict({'n': 2, 'm': 3}, 'go'), "
            "m.predict({'n': 1, 'm': 3}, 'go', base=lambda s, a: 0))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["2", "1", "0"]


def test_a_grammar_grows_and_the_buffer_indexes_a_late_atom_on_its_old_rows():
    g = door_grammar()
    b = buffer_of(g)
    g.add(Atom("locked_dark", lambda s: s["locked"] and not s["light"], feature="ld"))
    with pytest.raises(ValueError):
        g.add(Atom("locked_dark", lambda s: True))
    assert "locked_dark" in g.by_name and g.feature["locked_dark"] == "ld"
    m = b.match("push", ("locked_dark",))
    assert m.bit_count() == 3 * 2                                         # 2 of 8 states per world, 3 worlds
    b.add(Transition({"locked": True, "key": False, "light": False}, "push", "same", group="w4"))
    assert b.match("push", ("locked_dark",)).bit_count() == 7
