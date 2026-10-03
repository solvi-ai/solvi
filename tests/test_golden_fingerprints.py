"""Golden fingerprints: the catalog, part, question and model fingerprints of a representative set of systems, as 0.9.0
computed them, must not change when solvi's code moves (the 1.0 layout moves most modules).

A stored decision records these fingerprints; a replay compares them with the system it is given. A fingerprint that
changes with a refactor makes every decision stored before it look as if another system had made it. What is pinned
(tests/fixtures/golden_fingerprints/fingerprints.json, the same on every supported Python):

- types/*: the fingerprint data of solvi's typed answers (Maybe, Span, Scale, Rank, Estimate, Bins, NotStated) and of
  its value classes, as plain lists — a diff shows which module name moved;
- objects/*: solvi objects used as catalog parts (decision parts, a cascade, a vote, a generator's part), hashed by
  their type;
- gallery/*: every gallery catalog with its questions (System.fingerprint: catalog, questions, parts, models);
- examples/*: the typed catalog, typed decisions and answer primitives examples, and the other offline builders;
- stand/*: the catalogs of the task stand that build offline (credit, NAB, NATURAL PLAN, CUAD, BIRD, Abt-Buy; τ-bench,
  Banking77 and RAGTruth build theirs from data or a live session);
- golden/*: tests/fixtures/golden_fingerprints/catalogs.py — every typed answer as a rule and as a decision, a
  callable object, a cascade, a vote, an LLM decider, a fitted head with a guarantee, a learned rule list, solvi.build.

A module that moves keeps its fingerprints through the table solvi._deprecate.MOVED (new module → 0.9 module):
test_moving_every_module_with_the_table_keeps_every_fingerprint moves every solvi class and function at once.

An intended change (a gallery task edited, a fingerprint fixed on purpose): rewrite the pinned values with
`uv run python tests/test_golden_fingerprints.py --write` and say in the commit which ones changed and why."""
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Annotated, Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "fixtures" / "golden_fingerprints"
# One set for every supported Python (3.11+). Python 3.10, supported until 1.0, recorded typing.Any (not yet a class
# there) as "typing.Any", so a catalog declaring Any had another fingerprint on 3.10; these are the 3.11+ values. Since
# 1.0 a union records "typing.Union" whatever its spelling (3.14 made X | Y and Union[X, Y] one class): the entries that
# declare `X | Y` (types/str_or_not_stated, golden/typed_rules, stand/naturalplan:calendar) differ from 0.9.0's.
PINNED = DATA / "fingerprints.json"
GALLERY = ROOT / "gallery"
EXAMPLES = ROOT / "examples"
STAND = ROOT / "benchmarks" / "tasks"


# --- loading the modules that hold the catalogs, under fixed names (a type's module name is in its fingerprint)
@contextlib.contextmanager
def _isolated(*folders):
    """sys.path and the modules imported from `folders` as they were before (the stand's tasks each have a `score`)."""
    path, before = list(sys.path), set(sys.modules)
    try:
        yield
    finally:
        sys.path[:] = path
        roots = tuple(str(f) for f in folders)
        for name in set(sys.modules) - before:
            f = getattr(sys.modules[name], "__file__", None) or ""
            if f.startswith(roots) or name.startswith(("golden_", "stand_", "task")):
                sys.modules.pop(name, None)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gallery_task(folder):
    """As the gallery's run.py loads it: executed in a fresh module named "task"."""
    mod = types.ModuleType("task")
    path = folder / "task.py"
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


def _fp(system):
    fp = system.fingerprint()
    return {k: fp[k] for k in ("catalog", "questions", "parts", "models")}


def _types():
    from solvi import Bins, Claim, Decision, Estimate, Maybe, NotStated, Quote, Rank, Scale, Span
    from solvi.provenance import type_fingerprint
    ts = {"maybe_bool": Maybe[bool], "maybe_span_float": Maybe[Span[float]], "span_float_note": Span[float, "note"],
          "scale_literal": Scale[Literal["low", "medium", "high"]], "scale_values": Scale[1, 2, 3],
          "rank_2": Rank[Literal["email", "phone", "letter"], 2], "rank_all": Rank[Literal["a", "b"]],
          "estimate": Estimate[0, 3, 7, 14], "maybe_estimate": Maybe[Estimate[0, 7]],
          "bins": Annotated[float, Bins([0, 2, 4], coverage=0.9, unit="weeks")], "str_or_not_stated": str | NotStated,
          "list_literal": list[Literal["a", "b"]], "not_stated": NotStated, "quote": Quote, "claim": Claim,
          "decision": Decision}
    return {f"types/{k}": json.loads(json.dumps(type_fingerprint(t))) for k, t in ts.items()}


def _objects(golden):
    from solvi.generate import generator
    from solvi.multi import Cascade, Vote
    from solvi.provenance import code_fingerprint
    m, m2 = golden.model(), golden.model("v2")
    parts = [m.decision("kind", "What kind?", "note", ["a", "b"]), m2.decision("kind", "What kind?", "note", ["a", "b"])]
    writer = generator("http://127.0.0.1:9/v1", "golden-model")

    def prompt(note):
        return note
    return {"objects/decision_part": code_fingerprint(parts[0]), "objects/cascade": code_fingerprint(Cascade(parts)),
            "objects/vote": code_fingerprint(Vote(parts)),
            "objects/generator_part": code_fingerprint(writer.part("text", prompt, k=2))}


def _gallery():
    out = {}
    for folder in sorted(p for p in GALLERY.iterdir() if (p / "task.py").is_file()):
        with _isolated(GALLERY):
            task = _gallery_task(folder)
            from solvi import System
            out[f"gallery/{folder.name}"] = _fp(System(task.cat, task.QUESTIONS))
    return out


def _examples():
    from solvi import System
    out = {}
    with _isolated(EXAMPLES):
        e = _module("11_answer_types_and_constraints", EXAMPLES / "11_answer_types_and_constraints.py")
        out["examples/11_answer_types_and_constraints"] = _fp(System(e.cat, e.QUESTIONS))
        e = _module("12_grounded_audit", EXAMPLES / "12_grounded_audit.py")
        out["examples/12_grounded_audit"] = _fp(System(e.build(), e.QUESTIONS))
        e = _module("13_decide_model", EXAMPLES / "13_decide_model.py")
        cat, _, qs = e.build(e.load_model())
        out["examples/13_decide_model"] = _fp(System(cat, qs))
        e = _module("14_typed_catalog", EXAMPLES / "14_typed_catalog.py")
        out["examples/14_typed_catalog"] = _fp(System(e.cat, e.QUESTIONS))
        e = _module("15_typed_decisions", EXAMPLES / "15_typed_decisions.py")
        out["examples/15_typed_decisions"] = _fp(System(*e.build(e.load_model())))
        e = _module("16_primitives", EXAMPLES / "16_primitives.py")
        out["examples/16_primitives:rules"] = _fp(System(*e.rule_catalog()))
        out["examples/16_primitives:decider"] = _fp(System(*e.model_catalog(e.load_model())))
    return out


def _stand():
    out = {}
    with _isolated(STAND):
        s = _module("stand_credit", STAND / "credit" / "solution.py")
        out["stand/credit:v1"] = _fp(s.build(1))
        out["stand/credit:v2"] = _fp(s.build(2))
        out["stand/credit:v2_young"] = _fp(s.build(2, young=True))
    with _isolated(STAND):
        s = _module("stand_nab", STAND / "nab" / "solution.py")
        out["stand/nab"] = _fp(s.build())
    with _isolated(STAND):
        s = _module("stand_naturalplan", STAND / "naturalplan" / "solution.py")
        for kind in ("calendar", "meeting", "trip"):
            out[f"stand/naturalplan:{kind}"] = _fp(getattr(s, kind)()[0])          # (system, search space, options)
    with _isolated(STAND):
        s = _module("stand_cuad", STAND / "cuad" / "solution.py")
        out["stand/cuad"] = _fp(s.system())
    with _isolated(STAND):
        s = _module("stand_bird", STAND / "bird" / "solution.py")
        writer = s.generator("http://127.0.0.1:9/bird/v1", "golden-model", api_key="proxy", max_tokens=3000, timeout=300,
                             extra_body={"reasoning": {"effort": "low"}})
        out["stand/bird"] = _fp(s.build(writer))
    with _isolated(STAND):
        sys.path.insert(0, str(STAND))
        s = _module("stand_abtbuy", STAND / "abtbuy" / "pairfacts.py")
        from solvi import System
        cat, question = s.build(s.Idf([{"name": "sony bravia kdl-40 tv"}, {"name": "sony 40 inch lcd tv black"}]))
        out["stand/abtbuy"] = _fp(System(cat, [question]))
    return out


def _golden():
    out = {}
    with _isolated(DATA):
        g = _module("golden_catalogs", DATA / "catalogs.py")
        out.update(_objects(g))
        for name, make in g.SYSTEMS.items():
            s = make()
            if hasattr(s, "dispatcher"):                     # solvi.build: System 1, and the slow path over it
                out[f"golden/{name}"] = _fp(s.system)
                out[f"golden/{name}:slow"] = s.slow.fingerprint()
            else:
                out[f"golden/{name}"] = _fp(s)
    return out


def collect():
    """Every pinned fingerprint, computed now."""
    old = os.environ.pop("SOLVI_DECIDE_MODEL", None)       # the examples' stand-in deciders, not a local checkpoint
    try:
        return {**_types(), **_golden(), **_gallery(), **_examples(), **_stand()}
    finally:
        if old is not None:
            os.environ["SOLVI_DECIDE_MODEL"] = old


def _pinned():
    return json.loads(PINNED.read_text())


@pytest.fixture(scope="module")
def pinned():
    return _pinned()


@pytest.fixture(scope="module")
def now():
    return collect()


def test_the_set_of_pinned_fingerprints_is_the_same(pinned, now):
    assert sorted(now) == sorted(pinned), "a pinned system was added or removed: rewrite with --write"


def test_every_fingerprint_is_the_pinned_one(pinned, now):
    changed = {k: {"pinned": pinned[k], "now": now[k]} for k in pinned if k in now and now[k] != pinned[k]}
    assert not changed, ("fingerprints changed — a moved module needs a line in solvi._deprecate.MOVED; an intended "
                         "change: `python tests/test_golden_fingerprints.py --write`\n"
                         + json.dumps(changed, indent=1, sort_keys=True)[:6000])


# --- the module-name table
def _solvi_objects():
    """Every class and function defined in a loaded solvi module (methods and nested classes included), with its module."""
    out, seen = [], set()

    def visit(obj, module):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        out.append((obj, module))
        if isinstance(obj, type):
            for v in list(vars(obj).values()):
                v = getattr(v, "__func__", v)                # classmethod / staticmethod
                if (isinstance(v, type) or inspect.isfunction(v)) and getattr(v, "__module__", None) == module:
                    visit(v, module)

    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "solvi" or name.startswith("solvi.")):
            continue
        for v in list(vars(mod).values()):
            if (isinstance(v, type) or inspect.isfunction(v)) and getattr(v, "__module__", None) == name:
                visit(v, name)
    return out


@contextlib.contextmanager
def _relocated(table):
    """Every solvi class and function (and each module's __name__, which functions defined later read) moved to
    "solvi.relocated.<old module>"; with table=True, MOVED maps each new module back to its old one."""
    import pkgutil
    import warnings

    import solvi
    from solvi import _deprecate, provenance
    with warnings.catch_warnings():                          # every module that imports without optional extras
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(solvi.__path__, "solvi."):
            if info.name not in sys.modules and not info.name.endswith("__main__"):
                with contextlib.suppress(Exception):
                    importlib.import_module(info.name)
    objs = _solvi_objects()
    mods = {m for _, m in objs}
    new = {m: "solvi.relocated." + m for m in mods}
    try:
        for obj, m in objs:
            obj.__module__ = new[m]
        for m in mods:
            sys.modules[m].__dict__["__name__"] = new[m]
        if table:
            _deprecate.MOVED.update({v: k for k, v in new.items()})
        provenance._CODE = None                              # cached code fingerprints were computed before the move
        yield
    finally:
        for obj, m in objs:
            obj.__module__ = m
        for m in mods:
            sys.modules[m].__dict__["__name__"] = m
        _deprecate.MOVED.clear()
        provenance._CODE = None


def test_moving_every_module_with_the_table_keeps_every_fingerprint(pinned):
    with _relocated(table=True):
        moved = collect()
    changed = sorted(k for k in pinned if moved.get(k) != pinned[k])
    assert not changed, changed


def test_moving_without_the_table_changes_the_fingerprints_that_name_a_module(pinned):
    """The test above has teeth: without the table, the fingerprints that record a solvi module change."""
    with _relocated(table=False):
        moved = collect()
    changed = {k for k in pinned if moved.get(k) != pinned[k]}
    assert {"types/maybe_bool", "types/quote", "objects/decision_part", "objects/cascade", "golden/typed_rules",
            "golden/decisions"} <= changed


OLD = '''
import enum
from pydantic import BaseModel


class Grade(str, enum.Enum):
    low = "low"
    high = "high"


class Order(BaseModel):
    amount: float
    grade: Grade


class Score:
    __name__ = "score"

    def __call__(self, order: Order) -> float:
        return order.amount
'''


def _catalog_over(mod):
    from solvi import Catalog
    from solvi.provenance import catalog_fingerprint
    cat = Catalog()
    cat.fn(mod.Score())

    @cat.rule("grade")
    def grade(order: mod.Order, score: float) -> mod.Grade:
        return mod.Grade.high if score > 10 else order.grade
    return catalog_fingerprint(cat)


@pytest.mark.parametrize("entry", ["module", "name"])
def test_a_type_moved_to_another_module_with_a_table_entry_keeps_the_fingerprint(entry):
    """A temporary module stands for the 0.9 location, another for the new one, with the same source."""
    from solvi import _deprecate, provenance
    old, new = types.ModuleType("fp_old_home"), types.ModuleType("fp_new_home")
    for m in (old, new):
        sys.modules[m.__name__] = m
        exec(OLD, m.__dict__)
    try:
        before = _catalog_over(old)
        provenance._CODE = None
        assert _catalog_over(new) != before                 # moved, no entry: another catalog as far as a replay can tell
        if entry == "module":
            _deprecate.MOVED["fp_new_home"] = "fp_old_home"
        else:
            _deprecate.MOVED.update({f"fp_new_home:{n}": "fp_old_home" for n in ("Grade", "Order", "Score")})
        provenance._CODE = None
        assert _catalog_over(new) == before
        if entry == "name":                                  # a name entry covers that name only
            _deprecate.MOVED.pop("fp_new_home:Grade")
            provenance._CODE = None
            assert _catalog_over(new) != before
    finally:
        _deprecate.MOVED.clear()
        provenance._CODE = None
        for m in (old, new):
            sys.modules.pop(m.__name__, None)


def test_the_table_maps_to_modules_of_0_9_0():
    """Every line of MOVED names a module 0.9.0 had (modules_0_9_0.json) — the name the stored records carry."""
    from solvi import _deprecate, provenance
    assert provenance._FP_MODULE is _deprecate.MOVED
    known = set(json.loads((DATA / "modules_0_9_0.json").read_text()))
    assert "solvi.typed" in known and "solvi.core" in known
    assert not {old for old in _deprecate.MOVED.values() if old not in known}
    assert all(k.count(":") <= 1 and v.count(":") == 0 for k, v in _deprecate.MOVED.items())


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: python tests/test_golden_fingerprints.py --write   (rewrites the pinned fingerprints)")
    PINNED.write_text(json.dumps(collect(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {PINNED}")
