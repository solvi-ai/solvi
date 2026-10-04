"""solvi.testing.conformance on every built-in implementation of an extension point of solvi.core (and on small classes
of our own, as a user would write them): what the Building blocks page says you get for free is tested here, not
claimed."""
import json

import numpy as np
import pytest
from test_decide import TEAMS as DECIDE_TEAMS
from test_decide import model as decide_model
from test_decide import team_catalog, texts
from test_dispatch import _plan_system, fast, slow_llm
from test_multi import _parts
from test_multi import _system as combination_system
from test_textin import REFUND, TODAY, shop

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core import (Adapter, Decider, DefaultStrategist, Environment, Extractor, Head, Monitor, Outcome, Proposer,
                        Scorer, SlowPath, Space, Strategist, Thought)
from solvi.core.deciders.heads import FastHead
from solvi.core.deciders.combine import Cascade, Vote
from solvi.core.dispatch import AskPath, RefinePath, SearchPath
from solvi.core.guarantees.drift import DriftMonitor
from solvi.core.plan.cost import CostStrategist
from solvi.core.slow.search import Tree
from solvi.core.store import DuckDBStorage, JSONLStorage, SQLiteStorage, TraceStorage
from solvi.core.textin import CueExtractor
from solvi.testing import conformance as cf


# ------------------------------------------------------------------------------------------------ TraceStorage
@pytest.mark.parametrize("kind", ["jsonl", "sqlite", "duckdb"])
def test_every_built_in_store_conforms(tmp_path, kind):
    if kind == "duckdb":
        pytest.importorskip("duckdb")
    cls, name = {"jsonl": (JSONLStorage, "s.jsonl"), "sqlite": (SQLiteStorage, "s.db"),
                 "duckdb": (DuckDBStorage, "s.duckdb")}[kind]
    out = cf.check_storage(lambda: cls(tmp_path / name), reopen=lambda: cls(tmp_path / name))
    assert out["reopen"] and out["tamper"] and out["redact"]


class MemoryStorage(TraceStorage):
    """A store of your own: the five methods of a backend, in memory."""

    def __init__(self, system=None):
        super().__init__(system)
        self.rows = []

    def _append(self, body):
        from solvi.core.store import record_hash
        rec = dict(json.loads(json.dumps(body)), seq=len(self.rows), time=float(self.clock()), prev=self.head()["hash"])
        rec["hash"] = record_hash(rec)
        rec["id"] = rec["hash"][:16]
        self.rows.append(rec)
        return rec

    def _raw(self, snap=None):
        return iter(list(enumerate(self.rows)))

    def _find(self, id):
        return next((r for r in self.rows if r["id"] == id), None)

    def head(self):
        return {"count": len(self.rows), "hash": self.rows[-1]["hash"] if self.rows else ""}   # "": the genesis

    def _rewrite(self, rec):
        self.rows[rec["seq"]] = rec


def test_a_store_of_your_own_conforms_and_a_backend_missing_a_method_is_refused():
    assert cf.check_storage(MemoryStorage)["verify"]

    class Half(TraceStorage):
        def head(self):
            return {"count": 0, "hash": ""}
    with pytest.raises(TypeError, match="abstract"):
        Half()


def test_check_storage_fails_a_store_whose_rewrite_does_nothing():
    class Forgetful(MemoryStorage):
        def _rewrite(self, rec):                    # silently keeps the old record: an edit would go unseen
            pass
    with pytest.raises(cf.ConformanceError, match="breaks verify"):
        cf.check_storage(Forgetful)


# ------------------------------------------------------------------------------------------------ SlowPath
def test_the_built_in_slow_paths_conform():
    s2, _ = slow_llm()
    out = cf.check_slow_path(AskPath(s2), [{"email": "my parcel is late"}, {"email": "I was charged twice"}],
                             price=(1.0, 2.0), system1=fast())
    assert out["dispatch"] and out["budget"]
    judge = _plan_system()

    def propose(state, rounds):
        return "9:00" if not rounds else "10:00"
    assert isinstance(propose, Proposer)
    cf.check_slow_path(RefinePath(judge, propose=propose, into="slot", rounds=3), [{}], system1=_plan_system())
    cf.check_slow_path(SearchPath(judge, space=["9:00", "10:00", "11:00"], into="slot"), [{}], system1=_plan_system())
    tree = Tree("", lambda n: [n + h for h in ("9:00", "10:00")] if not n else [], complete=lambda n: bool(n))
    assert isinstance(tree, Space)
    cf.check_slow_path(SearchPath(judge, space=tree, into="slot"), [{}])


class Record:
    """A record of your own: the Responses of the Systems asked, in order."""

    def __init__(self, responses):
        self.responses = responses

    def to_dict(self):
        return {"responses": [r.to_dict() for r in self.responses]}


class FirstSure(SlowPath):
    """A slow path of your own: ask several Systems in turn, the first one that does not abstain answers."""
    mode = "test-first-sure"

    def __init__(self, systems):
        super().__init__(systems[0])
        self.systems = systems

    @property
    def steps(self):
        return len(self.systems)

    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        done = []
        for s in self.systems:
            if done and budget is not None and budget.over(Thought("x").cost, expected_round):
                return Thought(self.mode, None, False, "no budget", Record(done), stopped="no budget")
            res = s.ask(dict(state), [question], store=store)
            done.append(res)
            if res[question].status != "abstain":
                return Thought(self.mode, res[question].answer, True, None, Record(done))
        return Thought(self.mode, None, False, "every System abstained", Record(done))

    def fingerprint(self):
        from solvi.core.provenance import digest
        return digest("FirstSure", [s.fingerprint() for s in self.systems])

    def replay(self, thought, trust_models=False):
        bad = []
        for i, (s, r) in enumerate(zip(self.systems, thought.record.responses)):
            bad += [(f"system {i}", str(m)) for m in r.trace.replay(s, trust_models=trust_models)["mismatches"]]
        last = thought.record.responses[-1]
        q = next(iter(last.results))
        ok = last[q].status != "abstain"
        if ok != thought.accepted:
            bad.append(("accepted", "not what the record gives"))
        return {"ok": not bad, "mismatches": bad}

    @classmethod
    def restore_record(cls, d, system=None):
        from solvi import Response
        return Record([Response.model_validate(r) for r in d["responses"]])


def test_a_slow_path_of_your_own_conforms_and_its_decisions_round_trip_through_a_store():
    s2, _ = slow_llm()
    path = FirstSure([fast(min_confidence=0.99), s2])
    assert SlowPath.path_of("test-first-sure") is FirstSure
    out = cf.check_slow_path(path, [{"email": "my parcel is late"}, {"email": "I was charged twice"}],
                             price=(1.0, 2.0), system1=fast())
    assert out["dispatch"] and out["round_trip"]


def test_a_slow_path_subclass_names_its_mode_and_keeps_it_to_itself():
    class NoMode(SlowPath):
        def think(self, *a, **k):
            return None

        def fingerprint(self):
            return "x"
    with pytest.raises(TypeError, match="without a mode"):
        NoMode(fast())
    with pytest.raises(ValueError, match="mode 'ask' is solvi.core.dispatch.AskPath's"):
        type("Ask2", (SlowPath,), {"mode": "ask", "think": lambda *a, **k: None, "fingerprint": lambda self: ""})
    with pytest.raises(KeyError, match="no slow path of mode 'nothing'"):
        Thought.from_dict({"mode": "nothing", "accepted": False, "record": {}})


def test_check_slow_path_fails_a_path_whose_replay_does_not_look_at_the_record():
    class Credulous(FirstSure):
        mode = "test-credulous"

        def replay(self, thought, trust_models=False):
            return {"ok": True, "mismatches": []}
    s2, _ = slow_llm()
    with pytest.raises(cf.ConformanceError, match="acceptance is not its record's"):
        cf.check_slow_path(Credulous([s2]), [{"email": "my parcel is late"}])


# ------------------------------------------------------------------------------------------------ Strategist
def _loans():
    cat = Catalog()

    @cat.fn(provides="score", cost=5)
    def quick_score(income: float) -> float:
        return income / 1000

    @cat.fn(provides="score", cost=50)
    def full_score(income: float, debts: float) -> float:
        return (income - debts) / 1000

    @cat.check(hard=True, then={"approve": "no"})
    def not_sanctioned(country: str) -> bool:
        return country != "XX"

    @cat.rule("approve")
    def approve(score: float) -> str:
        return "yes" if score > 30 else "no"
    return cat, [Question("approve", "Approve?", Answer.choice(["yes", "no"]))]


STATES = [{"income": 50000.0, "debts": 1000.0, "country": "FR"}, {"income": 10000.0, "debts": 0.0, "country": "XX"},
          {"income": 90000.0, "country": "DE"}]


@pytest.mark.parametrize("strategist", [DefaultStrategist(), CostStrategist(), CostStrategist(producers="equivalent")],
                         ids=["default", "cost-declared", "cost-equivalent"])
def test_the_built_in_strategists_conform(strategist):
    cat, qs = _loans()
    assert cf.check_strategist(strategist, cat, qs, STATES)["hard_checks"] == 1


def test_the_default_strategist_plans_as_a_system_without_one():
    cat, qs = _loans()
    plain, explicit = System(cat, qs), System(cat, qs, strategist=DefaultStrategist())
    for st in STATES:
        a, b = plain.ask(dict(st)), explicit.ask(dict(st))
        assert a["approve"].answer == b["approve"].answer
        assert a.trace.records[-1].hash == b.trace.records[-1].hash        # the same trace, record for record
    assert plain._computable(STATES[0]) == explicit._computable(STATES[0])


def test_check_strategist_fails_a_planner_that_leaves_a_hard_check_out():
    class Careless(DefaultStrategist):
        def plan(self, catalog, questions, init_keys, heads=None):
            flow = super().plan(catalog, questions, init_keys, heads)
            flow.steps = [s for s in flow.steps if s.part.name != "not_sanctioned"]
            flow.per_question = {q: [n for n in ns if n != "not_sanctioned"] for q, ns in flow.per_question.items()}
            return flow
    cat, qs = _loans()
    with pytest.raises(ValueError, match="hard check not_sanctioned sets `then=`"):
        cf.check_strategist(Careless(), cat, qs, STATES)


# ------------------------------------------------------------------------------------------------ Head
def _rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"x": float(rng.normal()), "y": float(rng.normal())} for _ in range(n)]
    return rows, ["high" if r["x"] + 0.3 * r["y"] > 0 else "low" for r in rows]


def test_the_built_in_head_conforms():
    rows, answers = _rows()
    assert isinstance(FastHead(["high", "low"]), Head)
    assert cf.check_head(lambda opts: FastHead(opts, lam=1.0), ["high", "low"], rows, answers)["system"]


def test_the_multi_label_head_keeps_its_0_9_fingerprint_as_a_method():
    from solvi.core.provenance import digest, fingerprint
    cat = Catalog()
    s = System(cat, [Question("tags", "Tags?", Answer.multi(["a", "b"]))])
    rows, _ = _rows()
    s.fit("tags", [(r, ["a"] if r["x"] > 0 else ["b"]) for r in rows], ["x", "y"], select=False)
    h = s.heads["tags"]
    assert h.fingerprint() == digest("MultiHead", h.options, {o: fingerprint(x) for o, x in h.heads.items()})
    assert isinstance(h, Head)


# ------------------------------------------------------------------------------------------------ Decider, Scorer, Adapter
def test_the_built_in_deciders_conform():
    m = decide_model()
    cat, part, q = team_catalog(m)
    system = System(cat, [q])
    inputs = [{"email": t} for team in DECIDE_TEAMS for t in texts(team, 2)]
    assert isinstance(m.scorer, Scorer)
    assert cf.check_decider(part, inputs, system=system, states=inputs[:3])["closed_set"]
    s, l_, _, _ = _parts()
    for comb in (Cascade([s, l_]), Vote([s, l_])):
        _, csys = combination_system(comb)
        cf.check_decider(comb, inputs[:4], system=csys, states=inputs[:2])


def test_a_decider_of_your_own_conforms_through_a_catalog():
    class Keyword:
        """A decider of your own: an object with options, fingerprint() and a call that returns a Decision."""
        __name__ = "team"
        options = ["billing", "shipping"]

        def fingerprint(self):
            return "keyword-1"

        def __call__(self, email):
            p = 0.9 if "charged" in email else 0.2
            return Decision("billing" if p > 0.5 else "shipping", {"billing": p, "shipping": 1 - p})
    dec = Keyword()
    assert isinstance(dec, Decider)
    cat = Catalog()
    cat.fn(dec, model=dec, options=dec.options)

    @cat.rule("route")
    def route(team):
        return team
    system = System(cat, [Question("route", "Route", Answer.choice(["billing", "shipping"]))])
    states = [{"email": "I was charged twice"}, {"email": "where is my parcel"}]
    assert cf.check_decider(dec, states, system=system, states=states)["system"]


class FakeAdapter:
    """An adapter of your own (the Adapter protocol), not LoRA: it shifts one option's logit while active."""
    kind = "test-shift"

    def __init__(self, shift):
        self.shift, self.active = shift, False

    def fingerprint(self):
        return f"shift-{self.shift}"

    def using(self, scorer, active=True):
        import contextlib
        me = self

        @contextlib.contextmanager
        def ctx():
            me.active, scorer.adapter = active, (me if active else None)
            try:
                yield
            finally:
                me.active, scorer.adapter = False, None
        return ctx()

    def save(self, path):
        return path

    @classmethod
    def load(cls, path):
        return cls(0.0)


def test_the_adapter_slot_reads_only_the_adapter_protocol():
    from solvi.core.deciders import lora_key
    m = decide_model()
    _, part, _ = team_catalog(m)
    ad = FakeAdapter(2.0)
    assert isinstance(ad, Adapter)
    before = m.fingerprint()
    m.loras[lora_key(part.spec)] = ad
    assert part.lora is ad and m.fingerprint() != before                       # its fingerprint is in the model's
    d = part(email="I was charged twice")
    assert d.extra["lora"]["adapter"] == "shift-2.0"                           # and in the decision's record
    meta = m.metadata()["loras"][0]
    assert meta["task"] == part.spec.task and (meta["adapter"], meta["kind"]) == ("shift-2.0", "test-shift")


def test_the_lora_adapter_has_the_adapter_protocol():
    lora = pytest.importorskip("solvi.experimental.lora")             # the class only; its tensors need torch
    for name in ("kind", "fingerprint", "using", "save", "load"):
        assert hasattr(lora.LoraAdapter, name), name


# ------------------------------------------------------------------------------------------------ Extractor
def test_the_built_in_extractor_conforms():
    _, s = shop()
    assert isinstance(CueExtractor(), Extractor)
    out = cf.check_extractor(CueExtractor(), s, [REFUND, "Please refund order A-10457, I paid 20 euros on 2026-09-20."],
                             question="request_refund", today=TODAY, patterns={"order_id": r"[A-Z]-\d+"},
                             synonyms={"currency": {"EUR": ["euro", "euros"], "RUB": ["rubles"]}})
    assert out["quotes"] > 0 and out["fields_read"] >= 6


def test_check_extractor_fails_an_extractor_whose_quote_is_not_the_text():
    from solvi.core.catalog import Quote

    class Loose(CueExtractor):
        def find(self, text, fs):
            return [Quote(q.value.upper() + "!", q.start, q.end, q.source, q.confidence) for q in super().find(text, fs)]
    _, s = shop()
    with pytest.raises(cf.ConformanceError, match="the text at its offsets"):
        cf.check_extractor(Loose(), s, [REFUND], question="request_refund", today=TODAY)


# ------------------------------------------------------------------------------------------------ Monitor
def _stream(n, conf, alone=True, seed=0):
    rng = np.random.default_rng(seed)
    return [{"value": "a" if rng.random() < 0.5 else "b", "confidence": float(np.clip(conf + 0.05 * rng.normal(), 0, 1)),
             "alone": alone} for _ in range(n)]


def test_the_built_in_monitor_conforms():
    make = lambda: DriftMonitor(window=30, sequential=False)
    assert isinstance(make(), Monitor)
    out = cf.check_monitor(make, _stream(80, 0.9), changed=_stream(60, 0.5, alone=False, seed=1))
    assert out["changed"] and out["reset"]


# ------------------------------------------------------------------------------------------------ Environment
class Corridor:
    """The smallest environment: walk right to the door at 3; a wall at 0 refuses "left"."""

    def reset(self, seed=None):
        self.pos = (seed or 0) % 2
        return {"pos": self.pos}

    def actions(self, state):
        return ["left", "right"]

    def step(self, action):
        if action == "left" and self.pos == 0:
            return Outcome({"pos": 0}, accepted=False, effect=None)
        self.pos += 1 if action == "right" else -1
        return Outcome({"pos": self.pos}, effect={"pos": self.pos}, done=self.pos == 3)


def test_an_environment_conforms():
    assert isinstance(Corridor(), Environment)
    assert cf.check_environment(Corridor, policy=lambda s, acts: "right")["steps"]
    cf.check_environment(Corridor, steps=3)                             # "left" at the wall: refused, state kept


def _shop_transitions(n, seed):
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st = {"status": rng.choice(["pending", "delivered", "cancelled"]), "paid": rng.choice(["card", "gift"])}
        action = rng.choice(["cancel", "refund"])
        args = {"method": rng.choice(["card", "gift"])}
        ok = st["status"] == "pending" if action == "cancel" else args["method"] == st["paid"]
        out.append((st, action, args, ok, {"status": "cancelled"} if ok and action == "cancel" else None))
    return out


def _shop_vocabulary():
    from solvi.core.knowledge import Vocabulary
    return Vocabulary({"status": lambda s, a: s["status"], "original": lambda s, a: a["method"] == s["paid"]})


def test_the_conservative_action_model_conforms():
    from solvi.core.knowledge import ConservativeActionModel
    rep = cf.check_action_model(lambda: ConservativeActionModel(_shop_vocabulary()), _shop_transitions(200, 1),
                                held_out=_shop_transitions(200, 2))
    assert rep["transitions"] == 200 and rep["held_out"]["refusal_precision"] == 1.0
    assert rep["held_out"]["refusal_recall"] == 1.0


def test_an_action_model_that_contradicts_what_it_saw_or_is_not_deterministic_fails():
    from solvi.core.knowledge import Prediction

    class Optimist:                                   # accepts everything, whatever the environment said
        def observe(self, state, action, args, accepted, effect=None):
            self.n = getattr(self, "n", 0) + 1

        def predict(self, state, action, args):
            return Prediction("accept", 0.0, 0, "always")

        def fingerprint(self):
            return f"optimist:{getattr(self, 'n', 0)}"
    with pytest.raises(cf.ConformanceError, match="contradicts"):
        cf.check_action_model(Optimist, _shop_transitions(50, 3))

    class Moody(Optimist):                            # its answer depends on how often it was asked: not reproducible
        def predict(self, state, action, args):
            self.calls = getattr(self, "calls", 0) + 1
            return Prediction("refuse" if self.calls > 5 else "unknown")
    with pytest.raises(cf.ConformanceError, match="same predictions"):
        cf.check_action_model(Moody, [t for t in _shop_transitions(50, 4) if not t[3]])

    class NotAModel:
        def predict(self, state, action, args):
            return "accept"
    with pytest.raises(cf.ConformanceError, match="lacks observe"):
        cf.check_action_model(NotAModel, [])


def test_every_protocol_is_runtime_checkable_and_names_its_promise():
    import typing

    import solvi.core as core
    for name in core.EXTENSION_POINTS:
        obj = getattr(core, name)
        assert obj.__module__ == core.EXTENSION_POINTS[name], name
        assert obj.__doc__, name
        if getattr(obj, "_is_protocol", False) and obj is not typing.Protocol:
            assert getattr(obj, "_is_runtime_protocol", False), name
            assert "You implement" in obj.__doc__ and "Stability" in obj.__doc__, name
    for base in (TraceStorage, SlowPath):
        assert base.__abstractmethods__ and "Stability" in base.__doc__
    assert {Strategist, Head, Monitor} and isinstance(DefaultStrategist(), Strategist)


def test_the_extension_points_load_on_first_use():
    import subprocess
    import sys
    code = ("import sys, solvi.core\n"
            "lazy = ['solvi.core.dispatch', 'solvi.core.environment', 'solvi.core.guarantees.monitor', "
            "'solvi.core.deciders.protocols']\n"
            "assert not [m for m in lazy if m in sys.modules], [m for m in lazy if m in sys.modules]\n"
            "solvi.core.SlowPath\n"
            "assert 'solvi.core.dispatch' in sys.modules and 'SlowPath' in dir(solvi.core)\n")
    run = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]

