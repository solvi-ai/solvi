# Building blocks

The ready-made systems (`solvi.build`, `solvi.Agent`, `solvi.Guard`, `solvi.Knowledge` — see [Using solvi](using.md))
are assembled from the low level, `solvi.core`. Every place where you can put a part of your own is exported there,
with what you implement, what you then get for free, and how far you can rely on it:

```python
from solvi.core import SlowPath, TraceStorage, Strategist, Head, Monitor, Environment   # and the rest of the table
```

Three kinds of extension point:

- **protocols** (`typing.Protocol`, runtime-checkable): an object with these methods is one — no subclassing;
  `isinstance(x, Head)` checks that the methods are there (not their behaviour);
- **base classes** with abstract methods (`TraceStorage`, `SlowPath`): subclass, implement the abstract methods, and
  the rest is inherited; a subclass missing one cannot be constructed;
- **concrete classes** (`System`, `Response`, `Dispatcher`): use them, extend them through what they take (a
  strategist, a store, a slow path, a monitor), do not subclass them.

Promises: **stable** — no breaking change in 1.x, a deprecation lasts at least one minor release; **stable to use,
provisional to subclass** — calling it is stable, a subclass may need a change in a minor release. The names load on
first use, so `import solvi.core` stays light.

"For free" is tested, not claimed: `solvi.testing.conformance` has a check per extension point that runs your
implementation through the parts of solvi that rely on it, and solvi's own test suite runs every check on every
built-in (`tests/test_conformance.py`).

```python
from solvi.testing.conformance import check_storage


def test_my_store(tmp_path):
    check_storage(lambda: MyStorage(tmp_path / "s"), reopen=lambda: MyStorage(tmp_path / "s"))
```

| name | kind | you implement | you get for free | check | promise |
|---|---|---|---|---|---|
| [`Scorer`](#scorer) | protocol | `logits(items)` | questions from types, adaptation, fit / teach, calibration and guarantees, identity in traces (through `DecideModel`) | — | stable |
| [`Decider`](#decider) | protocol | `__call__(**facts) → Decision`, `options`, `fingerprint()` | closed set, constraints, guarantees on its signal, identity recorded, replay | `check_decider` | stable |
| [`Adapter`](#adapter) | protocol | `kind`, `fingerprint()`, `using(scorer, active)`, `save(path)`, `load(path)` | in the model's fingerprint and every decision's record, saved and loaded with the calibration | — | stable |
| [`Head`](#head) | protocol | `options`, `features`, `fit`, `predict`, `contributions`, `teach`, `fingerprint` | `System.fit(head=)`, answers with abstention on missing facts, teach, guarantee, replay | `check_head` | stable |
| [`Extractor`](#extractor) | protocol | `find(text, field) → [Quote]`, `fingerprint()` | parsing per type, "not stated", literal-quote check, provenance, replay | `check_extractor` | stable |
| [`Strategist`](#strategist) | protocol | `plan(catalog, questions, init_keys, heads) → Flow` | hard checks enforced, plan recorded and replayed | `check_strategist` | stable |
| [`TraceStorage`](#tracestorage) | base class | `_append`, `_raw`, `_find`, `head`, `_rewrite` | hash chain, verify, replay, query, report, redact, signature, quarantine | `check_storage` | stable, format stable |
| [`Monitor`](#monitor) | protocol | `observe`, `flagged`, `reset` | wired into the Dispatcher (drift signal) and the system report | `check_monitor` | stable |
| [`Proposer`](#proposer) | protocol | `(state, rounds) → proposal` | checks judge it, reasons fed back, rounds and budget, replay | via `check_slow_path` | stable |
| [`Space`](#space) | protocol | `root`, `children(node)` (+ `complete`, `bound`) | pruning by checks, lean asks, budget, exactness, replay | via `check_slow_path` | stable |
| [`SlowPath`](#slowpath) | base class | `mode`, `think(...) → Thought`, `fingerprint()` | routing, budgets, cost, calibration per slice, dispatch record, replay, report | `check_slow_path` | stable to use, provisional to subclass |
| [`Environment`](#environment) | protocol | `reset(seed)`, `actions(state)`, `step(action) → Outcome` | everything the environment agent (`solvi.Agent`) does | `check_environment` | stable to use, provisional to implement |
| [`KnowledgeStore`](#knowledge) | concrete | — (any `TraceStorage` backend underneath) | source check, disputes to a person, exact retraction cascade, staleness flags, snapshots that replay, redecide | `check_storage` (its backend) | stable |
| [`WriteGate`](#knowledge) | protocol | `admit(store, item, shadow) → Verdict` | runs after the source check on every proposal, verdict journaled, holds behaviour-changing items | — | stable |
| [`ActionModel`](#knowledge) | protocol | `observe(...)`, `predict(state, action, args) → Prediction`, `fingerprint()` | refusals as hard checks, unknown → System 2, prediction vs outcome recorded | `check_action_model` | stable (scope stated) |
| [`Vocabulary`](#knowledge) | value | `{name: predicate(state, args)}`, `hard=` | predicate fingerprints in every action item | — | stable |
| [`Agenda`](#knowledge) | concrete | done checks and gate checks in code | open / blocked / done, journaled, no action past a gate, overrides, `dry_run` | — | stable |
| [`RiskPolicy`](#knowledge) | protocol | `decide(action, prediction, gain, key=) → RiskDecision`, `new_episode()` | every risky decision recorded; `Protect` default, `RiskBudget` option | — | stable |
| [`System`, `Response`, `Dispatcher`](#system-response-dispatcher) | concrete | — | — | — | stable (final) |

The catalog's own extension point is the function part (`cat.fn`, `cat.extract`, `cat.check`, `cat.rule`,
`cat.constraint`): a typed function, and the planning, validation, provenance, trace, replay, hard checks and audit
come with it — see the [guide](guide.md). The value classes it meets are exported here too: `Catalog`, `Part`,
`Question`, `Answer`, `AnswerType`, `Quote`, `Claim`, `Decision`, `Fail`, `Unknown`.

## Rebuilding one part of a ready system

Each ready system takes its parts as arguments, so one part can be yours and the rest stays as it is:

| ready system | the part | how |
|---|---|---|
| `solvi.build` | System 1 | a rule of your catalog for the question (`@cat.rule`), or `learner=`: a function `[(state, answer)] → a decision part` (your classifier wrapped as a part) |
| `solvi.build` | System 2 | `slow=`: a [`SlowPath`](#slowpath) of your own, a `System`, a decision part, or a function `state → answer` |
| `solvi.build` | the store | `storage=`: a path, or a [`TraceStorage`](#tracestorage) of your own |
| `solvi.Agent` | the action model | `solvi.Knowledge(actions=...)`: an [`ActionModel`](#knowledge) of your own (read from a game's data file, say) |
| `solvi.Agent` | System 2 | `s2=`: a [`SlowPath`](#slowpath); `budget=` limits it per episode |
| `solvi.Agent` | risk | `risk=`: `Protect` (default), `RiskBudget(...)`, or a [`RiskPolicy`](#knowledge) of your own |
| `solvi.Guard` | "did the user ask for this?" | `authorizer=`: any decision part, e.g. a decider's `decision(...)` with `act_guard` |
| `solvi.Knowledge` | what may be written | `write_gate=`: a [`WriteGate`](#knowledge) that runs after the built-in source check |

## The areas of `solvi.core`

Where each area is explained at length (the guide's chapters of the Building blocks part) and its API page:

| area | what it holds | guide | API |
|---|---|---|---|
| `solvi.core.plan` | the strategist: a flow per request; the cost strategist | [How the strategist plans a flow](guide.md#how-the-strategist-plans-a-flow), [Code strategist](strategist.md) | [strategist](api/strategist.md), [cost](api/strategy.md) |
| `solvi.core.deciders` | decision parts, deciders, heads, rule lists, combinations | [Types, questions and model decisions](guide.md#types-questions-and-model-decisions), [checkpoint format](decide_format.md) | [deciders](api/decide.md), [protocols](api/protocols.md), [combine](api/multi.md), [heads](api/heads.md) |
| `solvi.core.calibration`, `solvi.core.guarantees` | reliability, thresholds with a promise, open-set gate, drift, monitors | [Confidence, calibration and abstention](guide.md#confidence-calibration-and-abstention) | [calibration](api/calibration.md), [guarantee](api/guarantee.md), [openset](api/openset.md), [drift](api/drift.md), [monitor](api/monitor.md) |
| `solvi.core.store`, `solvi.core.runtime` | the trace, hash-chained stores, replay, audit, reports, diff, signature | [The trace and verification](guide.md#the-trace-and-verification), [Grounded decisions](guide.md#grounded-decisions-provenance-audit-and-safeguards) | [store](api/storage.md), [audit](api/audit.md), [report](api/report.md), [sysreport](api/sysreport.md), [diff](api/diff.md), [signature](api/signature.md) |
| `solvi.core.slow` | generation, agreement, the re-ask loop, search | [A model that writes](guide.md#a-model-that-writes-generation-agreement-and-the-re-ask-loop) | [generate](api/generate.md), [agree](api/agree.md), [refine](api/refine.md), [search](api/search.md) |
| `solvi.core.dispatch` | System 1, a slow path or a person, with a budget | [Who answers](guide.md#who-answers-system-1-the-slow-path-or-a-person-solvicoredispatch) | [dispatch](api/dispatch.md) |
| `solvi.core.knowledge` | the knowledge store, write gates, action model, risk, agenda, failure memory, world map, episodes, corrections | [Knowledge](guide.md#knowledge-what-a-system-learned-and-from-whom), [the Pokémon world map](guide.md#system-1-and-system-2-on-a-game-the-pokémon-world-map) | [knowledge](api/knowledge.md) |
| `solvi.core.extract`, `solvi.core.textin` | fields from documents by description; text in | [Extracting fields from documents](guide.md#extracting-fields-from-documents), [Text in](guide.md#text-in-from-a-message-to-a-question) | [extract](api/extract_long.md), [textin](api/textin.md) |

## Scorer

A model's raw scores. `DecideModel(scorer)` turns any object with `logits(items)` into a decider: one array per item
(a logit per option, a column per mode), optionally `logits_pass(passes)` for several questions in one forward pass,
`fingerprint()` and `model_id`. The built-ins are `OnnxScorer` and `TorchScorer` (`solvi.core.deciders`).

Reference: [`Scorer`](api/protocols.md#solvi.core.deciders.protocols.Scorer).

## Decider

A model as a catalog part: called with the facts it reads, it returns a `Decision(value, probs)` within its `options`.
A decision part (`model.decision(...)`) and `Cascade` / `Vote` / `Route` are deciders and bring their options and
identity to the catalog themselves; an object of your own is registered with both:

```python
from solvi import Answer, Catalog, Decision, Question, System


class Keyword:
    __name__ = "team"                       # the fact it produces
    options = ["billing", "shipping"]

    def fingerprint(self):
        return "keyword-1"

    def __call__(self, email):
        p = 0.9 if "charged" in email else 0.2
        return Decision("billing" if p > 0.5 else "shipping", {"billing": p, "shipping": 1 - p})


dec = Keyword()
cat = Catalog()
cat.fn(dec, model=dec, options=dec.options)


@cat.rule("route")
def route(team):
    return team


system = System(cat, [Question("route", "Route", Answer.choice(["billing", "shipping"]))])
```

`check_decider(dec, inputs, system=system, states=...)` checks the closed set, the probabilities, determinism and that
every decision records the decider's fingerprint and replays.

Reference: [`Decider`](api/protocols.md#solvi.core.deciders.protocols.Decider).

## Adapter

A per-question adaptation of a model's weights in a decision part's adapter slot (`model.loras`). solvi reads only the
protocol: the adapter's `fingerprint()` goes into the model's fingerprint and into the record of every decision it made
(`extra["lora"]["adapter"]`), `using(scorer, active)` switches it on while scoring, and `save` / `load` write it next
to a calibration file and read it back (checked against the recorded fingerprint). `kind` names the module that reads
its files (`solvi.core.deciders.ADAPTERS`). The one built-in, LoRA, is experimental (`solvi.experimental.lora`).

Reference: [`Adapter`](api/protocols.md#solvi.core.deciders.protocols.Adapter).

## Head

A learned answer head: the probability of each option from the facts the System computed. `System.fit` builds a
`FastHead` (ridge, closed form) by default; `head=` gives your own, one per option for a multi-label question:

```python
from solvi.core.deciders.heads import FastHead

system.fit("priority", examples, head=lambda options: FastHead(options, lam=1.0))
```

The System then answers from it — abstaining when one of its features could not be computed or its probabilities are
not finite — records its fingerprint with every answer (a replay with another head is a mismatch), updates it on
`System.teach` and puts guarantees on its confidence. `check_head(make, options, rows, answers)` checks the
distribution, determinism, that `teach` changes the fingerprint, and the System wiring.

Reference: [`Head`](api/protocols.md#solvi.core.deciders.protocols.Head).

## Extractor

What points at the piece of a text that holds a field when `TextIn` reads a message into a typed state:
`find(text, field)` → quotes, best first. `CueExtractor` (deterministic, the default) and `DeciderExtractor` (the
decider's span pointer) are the built-ins; `TextIn(system, extractor=[mine, CueExtractor()])` tries several in turn. A
deterministic parser turns the quote into the value, a field the text does not state is "not stated", and replay
checks that the quote is literally in the text. `check_extractor(extractor, system, texts)` checks literal quotes,
determinism and replay.

Reference: [`Extractor`](api/textin.md#solvi.core.textin.Extractor).

## Strategist

The planner of a System: which parts and checks run for these questions on these given facts. `System(strategist=)`
takes one; without one, the deterministic planner runs, which is also a class, `DefaultStrategist`, to subclass when
you change one step:

```python
from solvi.core.plan.strategist import DefaultStrategist


class Logged(DefaultStrategist):
    def plan(self, catalog, questions, init_keys, heads=None):
        flow = super().plan(catalog, questions, init_keys, heads)
        print(flow)
        return flow
```

Whatever the plan, a hard check whose `then=` names a question must be in that question's flow — `System(...)` refuses
a strategist that leaves one out. `CostStrategist` (`solvi.core.plan.cost`) picks the cheapest verified plan by
declared costs and records it in the trace. `check_strategist(strategist, catalog, questions, states)` checks the order
of the steps, determinism, the hard checks and replay.

Reference: [`Strategist`](api/strategist.md#solvi.core.plan.strategist.Strategist), [`DefaultStrategist`](api/strategist.md#solvi.core.plan.strategist.DefaultStrategist).

## TraceStorage

The base class of every decision store: JSON lines, SQLite, PostgreSQL and DuckDB are subclasses. A backend implements
five methods and inherits the rest — the hash chain and `verify`, `save` / `get` / `rederive`, corrections and verified
labels, `query`, `report`, `replay_all`, `redact`, `signature`, `quarantine`. A store in memory:

```python
import json

from solvi.core import TraceStorage
from solvi.core.store import record_hash


class MemoryStorage(TraceStorage):
    def __init__(self, system=None):
        super().__init__(system)
        self.rows = []

    def _append(self, body):
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
        return {"count": len(self.rows), "hash": self.rows[-1]["hash"] if self.rows else ""}

    def _rewrite(self, rec):
        self.rows[rec["seq"]] = rec
```

`check_storage(make, reopen=)` stores decisions and a correction through a System, verifies the chain, reads them back,
replays them, edits an answer in place through `_rewrite` (verify must catch it), redacts one (the chain must still
verify) and, with `reopen`, opens the store again.

Reference: [`TraceStorage`](api/storage.md#solvi.core.store.TraceStorage).

## Monitor

A watcher of the decision stream: `observe(decision)` → a report with `drift` and `why`, `flagged`, `reset()`.
`DriftMonitor` (`solvi.core.guarantees.drift`) is the built-in. `Dispatcher(monitor=...)` feeds it every System 1
response and turns its flag into the "drift" signal (the slow path checks System 1's answers until `reset_drift()`);
`system_report(store, monitor=lambda: MyMonitor())` runs a fresh one over a stored stream. `check_monitor(make, stream,
changed=)` checks the reports, the flag on a changed stream and `reset`.

Reference: [`Monitor`](api/monitor.md#solvi.core.guarantees.monitor.Monitor).

## Proposer

What proposes in a refinement: `propose(state, rounds)` → a proposal, having seen what the checks said about the earlier
rounds' proposals. `Generator.proposer(...)` (`solvi.core.slow.generate`) builds one from an LLM; a plain function is
one.

Reference: [`Proposer`](api/refine.md#solvi.core.slow.refine.Proposer).

## Space

A space of candidates walked depth-first by `search` and `SearchPath`: `root` and `children(node)`, with `complete(node)`
and `bound(node)` when you have them. `Tree(root, children, complete, bound)` is the ready one; a list, a dict of domains
or a function of the facts are spaces too.

Reference: [`Space`](api/search.md#solvi.core.slow.search.Space).

## SlowPath

System 2 for the [dispatcher](guide.md): a slow, checked way to answer when System 1 cannot answer alone. The built-ins
are `AskPath(system)` (a System that answers the question itself), `RefinePath(system, propose=, into=)` and
`SearchPath(system, space=, into=)`. A path of your own subclasses `SlowPath`: a `mode` (a name of its own), `think`
and `fingerprint`; the base class's `run` adds the cost, and the dispatcher the routing, the budgets, the calibration
per slice, the records and the replay:

```python
from solvi.core.dispatch import SlowPath, Thought


class FirstSure(SlowPath):
    """Ask several Systems in turn; the first that does not abstain answers."""
    mode = "first-sure"

    def __init__(self, systems):
        super().__init__(systems[0])
        self.systems = systems

    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        for s in self.systems:
            res = s.ask(dict(state), [question], store=store)
            if res[question].status != "abstain":
                return Thought(self.mode, res[question].answer, True, None, res)
        return Thought(self.mode, None, False, "every System abstained", res)

    def fingerprint(self):                               # what its answers depend on: the Systems' catalogs
        return "first-sure:" + ",".join(s.fingerprint()["catalog"] for s in self.systems)

    def replay(self, thought, trust_models=False):     # the answering System's trace (the default: the mode only)
        res = thought.record
        system = next(s for s in self.systems if s.fingerprint()["catalog"] == res.trace.fingerprint["catalog"])
        rep = res.trace.replay(system, trust_models=trust_models)
        bad = [("trace", str(m)) for m in rep["mismatches"]]
        if (next(iter(res.results.values())).status != "abstain") != thought.accepted:
            bad.append(("accepted", "not what the record gives"))
        return {"ok": not bad, "mismatches": bad}

    @classmethod
    def restore_record(cls, d, system=None):            # the record is the last Response, stored as a dict
        from solvi import Response
        return Response.model_validate(d)

    @classmethod
    def responses_of(cls, record):                      # its model outputs are the path's cost
        return [record]
```

A stored `Thought` is read back by the class of its mode (`SlowPath.modes`): override `restore_record`,
`responses_of` and `generated_of` for a record of your own, `replay` to re-derive the answer and its acceptance from
it, `signal` for the number `Dispatcher.calibrate` thresholds, `steps` when one run takes several (the budget is then
checked between them). `check_slow_path(path, states, system1=)` checks the Thoughts, their round trip through JSON,
replay (including that a Thought whose acceptance is not its record's is caught), the cost, the budget, and stored
dispatcher decisions. `SlowPath(system, propose=..., space=...)` — the 0.9 constructor — still builds the matching
built-in in 1.0.x, with a `SolviDeprecationWarning`.

Reference: [`SlowPath`](api/dispatch.md#solvi.core.dispatch.SlowPath), [`Thought`](api/dispatch.md#solvi.core.dispatch.Thought).

## Environment

A world acted in step by step: `reset(seed)` → the first state, `actions(state)`, `step(action)` →
`Outcome(state, accepted, effect, done)`. It is what the environment agent (`solvi.Agent`, see
[Using solvi: agents and knowledge](agent.md)) runs on, and what an action model learns from: the effects and refusals
the environment itself reports.
`check_environment(make)` checks that the same seed and actions give the same outcomes and that a refused action leaves
the state as it was.

Reference: [`Environment`](api/environment.md#solvi.core.environment.Environment), [`Outcome`](api/environment.md#solvi.core.environment.Outcome).

## System, Response, Dispatcher

The concrete classes everything above plugs into: `System` (a catalog and its questions), `Response` (one ask's
answers, flow and trace) and `Dispatcher` (System 1, the slow path or a person, with a budget). Stable and final: use
them, extend them through what they take. Their references: [solvi](api/solvi.md), [solvi.core.dispatch](api/dispatch.md).

## Knowledge

What a system knows across decisions (`solvi.core.knowledge`; the guide's
[Knowledge](guide.md#knowledge-what-a-system-learned-and-from-whom) section shows the pieces together).

- **`KnowledgeStore`** (concrete, stable): a hash-chained journal of facts, rules, skills, actions and episodes over any
  `TraceStorage` — yours included — or in memory. You get the source check (never the system's own answers), disputes
  to a person, the exact retraction cascade, staleness flags, `snapshot` for decisions (with the store's fingerprint,
  so they replay), `redecide`, `verify` and `rebuild`. A store of your own backend is checked by `check_storage`.
- **`WriteGate`** (protocol, stable): `admit(store, item, shadow) → Verdict(admit, reason, measured)`. Your gate runs
  after the built-in source check on every proposed item; its verdict is journaled; holding a rule, skill or action
  keeps it a hypothesis.
- **`ActionModel`** (protocol, stable): `observe(state, action, args, accepted, effect)`, `predict(state, action, args) →
  Prediction(verdict, risk, support, reason, hard, effects)`, `fingerprint()`. A model of your own — read from a game's
  data file, say — plugs in where `ConservativeActionModel` does: its refusals become hard checks, "unknown" goes to
  System 2, its prediction is compared with what happened. `check_action_model(make, transitions, held_out=)` checks
  the Predictions, determinism, the fingerprint, and that it never contradicts an outcome it observed.
  `ConservativeActionModel` is stable with its scope stated: vocabulary-bound, it learns what the environment checks,
  and sufficient conditions are not promised to transfer to another world.
- **`Vocabulary`** (value, stable): `{name: predicate(state, args)}`, `hard=` names that are hard rules; the
  predicates' code fingerprints are in every action item.
- **`Agenda`** (concrete, stable): `goal(name, done=, requires=, gates=)`, `gate(name, check, blocks=)`; you write the
  done checks and gate checks in code; you get open / blocked / done, every state change journaled, no action past a
  gate, person overrides, and `dry_run` to validate a gate on recorded successes.
- **`RiskPolicy`** (protocol, stable): `decide(action, prediction, gain, key=) → RiskDecision`, `new_episode()`.
  `Protect` (the default) keeps the verdict; `RiskBudget` takes justified risks within a per-episode budget and never
  a hard prediction. A policy of your own receives every prediction with its risk and support and must record why.

Reference: [`solvi.core.knowledge`](api/knowledge.md) and its modules ([store](api/knowledge_store.md),
[gates](api/knowledge_gates.md), [actions](api/knowledge_actions.md), [risk](api/knowledge_risk.md),
[agenda](api/knowledge_agenda.md), [failures](api/knowledge_failures.md)).
