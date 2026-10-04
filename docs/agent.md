# Using solvi: agents and knowledge

Two entry points of the high level for systems that act and remember: `solvi.Knowledge` keeps what a system learned
and from whom, and `solvi.Agent` acts in an environment on that knowledge. Both are assembled from the low level
(`solvi.core.knowledge`, `solvi.core.dispatch`); every part can be replaced from there (see
[Building blocks](building_blocks.md)). Runnable: [examples/25_environment_agent.py](../examples/25_environment_agent.py).

## Knowledge: one object for what the system knows

```python
import solvi

km = solvi.Knowledge("knowledge.jsonl", vocabulary={"here": lambda s, a: s["here"], "wood": lambda s, a: s["wood"]})
km.tell({"s": "c17", "r": "tier", "o": "vip"}, source="person", by="crm")    # a fact, with its source
km.goal("wood", done=lambda s: s["wood"])                                   # an agenda goal: done is code
km.goal("table", done=lambda s: s["table"], requires=["wood"])
km.agenda.gate("tool_first", lambda s: s["pickaxe"], blocks=["collect_stone"])   # a gate: a hard check
km.observe(state, "collect_wood", outcome)        # an environment step: action model, map facts, agenda, skills
km.retract(item_id, why="wrong label")            # with everything derived from it
km.report()
```

`Knowledge(path=None, *, vocabulary=None, write_gate=None, actions=None, failures=None)` holds a `KnowledgeStore`
(`km.store`: the hash-chained journal), an `Agenda` (`km.agenda`: goals, gates, order — in the same journal), an action
model (`km.actions`: `ConservativeActionModel` over the vocabulary you give, or your own `ActionModel` with
`actions=`) and, with `failures=True`, a failure memory (`km.failures`). Its methods: `tell` (a fact from a person, an
outcome, a written spec or a verified System 2 answer — never the system's own answer: the source check refuses it),
`retract` (with the decisions that rested on the item, split into answer changes and justification only, when you pass
the decisions' store), `goal`, `observe` and `report`.

## Agent: acting in an environment

```python
from solvi.core.knowledge import RiskBudget

agent = solvi.Agent(env, knowledge=km, key=lambda s: s["place"])   # protection: a predicted refusal is never taken
agent.run(seed=7, steps=150)                                      # one episode → its numbers
agent.run(seed=7, steps=150)                                      # the same world again: fewer slow decisions
agent.report(); agent.replay()                                    # what it did; every decision re-checked
solvi.Agent(env, knowledge=km, risk=RiskBudget(max_risk_per_episode=1.0))   # justified risk, within a budget
```

`env` is an `Environment` (`reset(seed)`, `actions(state)`, `step(action) → Outcome`). Each step is one decision of
a `Dispatcher` (`agent.dispatcher`) over System 1 (`agent.system`):

- **System 1** takes an action the knowledge predicts will work and that advances an open goal: a goal's skill (the
  action after which its done check turned true before) or the first step of a confirmed route to where the skill
  worked. With the default protection that means the action model predicts "accept".
- **Hard checks**, in System 1 and System 2 alike: the agenda's gates, a hard refusal of the action model (never
  traded), what the risk policy avoids, and the failure memory.
- **System 2** — when System 1 has nothing it is sure of — is a search over the offered actions in an exploration order
  (an open goal's action first, then what was never tried here, then the way to the nearest place with untried
  actions), through the same hard checks; an action the model is unsure of may be tried there. `s2=` takes a
  `SlowPath` of your own, `budget=` limits System 2 per episode.
- **Written back** after every step: the action model learns from the outcome, a goal done right after an action makes
  a skill, and where each action led becomes a map fact. Rules (the action model, skills) are carried across
  episodes; map facts belong to the map an episode plays on (`run(seed, steps, map=)`, default: the seed), and the
  first arrival that contradicts a carried map fact drops the carried map — its facts become hints until confirmed.
- **Recorded**: every decision is stored (`storage=`) with the facts it was given — the options, each with its
  prediction, risk, support, blocks and goal — and `agent.replay()` re-checks them without the environment.

In example 25 (a toy crafting world: 8 places, six operators, six goals), the agent explored the world in its first
run — 61 steps, all 61 decided by System 2, 23 actions refused — and reached all six goals in 12 steps the second
time, 11 of them by System 1 and none refused. In a new world it kept the rules and re-learned the map: 17 steps, 13 by
System 2, 3 refused. Every decision replayed.

## Protection by default, justified risk as an option

Knowledge used only as hard gates removes the failures it targets, and it can cost a goal that needs a risk. `risk=None`
is `Protect`: a predicted refusal is never taken. `risk=RiskBudget(max_risk_per_episode=, min_gain_ratio=,
min_support=)` takes a refused action when its expected gain (`gain=`, a function of the state and the action; default:
1 for an action that advances an open goal) is at least `min_gain_ratio` times its estimated risk, within a per-episode
budget; a refusal resting on fewer than `min_support` cases goes to System 2. Gates, hard predictions and the failure
memory are never traded in any mode.

In example 25's second world the iron lies across a bridge that breaks one time in three (the episode ends), behind a
written gate "no bridge without the stone pickaxe". Over 30 episodes: with protection the agent fell once and never
crossed again — iron in 2 episodes, 5.07 goals of 6 per episode; with `RiskBudget(max_risk_per_episode=1.0,
min_gain_ratio=1.0, min_support=1)` it crossed 27 times against the prediction, fell 6 times and got the iron in 24
episodes, 5.80 goals per episode. The gate held in both. Measure `RiskBudget` on your own metric before you turn it
on: in a dungeon game it won back part of what protection cost, not all of it, and it does not promise parity with an
agent that has no knowledge.

## Knowledge in decisions and tool calls

```python
s = solvi.build(question, examples, catalog=cat, max_risk=0.02, knowledge=km)
guard = solvi.Guard(storage="calls.db", knowledge=km)
```

`build(..., knowledge=km)` gives every decision — and every example at build time — `km.snapshot()` as the fact
`knowledge`, which System 1's rules read like any fact (`solvi.Knowledge.value(knowledge, "c17", "tier")`); the trace
records what the memory said, so the decision replays. A function `state → snapshot` instead of `km` gives your own
query (`lambda s: km.snapshot(about=s["customer"])`).

`Guard(..., knowledge=km)` adds two hard checks to every tool: the action model's prediction over the facts your app
gives (`facts=`) — a predicted refusal denies the call, "unknown" leaves it to the other checks — and the agenda's
gates that block the tool. `guard.observe(decision, accepted)` tells the knowledge what the environment did.

## What it does and does not promise

- **Growth was shown in environments met again**: a crafting game and the Pokémon world map
  ([example 23](../examples/23_pokemon_world_map.py): 147 slow decisions of 183 in the first run, 2 of 62 in the
  second). `benchmarks/knowledge/toy_crafting.py` runs the protocol on a toy.
- **It was not shown on decision streams** — classification and matching, where a static System 1 already answers most
  inputs and facts added no measurable gain — **nor on a support agent with tools**, where a learned memory did not
  improve over time and a hand-written guard did better than the memory learning the same ground. There the store
  gives accountability (journal, sources, retraction, disputes), not growth.
- **The action model learns only what the environment checks, over your vocabulary.** A rule the environment does not
  enforce (confirm with the customer, authenticate first) comes from a written policy as gates. Rules learned in one
  world can be over-cautious in another: conditions that held at every success by accident do not transfer ("unknown",
  not wrong).
- **A learned memory on top of written gates added nothing** where both were measured: they substitute, they do not
  add. Where the rules are written, write them as gates.
- **Validate gates compiled from policy text before you make them hard**: a gate read from a policy ("remind the
  customer to confirm every item") can block the calls a customer wanted. `km.agenda.dry_run(recorded_successes)`
  reports, per gate, how many recorded successful actions it would have blocked.
- **Justified risk reduces the cost of protection; it does not promise parity** with an agent without knowledge.
- The store has been measured to 10,000 items (`benchmarks/knowledge/retraction.py`), not beyond.

Reference: [`solvi.solutions.agent`](api/solutions_agent.md), [`solvi.solutions.knowledge`](api/solutions_knowledge.md),
[`solvi.core.knowledge`](api/knowledge.md); the guide's [Knowledge](guide.md#knowledge-what-a-system-learned-and-from-whom)
section for the low level.
