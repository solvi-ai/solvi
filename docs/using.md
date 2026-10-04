# Using solvi

The high level: ready systems you configure, each assembled from the [building blocks](building_blocks.md) and each
replaceable part by part. Everything here is imported from `solvi` or `solvi.models` and is stable in 1.x.

```bash
pip install solvi              # Python 3.11+; the core needs numpy, scipy and pydantic
```

| you have | use | what you get |
|---|---|---|
| a question, labelled examples, a promise | [`solvi.build`](#decisions-solvibuild) | a decision system: System 1 fitted, its guarantee calibrated, a slow path and a person for what it hands over, every decision stored |
| an environment to act in | [`solvi.Agent`](#acting-in-an-environment-solviagent) | an agent that acts on what it knows, searches where it does not, keeps gates as hard checks, replays every step |
| an LLM agent that calls tools | [`solvi.Guard`](#an-agents-tool-calls-solviguard) | every tool call checked before it runs: allowed, denied with reasons, or sent to a person |
| facts, goals, rules from people, outcomes or a written policy | [`solvi.Knowledge`](#what-a-system-knows-solviknowledge) | one store for what the system knows and from whom, read by the three above |
| a model | [`solvi.models`](#models-solvimodels) | an LLM, a decision service or a local checkpoint as a decider |

The shared vocabulary — `Catalog`, `Question`, `Answer`, `System`, `Response`, `Quote`, `Claim`, `Decision`, `Fail`,
`Unknown`, the typed answers `Span`, `Maybe`, `Rank`, `Estimate`, `Scale`, `Bins`, and `Budget` — is exported by
`solvi` too: a catalog of your own functions and checks is how you tell any of these systems what to compute.

## Decisions: `solvi.build`

```python
import solvi
from solvi import Answer, Question

s = solvi.build(Question("team", "Which team?", Answer.choice(["billing", "shipping", "other"])), examples,
                catalog=cat, max_risk=0.02, storage="decisions.jsonl")   # examples: [(state, correct answer)]
res = s.ask({"email": "my parcel never came"})     # res.answer, res.by ("s1", "s2" or "human"), res.reasons, res.cost
print(s.explain())                                 # what System 1 is, its signal and promise, who answers each slice
print(s.report())                                  # what it did, read from the store
```

`build` composes what the library has, with defaults, and records every choice: System 1 is the catalog's rule, your
own fitted part (`learner=`) or a head fitted on the facts the catalog computes; its guarantee and — with `slow=` — who
answers what it hands over are calibrated on examples it did not see; a choice among more than two options gets a gate
for kinds of input no example shows. It does not make either path more accurate. The guide's
[quick start](guide.md#quick-start-solvibuild) has the details; `examples/24_one_entry_point.py` runs without a model.

Writing the catalog and the questions yourself, and asking a `System` directly, is the same library one step lower:
the guide's chapters from [Concepts](guide.md#concepts) to [Asking](guide.md#asking-system-and-response).

## Acting in an environment: `solvi.Agent`

```python
km = solvi.Knowledge("knowledge.jsonl", vocabulary={...})
agent = solvi.Agent(env, knowledge=km)          # env: reset(seed), actions(state), step(action) → Outcome
agent.run(seed=7, steps=150)
agent.report(); agent.replay()
```

System 1 takes an action the knowledge predicts will work and that advances an open goal; System 2 searches when it
has nothing it is sure of; the agenda's gates and the action model's hard refusals hold in both. Protection is the
default; justified risk (`risk=RiskBudget(...)`) is an option. The whole page, with what was and was not shown:
[Using solvi: agents and knowledge](agent.md).

## An agent's tool calls: `solvi.Guard`

```python
guard = solvi.Guard(storage="calls.db")

@guard.tool(ground=["iban", "amount"])       # these arguments must be quoted from the conversation
def pay(iban: str, amount: float) -> str:
    """Pay an invoice."""
    return bank.pay(iban, amount)

d = guard.call({"name": "pay", "arguments": {"iban": "DE89370400440532013000", "amount": 250}}, context=messages)
d.outcome                                    # "allow" (and the tool ran), "deny" with d.reasons, or "escalate"
```

The tool must be in the catalog, its arguments must validate, values that must come from the user must be quoted from
the user's own messages, your policies are hard checks, and an optional authorizer decides "did the user ask for
this?" under a guarantee. Every decision is a stored, replayable trace. Guide:
[Guarding an agent's tool calls](guide.md#guarding-an-agents-tool-calls); runnable: `examples/19_agent_guard.py`.

## What a system knows: `solvi.Knowledge`

One object holds the knowledge store (facts with their sources, retraction with everything derived from it), the
agenda (goals with done checks in code, gates) and the action model learned from outcomes. `solvi.build(...,
knowledge=km)` gives every decision a snapshot of it, `solvi.Guard(..., knowledge=km)` turns predicted refusals and
gates into hard checks on tool calls, and `solvi.Agent` acts on it. See
[Using solvi: agents and knowledge](agent.md#knowledge-one-object-for-what-the-system-knows).

## Models: `solvi.models`

```python
import solvi.models as models

m = models.llm("http://127.0.0.1:8080/v1", "qwen2.5-7b-instruct")     # any OpenAI-compatible server
m = models.systemone("https://...", "my-model")                       # a System One service
m = models.decider("solvi-base")                                      # a local checkpoint (solvi models pull ...)
part = m.decision("team", "Which team?", "email", ["billing", "shipping"])
```

A model proposes; checks, constraints, guarantees and rules decide. Whatever the model, read what it was measured on
and calibrate it on your own labelled stream before acting on its answers: the guide's
[model decisions](guide.md#the-model-proposes-decisions-with-a-decider) and
[best practices](best_practices.md#an-llm-as-a-decider).

## Tools around a system

- `solvi serve module:system` — the questions over HTTP, MCP and the System One API
  ([Serving](guide.md#serving-http-mcp-and-system-one)).
- `solvi init`, `solvi ask`, `solvi test`, `solvi check`, `solvi calibrate`, `solvi models`, `solvi report`,
  `solvi diff`, `solvi migrate` — the [command line](guide.md#command-line).
- `solvi.show` prints a response; `solvi.testing` runs decision regression tests ([Regression tests](testing.md));
  `solvi honesty` is the release gate ([Honesty suite](honesty.md)).

## Coming from 0.9

Every 0.9 import path still works in 1.0.x with a `SolviDeprecationWarning` naming the new path; 1.1 removes them.
`solvi migrate PATH` rewrites your code (`--check` only reports). The table of moves is in the
[changelog](../CHANGELOG.md#breaking-changes-and-migration).
