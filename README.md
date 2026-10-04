# solvi

Decision systems you can check. You describe what to compute with plain Python functions and checks; a model — an LLM,
your own classifier or a small local one — proposes where judgement is needed; solvi's checks decide. Every answer
comes with a confidence, a reason you can check (the rule's inputs, a formula over computed facts, or a quote with
character offsets in the source) and a hash-chained trace that can be re-executed later to confirm the answer or
pinpoint the step that was altered. When something cannot be computed or checked, solvi abstains or hands the case to
a person instead of guessing; a failed hard check always overrides any model's confidence.

solvi 1.0 has two levels. **Ready systems** you configure: `solvi.build` (decisions from labelled examples, with a
promise on the errors), `solvi.Agent` (acting in an environment), `solvi.Guard` (an agent's tool calls) and
`solvi.Knowledge` (what they know, and from whom). **Building blocks** in `solvi.core`, which the ready systems are
made of, each replaceable by a part of your own. What works but has no measured gain yet is kept apart in
`solvi.experimental`.

## Install

```bash
pip install solvi              # core: rules, checks, learned answer heads, build / Agent / Guard (numpy, scipy, pydantic)
pip install "solvi[model]"     # + torch, transformers: ModernBERT field extractors for documents and the decider
pip install "solvi[onnx]"      # + onnxruntime, tokenizers: the decider (solvi.models.decider) on CPU without torch
pip install "solvi[serve]"     # + fastapi, uvicorn: `solvi serve app.py:system` — the questions over HTTP (also --mcp)
pip install "solvi[mcp]"       # + the official MCP SDK for solvi serve --mcp (without it, a built-in stdio server)
pip install "solvi[duckdb]"    # + duckdb: stored decisions in a DuckDB file (solvi.core.store.DuckDBStorage); [postgres] for PostgreSQL
pip install "solvi[lora]"      # + torch, transformers, peft: solvi.experimental.lora.adapt_lora, a LoRA adapter per question (experimental)
```

Requires Python 3.11+ (tested on 3.11–3.14). Coming from 0.9: see [below](#coming-from-09).

## Start

### Decisions: `solvi.build`

Give the question, labelled examples and the promise — and a slow path if you have one. `solvi.build` fits System 1,
calibrates its guarantee and who answers what it hands over on examples it did not see, stores every decision, and
says what it chose:

```python
import solvi
from solvi.models import llm

s = solvi.build(question, examples, catalog=cat, max_risk=0.02,     # examples: [(state, correct answer)]
                slow=llm(URL, "openai/gpt-oss-120b"), price=(0.037, 0.17), storage="decisions.jsonl")
res = s.ask(state)              # res.answer, res.by ("s1", "s2" or "human"), res.reasons, res.cost
print(s.explain())              # System 1, its signal and promise, who answers each slice, what is not covered
```

On four tasks of the stand (Banking77, Abt-Buy, CUAD, RAGTruth) the defaults matched or beat the hand-written setups
with every promise kept on eval, in a quarter to half of the code — closer to the promised level than the hand-written
ones, and the slow path was rarely given anything ([guide](docs/guide.md#quick-start-solvibuild),
[examples/24_one_entry_point.py](examples/24_one_entry_point.py)).

### Environments: `solvi.Agent` and `solvi.Knowledge`

```python
km = solvi.Knowledge("knowledge.jsonl", vocabulary={"here": lambda s, a: s["here"], "wood": lambda s, a: s["wood"]})
km.goal("table", done=lambda s: s["table"])     # an agenda goal: done is code
agent = solvi.Agent(env, knowledge=km)          # env: reset(seed), actions(state), step(action) → Outcome
agent.run(seed=7, steps=150)                    # explore; the same world again needs fewer slow decisions
agent.replay()                                  # every decision re-checked from what it was given
```

`solvi.Knowledge` keeps what a system learned and from whom: facts with their sources (never the system's own
answers), goals and gates, an action model learned from outcomes; a retraction takes back everything derived from the
item. `solvi.Agent` acts on it: System 1 where the knowledge predicts the action works, a search where it does not,
gates and predicted refusals as hard checks — protection by default, justified risk (`RiskBudget`) as an option. In
the toy crafting world of [examples/25_environment_agent.py](examples/25_environment_agent.py) the first run takes 61
steps, all decided by System 2, and the same world met again 12 steps, 11 of them by System 1. Growth was shown in
environments met again, not on decision streams or a support agent with tools; justified risk lowers the cost of
protection without promising parity ([docs/agent.md](docs/agent.md)). `solvi.build(..., knowledge=km)` and
`solvi.Guard(..., knowledge=km)` read the same knowledge.

### Tool calls: `solvi.Guard`

The agent proposes `{"name": tool, "arguments": {...}}`; `solvi.Guard` checks it — the tool is in the catalog, the
arguments validate against its types, the values that must come from the conversation are quoted there (and not only
from a tool output that says "ignore previous instructions"), your policies (limits, roles, allow-lists) are ordinary
hard checks, and an optional decider asks "did the user ask for this?" under `act_guard` and `perturb` — then allows
it (solvi runs the function), denies it with the reasons, or escalates it to a person. Every decision is a stored,
replayable trace; any framework's tool calls go through `guard.check` / `guard.call`
([guide](docs/guide.md#guarding-an-agents-tool-calls), [examples/19_agent_guard.py](examples/19_agent_guard.py)). On
τ-bench retail no call it let through was refused by the environment, at a cost in solved tasks (the τ-bench row of
the [task table](#nine-public-tasks)).

### Models: `solvi.models`

Types declare questions; a model proposes; checks decide. The fields of a pydantic model are the questions, their types
the kinds (one option, several, an ordered score, yes/no); a **decider** answers them about a text or a JSON / pydantic
state with probabilities, a calibrated confidence and act / escalate, and hard checks, constraints and rules still
decide. The decider is whichever model you have:

- **An LLM** — `solvi.models.llm(base_url, model, api_key=...)`: any OpenAI-compatible chat-completions server (OpenAI,
  OpenRouter, vLLM, llama.cpp, Ollama). Nothing to install beyond the core; its JSON replies are validated, and an invalid
  one escalates, never a guess.
- **A decision service** — `solvi.models.systemone(url, model)`: anything that speaks the System One API.
- **A local checkpoint, for offline or cheap cases** — `solvi.models.decider("solvi-base")` (`solvi[onnx]`, no GPU;
  `solvi models pull solvi-ai/solvi-base` downloads it). solvi-base is a 150M ModernBERT-base cross-encoder distilled
  from solvi-large: about 50 ms per question on a CPU (ONNX fp16, 4 threads). Its card is honest about where it stands: 54.5% zero-shot on typed questions over JSON states
  (the same as solvi-large), 56.3% on Fast Decisions dev — not better than earlier small models there — and 0.602 on the
  jabr classifier benchmark, where Jev reaches 0.966. A preview: fit it on 30–60 labelled examples of your task and
  calibrate its escalation on your own stream before relying on it.

```python
import os
from typing import Literal
from pydantic import BaseModel, Field
from solvi import Catalog, Scale, System
from solvi.models import llm

class Triage(BaseModel):
    team: Literal["billing", "technical", "shipping"] = Field(description="Which team should handle this ticket?")
    urgency: Scale[Literal["low", "medium", "high", "critical"]] = Field(description="How urgent is it?")
    angry: bool = Field(description="Is the customer angry?")
    topics: list[Literal["refund", "delay", "bug"]] = Field(description="What does the ticket mention?")

model = llm("https://api.openai.com/v1", "gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
# offline: model = solvi.models.decider("solvi-base")   — the same questions, the same checks
cat = Catalog()
questions = model.questions(cat, Triage, text_fact="ticket", min_confidence=0.6)

@cat.check(hard=True, then={"urgency": "critical"})         # a legal threat is critical, whatever the model says
def no_legal_threat(ticket) -> bool:
    return "lawyer" not in str(ticket).lower()
questions[1].requires.append("no_legal_threat")

res = System(cat, questions).ask({"ticket": {"subject": "Charged twice", "body": "Refund my double payment!",
                                             "customer": {"tier": "pro"}}})
print({q: (r.answer, r.status) for q, r in res.results.items()})
print(res.audit("team"))           # the probabilities, the model's fingerprint, what escalated and why
```

A state is read as key paths (`customer.tier: pro`); an unsure or escalated answer abstains with the reason
(`system.stats["model_escalated"]`, `["low_confidence"]`). [examples/15_typed_decisions.py](examples/15_typed_decisions.py)
runs the whole story with a stand-in model, without network. Whatever the model, read what it was measured on before
relying on it, and calibrate it on your own labelled stream (`part.act_guard`, under Building blocks): checks, constraints and
escalation are what make the answers safe to act on, not the model alone. The checkpoint contract of a local decider is in
[docs/decide_format.md](docs/decide_format.md); a local checkpoint answers one question per forward pass with the published checkpoints,
several with `DecideModel.load(..., multi_question=True)`.

Every answer is a value and a confidence, and the types also declare answer primitives: `Maybe[T]` ("not stated" —
`solvi.Unknown` — is a real answer, unlike an abstention), `Span[float]` (an exact piece of the text, parsed), `Rank[...]`
(the top k, in order), `Estimate[0, 7, 14]` (a number with an interval), and evidence quotes on any answer
(`Claim(value, evidence=[...])`, `Question(require_evidence=True)`) — each checked in the text, from rules or a model
([guide](docs/guide.md#answer-primitives-not-stated-evidence-spans-rankings-estimates),
[examples/16_primitives.py](examples/16_primitives.py)).

## Try it

- [solvi playground](https://huggingface.co/spaces/solvi-ai/playground): write a decision task in Python and run it, watch the
  strategist's plan, tamper with a trace and see the replay catch it, learn rules from examples. Its "New in 1.0" tab:
  `solvi.build` from labelled examples with `explain()`, `res.checks` and `then=` as a function of facts, quotes matched
  on a normalized view, the compact journal, the two levels.
- [solvi documents](https://huggingface.co/spaces/solvi-ai/documents): cited, typed answers from contracts, invoices, receipts,
  leases and more — the ModernBERT extractor (ONNX) and the decisions both run in your browser; add a field by describing it.
- [solvi arcade](https://huggingface.co/spaces/solvi-ai/arcade): game agents that explain every move — tic-tac-toe, maze,
  minesweeper, 20 questions, Mafia detective, a bot arena, and "hack the trace". Its "Agent and knowledge (1.0)" tab:
  `solvi.Agent` in a small world — run 1 explores (System 2), run 2 in the same world walks what it learned (System 1),
  the knowledge report with a retraction, and protection vs justified risk (`RiskBudget`) behind a gate that holds.
- [solvi realms](https://huggingface.co/spaces/solvi-ai/realms): an endless strategy game whose factions are solvi systems —
  tested for 100 000 turns: flat decision time (~0.3–0.6 ms), bounded memory and state, every sampled trace replay OK.
- All run **entirely in your browser** (Pyodide): no server, no GPU, nothing you type leaves the page. Each Space
  installs the solvi release it pins (1.0.0).
- Models: [solvi-ai/solvi-large](https://huggingface.co/solvi-ai/solvi-large) (typed decisions, 396M, preview),
  [solvi-ai/solvi-base](https://huggingface.co/solvi-ai/solvi-base) (the same answers on a CPU / in ONNX, 150M, preview),
  [solvi-ai/extract-base](https://huggingface.co/solvi-ai/extract-base) (fields by description) and
  [solvi-ai/extract-receipts](https://huggingface.co/solvi-ai/extract-receipts). Each model card states what the model was
  measured on, how well it does, and its limits; all models are listed at [huggingface.co/solvi-ai](https://huggingface.co/solvi-ai).

## Building blocks (`solvi.core`)

The ready systems are assembled from these; you can use them directly, and put a part of your own at every extension
point (a store, a slow path, a strategist, a head, a decider, an action model, ...).
[docs/building_blocks.md](docs/building_blocks.md) has one reference per protocol, with a conformance check for each.

### Quickstart (a catalog by hand, no model)

```python
from datetime import date
from solvi import Answer, Catalog, Question, System

cat = Catalog()

@cat.fn                                            # argument names = facts it reads; function name = fact it sets
def days_requested(start, end):
    return (end - start).days + 1

@cat.fn
def remaining_after(balance, days_requested):
    return balance - days_requested

@cat.check(hard=True, then={"approve": "reject"})  # if this check is False, "approve" is forced to "reject"
def enough_balance(remaining_after):
    return remaining_after >= 0

@cat.check
def enough_notice(start, today, days_requested):
    return days_requested < 5 or (start - today).days >= 14

@cat.rule("approve")
def approve(enough_notice):
    return "approve" if enough_notice else "needs_manager"

system = System(cat, [Question("approve", "Approve the leave?",
                               Answer.choice(["approve", "needs_manager", "reject"]),
                               requires=["enough_balance"])])
res = system.ask({"start": date(2026, 10, 19), "end": date(2026, 10, 23),
                  "today": date(2026, 9, 25), "balance": 14})
print(res["approve"].answer, res["approve"].confidence, res["approve"].why)
print(res.state_text())
print(res.trace.replay(system))
```

Output:

```
approve 1.0 enough_notice = True
days_requested           = 5
remaining_after          = 9
enough_balance           = True
enough_notice            = True
{'ok': True, 'steps': 5, 'mismatches': [], 'models': [], 'answers': 'same', 'catalog': 'same'}
```

With `"balance": 3` the hard check fails and the answer is `reject` with `status == "forced"`, whatever the rule says.
`solvi.show.show(res, cat)` prints answers, the planned flow, the computed state and the replay result in one go.

### How it works

- **Catalog.** `@cat.fn` (computation), `@cat.check` (bool), `@cat.extract` (value from text, returned as a `Quote` with
  offsets) and `@cat.rule(question)` (answer rule). A part's contract is its signature: argument names are the facts it
  reads, the function name is the fact it sets. Type hints are optional and become the facts' types
  (`def risk_score(risk_points: dict[str, float]) -> float`): producer and consumer types are checked when a part is
  registered, values are validated / coerced with pydantic at run time, and a value that fails is rejected like an
  ungrounded quote (safeguard `type_rejected`). Untyped parts cost nothing.
- **Questions.** `Question(name, text, Answer.yes_no() | Answer.choice([...]), requires=[...])`. Questions without a
  rule get a small answer head trained from labeled examples (`system.fit`) or a readable learned rule list
  (`system.learn_rule`).
- **Strategist.** For each question it walks backwards from the rule's arguments (or the learned features) through the
  catalog signatures to the keys of `init_state`, adds the question's required parts (`requires`) and every check that touches a computed
  fact. Everything else in the catalog is not executed; the flow records why each part was taken or skipped.
- **Execution.** Each part runs once, even if several questions need it. Hard checks and their inputs run first; when one
  fails, the steps only the settled questions needed are skipped (`res.trace.skipped`). With `System(..., workers=8)`
  independent steps run in parallel threads as soon as their inputs are ready. Results go into the computed state (`res.values`, `res.state_text()`) with their
  provenance; extracted values keep their quote. Each step record is hashed and chained to the previous one in flow order,
  so the trace does not depend on scheduling.
- **Answers and trace.** `res[q].answer / .confidence / .why / .status` (`ok`, `forced`, `abstain`), plus
  `res.trace.replay(system)`, which recomputes every step from recorded inputs and reports mismatches, broken hash links
  and quotes outside the text — and, given the System, a stored answer that is not the one the trace gives.

**Grounded decisions.** Fuzzy proposes, deterministic decides, everything is in the trace. Each fact and answer records
its provenance — `given`, `computed`, `quoted`, `decided` (a model's choice among options, with probabilities) or
`learned` — and model-backed steps record the model's id and fingerprint, so a replay can tell when the model changed
since a decision. A model's quote that is not literally the text at its offsets, or a choice outside its options, is
rejected and counted (`system.stats`); a fallback producer runs or the question abstains. `print(res.audit())` shows
what each answer rests on, which safeguards fired, and how much of its support is deterministic. A decision without
models and one with models are the same system ([examples/12_grounded_audit.py](examples/12_grounded_audit.py)).

### Guarantees, several models, serving

- **A guaranteed risk.** `part.act_guard(examples, max_risk=0.10)` calibrates on a few hundred labelled examples of
  your stream so that P(answered alone and wrong) ≤ 10% for inputs like them (conformal risk control) — a share of all
  inputs, not the error among the answers given alone (`calibrate_for(max_error=0.10, method="ltt")` bounds that); the
  audit shows the promise behind every answer, or says there is none. `part.conformal(examples)` gives the person who
  takes an escalation a short list of candidates. Near ties escalate (`min_margin=`), and the answer does not depend on
  the order the options are listed in (sorted by default).
- **Several models.** Any decider — an LLM, a System One service (Jev, Kev, Von, Laya-serve, …), a local checkpoint —
  combines with the others: `Cascade`, `Vote` and `Route` (`solvi.core.deciders.combine`) — the next model only when
  one escalates, an answer only when models of different families agree, or a model picked by code — under one
  guarantee ([examples/20_vote_across_families.py](examples/20_vote_across_families.py) shows a vote with stand-in
  servers).
- **Serving and operations.** `solvi serve module:system` exposes the questions over HTTP (OpenAPI from the same
  types), MCP and the System One API; `await system.aask(...)` runs async parts concurrently with timeouts.
  `TraceStorage` keeps decisions with a hash chain across them; `solvi diff` shows which stored decisions a rule or
  model change would flip; `solvi test`, `solvi check` and the honesty suite (`solvi honesty`) belong in CI;
  `res.report(format="html")` and `solvi report decisions.db --html out.html` give an auditor one page per decision or
  per period; `store.signature()` — 64 bytes kept next to the chain's head — later names the one stored record that
  was edited and restores its hash.
- **Text in.** `system.ask_text("please refund order A-10457, 1.5 million RUB, paid on 12 September 2026",
  textin=TextIn(system, decider, patterns={"order_id": r"A-\d+"}))`: the decider picks which question the message asks
  (or escalates when unsure), each input field is read with a quote (found by a deterministic cue finder — chosen over
  the checkpoint's span pointer by [measurement](docs/guide.md#text-in-from-a-message-to-a-question)) and a
  deterministic parser (numbers, dates, enums, yes / no; a date without a year is not guessed), missing required fields
  are listed for a clarifying question, and the trace says those values were read by a model, not given. `solvi serve`
  answers texts at `POST /ask_text` and as the MCP tool `ask_text`.
- **Long documents.** `decider.decision(..., long="retrieve")`: a contract longer than the decider reads is split into
  sections, BM25 picks the few that bear on the question, the decider reads only those, and quotes point into the whole
  document; the trace lists the sections read. `long="full"` reads a text whole up to the length a checkpoint trained
  on long inputs declares (`max_len_long`), and retrieves within that length beyond it.
- **Corrections.** `fit` heads refit on all kept examples as corrections accumulate;
  `solvi.core.knowledge.memory.attach(part)` escalates an answer when similar corrected cases say another one
  ([guide](docs/guide.md#a-memory-of-corrections-solvicoreknowledgememory)). The audit, `show` and the safeguard report
  render in Russian with `System(..., lang="ru")`.

### System 1 and System 2

A System under a guarantee is fast, cheap and knows when it is unsure: that is System 1. An LLM, a re-ask loop or a
search is slow and costs money per input: System 2. `solvi.build` and `solvi.Agent` put them in one system; underneath:

- **Who answers** (`solvi.core.dispatch`). System 1 is asked first; when its own signals say its answer cannot be given
  alone — below its guarantee, unlike the calibration examples, an abstention, a broken constraint — the slow path
  answers, and when that cannot either, or the budget is spent, a person gets the input with both candidates and the
  reasons. `Dispatcher.calibrate(examples, max_risk=...)` chooses, on each slice of what System 1 hands over, whether
  System 1's own guess, the slow path, their agreement or a person answers, under one promise. Every decision is one
  stored record with its cost and replays without calling a model
  ([guide](docs/guide.md#who-answers-system-1-the-slow-path-or-a-person-solvicoredispatch)).
- **What happened** (`System.report`, `solvi report decisions.db --overview`). From the store alone: who answered and
  how often each handed over, the time, calls, tokens and dollars, the promise in force next to the error the stored
  labels show, and drift ([guide](docs/guide.md#the-system-report-systemreport-solvi-report---overview)).
- **A showcase.** A player walks the world map of Pokémon Red (recorded from a real playthrough: place names and exits,
  no ROM) through the first fifteen goals twice; System 1 is two rules over remembered routes, System 2 a search over
  the world map it writes as it goes, and after each goal what System 2 found becomes System 1's routes. The first run
  takes 147 slow decisions of 183, the second 2 of 62, the fewest moves possible; every decision replays
  ([examples/23_pokemon_world_map.py](examples/23_pokemon_world_map.py), the replay viewer in
  [spaces/pokemon](spaces/pokemon)).

### Planning around dead ends and costs (code strategist)

The default strategist needs the inputs of every producer of a fact. `solvi.core.plan.cost.CostStrategist()` plans around
producers whose inputs are never given, and with `producers="equivalent"` picks the cheapest verified plan by declared
`cost=` (an exact 0/1 program; hard checks that govern a question always stay in the plan). No model is involved; the plan
is one hashed record in the trace and replay re-verifies it.

```python
from solvi.core.plan.cost import CostStrategist
system = System(cat, questions, strategist=CostStrategist(producers="equivalent"))
```

See [docs/strategist.md](docs/strategist.md) and [examples/17_cost_strategist.py](examples/17_cost_strategist.py).

## Extract from documents

With `solvi[model]`, fields are found by a fine-tuned ModernBERT extractor. The extracted value is always a span of the
document, so every answer built on it can be cited.

```python
from solvi.core.extract import LongSpanExtractor

ex = LongSpanExtractor.load("solvi-ai/extract-base")      # a field is a description; long texts in 1024-token windows
ex.fit([(text, "the total amount paid", (start, end)), ...], epochs=3)   # a few dozen labeled documents of your task
ex.save("my-extractor")
cat.extract(ex.field("total", "the total amount paid"))   # doc -> Quote(text, start, end, confidence), or no answer
cat.extract(ex.field("date", "the date of the purchase"))

@cat.fn
def amount(total):                                        # extracted values are strings; parse them in ordinary functions
    return float(total.replace(",", ""))
```

`LongSpanExtractor` reads the whole text in overlapping windows and supports "no answer" with a per-field threshold
(`ex.tune_threshold(name, held_out)`). What the published `solvi-ai/extract-base` does without labels of your task,
from its model card (held-out fields and data sets, one seed): CORD receipt fields it was never trained on, 46.5% and
67.9%; five never-trained CUAD clause types, 73.1%; Kleister-NDA and SROIE, never seen, 2–95% by field (addresses
2%). With per-field thresholds from 40 labeled contracts it reached 85.5% on CUAD; fine-tuned on 25–100 SROIE
receipts, 86–89%. So describing a field is a start, not a finished extractor: label 25–100 documents and fine-tune.
The receipt numbers below are a `LongSpanExtractor` too: `solvi-ai/extract-receipts` is `extract-base` fine-tuned on
CORD and loads with `LongSpanExtractor.load("solvi-ai/extract-receipts")` (its card says so;
[examples/07_receipts_model.py](examples/07_receipts_model.py) uses it). See
[docs/guide.md](docs/guide.md#extracting-fields-from-documents).

## Experimental (`solvi.experimental`)

Pieces that work and are tested but have not yet shown a measured gain. Importing one warns (`ExperimentalWarning`),
a decision that used one records it, and each one graduates or is removed by 1.2; `solvi.experimental.STATUS` and
[docs/experimental.md](docs/experimental.md) say what each is missing.

- **A policy text compiled into the catalog** (`solvi.experimental.compile`). An LLM writes plain functions, hard checks
  and rules from a policy; they are accepted without labelled examples only when every part cites its clauses, the code
  runs in a sandbox, two independent drafts agree on every generated input and tests derived from the text pass. A
  person settles what the drafts dispute, a changed text is recompiled with the stored decisions it moves listed.
  Agreement is not correctness: a misreading both drafts share is accepted
  ([guide](docs/guide.md#a-specification-compiled-into-the-catalog-solviexperimentalcompile)).
- **Behind a coding agent's hooks** (`solvi.experimental.hooks`). `solvi hook install` puts solvi in front of Claude
  Code's edits and prompts: every Edit / Write is checked against a rules file (forbidden patterns, required functions,
  Python calls read from the code; fuzzy questions for a model, which block only with a calibration) and denied with
  the rule and the lines, sent to the user, or let through; a prompt gets the one project skill it needs, or nothing
  ([guide](docs/guide.md#solvi-behind-a-coding-agents-hooks),
  [examples/22_coding_agent_hooks.py](examples/22_coding_agent_hooks.py)).
- **Verified charts** (`solvi.experimental.charts`): every number quoted from the text and checked (unit, scale, a pie
  that adds up), as a deterministic SVG that replays to the same bytes
  ([examples/21_verified_chart.py](examples/21_verified_chart.py)).
- **Learning** (`solvi.experimental.learning`, `solvi.experimental.lora`, `solvi.experimental.oncalib`): a loop that
  promotes an update from trusted corrections only when it passes held-out, honesty and calibration gates, with
  rollback; a LoRA adapter per question on solvi-base (`solvi[lora]`); recalibration on the fly from outcomes, which
  does not keep the promise.
- **The MCP proxy** (`solvi serve --guard catalog.py:guard --upstream CMD`: the guard in front of an MCP server) and
  **counterfactuals** (`solvi.experimental.counterfactual`: the smallest input change that flips an answer).

## Coming from 0.9

Every 0.9 import path still works in 1.0.x: it imports the same module and warns (`SolviDeprecationWarning`) with the
path to use; 1.1 removes the old paths. `solvi migrate PATH` rewrites your code (imports and dotted paths in strings,
in `.py` and `.md` files); `solvi migrate PATH --check` only reports. Stored decisions, calibration files and
fingerprints do not change. The full table and what was removed: [CHANGELOG.md](CHANGELOG.md#breaking-changes-and-migration).

## Command line

```bash
solvi init triage --with-model && cd triage     # a typed catalog, passing cases.json, README, CI workflow
solvi test . && solvi check catalog.py:system   # regression cases and the catalog lint (what CI runs)
solvi ask catalog.py:system example.json --audit            # one decision and what it rests on (--json, --report html)
solvi models pull solvi-ai/solvi-base           # a local decider for offline use; `solvi models` lists, `check` measures
solvi calibrate catalog.py:system route labels.csv --risk 0.1   # act_guard → route.calib.json, loaded by the catalog
solvi hook install                              # (experimental) Claude Code's edits checked against .claude/solvi-rules.toml
solvi migrate src/ --check                      # code still on the 0.9 import paths (without --check: rewrite it)
```

Every command is in the [guide](docs/guide.md#command-line).

## Results

Every number in this README comes from a script in [benchmarks/](benchmarks/) or from a published model card. Field
extraction: the [extract-receipts](https://huggingface.co/solvi-ai/extract-receipts) card reports 97.3% on typed questions
over CORD receipts (100 test receipts) with ECE 0.011 and 98.6% of questions answered at ≥ 99% precision, and the
[extract-base](https://huggingface.co/solvi-ai/extract-base) card its zero-shot and few-label numbers; the decision
models' cards ([solvi-base](https://huggingface.co/solvi-ai/solvi-base), [solvi-large](https://huggingface.co/solvi-ai/solvi-large))
theirs. On the gallery, the rules, quotes and checks are the answer: every answer is backed by a quote at stated offsets,
a computed fact, or the system abstains. Details: [docs/benchmarks.md](docs/benchmarks.md).

### Nine public tasks

Real tasks on public data, each with a baseline that does not use solvi and one solution with solvi, chosen on dev and
scored on a held-out split ([benchmarks/tasks/](benchmarks/tasks/), solvi 0.8.0, gpt-oss-120b where an LLM is used):

| Task | Baseline (no solvi) | solvi |
|---|---|---|
| CUAD contracts, 1,025 questions: accuracy; answered alone, wrong among them | 0.882; 100%, 11.8% | 0.899; 67.6%, 4.0% |
| Banking77 stream, 20 unseen intents from request 1,000, promise ≤ 5% wrong: wrong after the shift | 22.5% (broken) | 0.7% (kept), at 13.7% answered alone |
| Abt-Buy, 1,916 product pairs: F1 | 0.872 (LLM per pair) | 0.933 (code reads the offers; a head fitted on 5,743 labelled pairs) |
| NATURAL PLAN, right of 100: calendar / meetings / trips | 92 / 75 / 43 (LLM plans) | 95 / 100 / 98 (`solvi.core.slow.search`, no LLM) |
| BIRD mini-dev, 150 questions: right; wrong among answered | 78; 48.0% | 73; 34.8% at 74.7% answered |
| RAGTruth, 600 responses: F1 | 0.784 | 0.766 |
| τ-bench retail, 30 tasks: solved; calls the environment refused | 18; 10 | 14; 0 |
| NAB, 33 series: F1 | 0.391 | 0.361 |

solvi does not make a model more accurate: on RAGTruth, BIRD, τ-bench and NAB it did not beat the baseline. What it
added is the promise on what is answered alone, consistency across items, a search where a model guessed, and a stored,
replayable record of every decision. German Credit (rules and an audit of a rule change) decides exactly as plain code
does and adds the audit. Every number, its caveats and the cost of a run: [benchmarks/tasks/README.md](benchmarks/tasks/README.md).

### How it compares to asking an LLM

We gave the same inputs and written rules to solvi and to four LLMs: Grok 4.7, gpt-oss-120b, Qwen3-235B-2507 and
DeepSeek-V3.2. The sets were the gallery, generated refund and 3-way-match cases built to trip models up, and Banking77
message routing. Strong reasoning LLMs followed short written rules as accurately as solvi. solvi was not more accurate
there. What differs is everything around accuracy:

| Refunds, 576 decisions | solvi | Grok 4.7 | gpt-oss-120b | Qwen3-235B-2507 |
|---|---|---|---|---|
| accuracy | 0.894 in the run, 1.000 with 0.7.0's claim reader (not blind) | 1.000 | 1.000 | 0.844 |
| limit violations | 0 | 0 | 0 | 70 |
| answers changed by reordering the options | 0% | 0% | 0% | 9.6% |
| per decision | 0.6 ms, CPU | 2.0 s | 1.9 s | 1.1 s |
| $ per 1,000 decisions | $0 | $1.30 | $0.12 | $0.03 |

Inside solvi, no LLM broke a hard check, and wrong answers given without a person stayed within the promised 10%. On free
text the LLMs win: 0.925-0.972 on bank messages against 0.675 for solvi-large. A hosted decision model, Jev by TypeSafe,
read bank messages almost as well as the best LLM (0.964) but scored 0.819 on refunds with 53 limit violations; inside
solvi it broke no hard check either. An open one that reasons first, Jeeves by PostHog, came closer on the rules (0.863
on refunds, 0.912 on 3-way match) but still broke hard checks when asked directly. Full tables, caveats and a reproducible
runner with every raw answer: [docs/vs_llm.md](docs/vs_llm.md), [benchmarks/vs_llm/](benchmarks/vs_llm/).

## Speed

One `ask` without a model ([benchmarks/ask_speed.py](benchmarks/ask_speed.py): planning, running the parts and hashing
the trace; Python 3.10, Intel i7-12700H shared with other jobs, medians of three runs of 15 passes):

| case | steps | per ask |
|---|---:|---:|
| README quickstart (leave request) | 5 | 0.29 ms |
| 15 gallery entries (median entry; range) | 4-23 | 0.88 ms (0.64-5.6 ms) |
| random catalog of 50 parts, 5 questions | 19 | 0.95 ms |
| random catalog of 1 000 parts, 5 questions | 114 | 6.1 ms |
| stream alert over an input of 3 712 floats (the NAB task's catalog) | 7 | 3.7 ms |

A large input costs its hashing: every given value is written as canonical JSON and hashed once per ask (about 0.25 µs
per float). The slowest gallery entry checks its texts for instruction-like sentences (`perturb`), which takes most of
its time. Saving each response to a store adds its own cost (the whole response is written as JSON).

Strategist on random layered catalogs ([benchmarks/strategist_scale.py](benchmarks/strategist_scale.py), one CPU core;
solvi 0.7.1 with the changes since, Python 3.10, Intel i7-12700H, medians of two runs). The parts are trivial
arithmetic: running all of them ("run all") costs less than one ask, so the saving shows only when parts are slow
(services, models), as in the insurance desk below.

| catalog parts | plan | plan + run + trace | parts run | share of catalog | run all, no plan |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.14 ms | 1.1 ms | 31 | 31% | 0.03 ms |
| 1 000 | 1.2 ms | 4.6 ms | 114 | 11% | 0.36 ms |
| 10 000 | 11 ms | 25 ms | 232 | 2.3% | 4 ms |

Insurance claim desk with six slow services of 100-300 ms ([examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py)):

| questions asked | script computing everything | solvi, one by one | solvi, `workers=8` |
|---|---:|---:|---:|
| fast track? | 1 122 ms | 503 ms | 313 ms |
| full decision + payout | 1 122 ms | 773 ms | 461 ms |
| all five questions | 1 122 ms | 1 025 ms | 463 ms |
| expired policy (hard check settles it) | 1 122 ms | 152 ms | 153 ms |

A rule-only decision on a small catalog takes well under a millisecond ([benchmarks/ask_speed.py](benchmarks/ask_speed.py));
on documents the extractor dominates.

## When to use it

- The answer is a computation or a rule over a few values found in a document or a record: receipts, invoices,
  contracts, requests, orders.
- You need to show why: auditors, compliance, or a human reviewing low-confidence cases.
- Some rules are non-negotiable (hard checks), and the rest can be learned from about 100 labeled examples.

## When not to use it

- Open-ended free-text questions or generated answers. solvi answers typed questions only: yes/no, choices, scores,
  multi-label, "not stated", exact spans of the text, rankings and number ranges. Around a model that writes (a query,
  a plan, a JSON extraction) it checks, compares and re-asks — `solvi.core.slow.generate`, `solvi.core.slow.agree`, `solvi.core.slow.refine` — but
  does not make the writing better. It searches for a plan only over a space you enumerate (`solvi.core.slow.search`).
- New fields with no labeled examples. Extracting a field from its description alone is not reliable yet: the
  published extract-base gets 2–95% by field on data sets it never saw (see "Extract from documents"); label 25–100
  documents and fine-tune.
- No labels at all. Plan on roughly 100 labeled documents (field positions) per task.
- CPU-only deployment with a quantized extractor: dynamic int8 quantization changed half of extract-base's spans (its
  model card), so none is provided; use fp32 or fp16 and measure the time per document on your hardware.

## Gallery

[gallery/](gallery) — fifteen decision tasks across directions (support triage, email routing, content guard, security alerts,
AI-agent audit, release rollout, KYC/AML, clinical screening, credit with adverse-action reasons, procurement 3-way match,
double-charge refunds, predictive maintenance), each with scenarios, a runner and a side-by-side against an answer-only model.
Three are helpers for coding agents: a pre-edit rule check (allow / block / escalate a file write), review triage (quick
review only when seven risk questions are a confident "no", with a stated bound on risky changes that slip through) and a
skill picker with an honest "none".

## Examples

| File | What it shows |
|---|---|
| [examples/01_leave_request.py](examples/01_leave_request.py) | Leave request from a plain dict: rules, hard checks, parts the strategist skips |
| [examples/02_shop_order.py](examples/02_shop_order.py) | Shop order: two rule-based questions plus a "suspicious?" question learned from history with `fit` |
| [examples/03_invoices.py](examples/03_invoices.py) | Invoice approval from text: regex extractors with quotes, four questions, one learned |
| [examples/04_refunds.py](examples/04_refunds.py) | Refund e-mails: yes/no learned from examples under a hard "within 30 days" check |
| [examples/05_tic_tac_toe.py](examples/05_tic_tac_toe.py) | Tic-tac-toe agent: each move is an answer with its reason (win, block, fork, ...); a hard check rejects invalid boards |
| [examples/06_learned_rules.py](examples/06_learned_rules.py) | `learn_rule`: route parcels to delivery zones from free-form addresses with a readable if-then list learned from labels |
| [examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py) | Insurance claim desk: the strategist generates a different plan per question set, hard checks first with early exit, slow services in parallel; timed |
| [examples/10_learn_in_milliseconds.py](examples/10_learn_in_milliseconds.py) | `fit`: a new question learned in milliseconds, then corrected one example at a time (each correction ~0.2 ms, nothing retrained) |
| [examples/11_answer_types_and_constraints.py](examples/11_answer_types_and_constraints.py) | Multi-label and ordinal answers tied by constraints between answers; contradictions in learned answers are repaired by joint decoding |
| [examples/12_grounded_audit.py](examples/12_grounded_audit.py) | One catalog with and without models: provenance, `res.audit()`, a hallucinated quote caught by grounding, a decision outside its options, a model changed since the decision, lifetime safeguard stats |
| [examples/13_decide_model.py](examples/13_decide_model.py) | Support-email routing by a decider model as a catalog part: bias correction on unlabelled emails, 16 labelled examples, abstention, a constraint with a rule-based question, a hard check, the audit, `teach`, escalation for a target error rate, a JSON ticket (the real model with `SOLVI_DECIDE_MODEL`, a stand-in otherwise) |
| [examples/14_typed_catalog.py](examples/14_typed_catalog.py) | Typed facts: a pydantic request, type hints as fact types, answer types from the rules' return types (Enum, Literal, bool), a mismatch caught at registration, rejected values → fallback / abstention, a response as JSON that loads back and replays |
| [examples/15_typed_decisions.py](examples/15_typed_decisions.py) | Typed decisions: a pydantic ticket, the questions as a pydantic model's fields (choice, ordinal score, yes/no, multi-label), four answers (one forward pass when the model shares passes), a hard check, a constraint and a rule over the model, an escalation in the audit and stats (the real model with `SOLVI_DECIDE_MODEL`, a stand-in otherwise) |
| [examples/16_primitives.py](examples/16_primitives.py) | Answer primitives: "not stated" vs abstain, spans parsed into numbers, evidence quotes checked in the text (`require_evidence`), a ranking with scores, an estimate with an interval — from rules and from a decider with the answer-primitives contract; confidence per kind, JSON round trip, replay |
| [examples/17_cost_strategist.py](examples/17_cost_strategist.py) | The code strategist: dead ends dropped, the cheapest verified plan by declared costs, the plan in the trace |
| [examples/18_several_models.py](examples/18_several_models.py) | Several models, one decision: a cascade small → large, a vote of two model families, a route by code — each under one `act_guard` guarantee, with cost per question; every stage in the audit and the trace |
| [examples/19_agent_guard.py](examples/19_agent_guard.py) | An accounts-payable agent's tool calls through a `Guard`: grounded arguments, an invented IBAN denied, a budget escalation approved by a person, an instruction hidden in an invoice, an authorizer with `act_guard` and `perturb`; every decision stored and replayed (a scripted agent, no API keys) |
| [examples/20_vote_across_families.py](examples/20_vote_across_families.py) | A vote of two model families behind the System One API (stand-in servers started in-process): each alone and the vote under one `act_guard` guarantee; a sure mistake of one family escalates; the audit and the replay |
| [examples/21_verified_chart.py](examples/21_verified_chart.py) | A verified chart (`solvi.experimental.charts`, experimental): every number quoted from the text and checked; a careless model's swapped digit, invented share and unquoted value dropped with reasons; a deterministic SVG that replays to identical bytes |
| [examples/22_coding_agent_hooks.py](examples/22_coding_agent_hooks.py) | A coding agent's session behind `solvi hook`: the hooks installed in a temporary project, a clean edit allowed, an edit that takes an employee id from the browser denied with the rule and the line, a migration with an empty downgrade and a comment that tries to talk past the rules denied, a skill picked for one prompt and none for another; the store verified and one decision audited and replayed |
| [examples/23_pokemon_world_map.py](examples/23_pokemon_world_map.py) | System 1 and System 2 on the world map of Pokémon Red (recorded, no ROM): rules over remembered routes, a search over the player's world map when they are unsure or surprised, routes compiled after each goal; 147 → 2 slow decisions from the first run to the second; every decision stored, replayed and reported (`System.report`) |
| [examples/24_one_entry_point.py](examples/24_one_entry_point.py) | one entry point (`solvi.build`): System 1 fitted from labelled examples, its guarantee and the slow path's slice calibrated on examples it did not see, every decision stored; `explain()` prints the choices (no model) |
| [examples/25_environment_agent.py](examples/25_environment_agent.py) | An environment agent (`solvi.Agent`) with `solvi.Knowledge` on a toy crafting world: System 1 acts on what the action model and the skills predict, System 2 searches, the agenda's gates are hard checks; the same world met again takes 12 steps instead of 61 (11 by System 1), and protection vs `RiskBudget` at a bridge that breaks one time in three; every decision replays |
| [examples/07_receipts_model.py](examples/07_receipts_model.py) | Expense check on a scanned receipt: a receipts-tuned extractor cites each field, rules and a hard check decide (needs `solvi[model]`) |
| [examples/08_contracts_by_description.py](examples/08_contracts_by_description.py) | Contract review with fields defined only in words: the general extractor reads the whole contract, cites clauses or says "absent" (needs `solvi[model]`) |

Run them from a clone: `python examples/01_leave_request.py`.

## More

- [docs/using.md](docs/using.md): the ready systems — `build`, `Agent`, `Guard`, `Knowledge` and `solvi.models`.
- [docs/agent.md](docs/agent.md): agents and knowledge, with what was and was not shown.
- [docs/building_blocks.md](docs/building_blocks.md): the extension points of `solvi.core`.
- [docs/experimental.md](docs/experimental.md): what is experimental and what each piece is missing.
- [docs/guide.md](docs/guide.md): the reference guide, in the same three parts.
- [docs/best_practices.md](docs/best_practices.md): what we learned while building on solvi, as advice.
- [docs/mission.md](docs/mission.md): what solvi is for, what it is not, and its honest limits.
- [docs/decide_format.md](docs/decide_format.md): the decider checkpoint contract (for training your own).
- [docs/benchmarks.md](docs/benchmarks.md): the scripts behind the numbers, what solvi adds to an ask, the dataset loaders.
- [docs/vs_llm.md](docs/vs_llm.md): solvi vs asking an LLM (Grok 4.7, gpt-oss-120b, Qwen3, DeepSeek), with raw answers.
- [ROADMAP.md](ROADMAP.md), [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md).
- [benchmarks/](benchmarks/): the benchmark scripts that can be rerun, and the dataset loaders behind the extraction
  numbers (SROIE, CORD, CUAD, Kleister-NDA — their scripts are not in the repository).
- Tests: `pytest`.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

A paper is in preparation. Until then, please cite the repository:

```bibtex
@software{solvi,
  title  = {solvi: verifiable decision systems from catalogs of functions and checks},
  author = {mxkuzn and solvi contributors},
  year   = {2026},
  url    = {https://github.com/solvi-ai/solvi}
}
```
