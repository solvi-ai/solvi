# solvi

[![PyPI](https://img.shields.io/pypi/v/solvi.svg)](https://pypi.org/project/solvi/)
[![CI](https://github.com/solvi-ai/solvi/actions/workflows/ci.yml/badge.svg)](https://github.com/solvi-ai/solvi/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-solvi--ai-yellow)](https://huggingface.co/solvi-ai)

Build decision systems from a catalog of Python functions and checks plus typed questions, and get answers you can verify.

## Why

You describe a task with plain Python functions (computations, checks, answer rules) and questions with typed answers
(yes/no, a choice, a score, "not stated", a span of the text, a ranking, a number range). For each request, a strategist
plans which functions and checks to run for the asked questions. Every answer comes with:

- a **confidence** (calibratable per question);
- a **reason you can check**: the rule inputs, a formula over computed facts, or a quote with character offsets in the
  source document;
- a **hash-chained trace** of every step, which can be re-executed later to confirm the answer or pinpoint the step that
  was altered.

When something cannot be computed, a function fails, or a rule returns an answer outside the allowed options, solvi
abstains instead of guessing. A failed hard check always overrides any model confidence.

**Grounded decisions.** Fuzzy proposes, deterministic decides, everything is in the trace. Each fact and answer records its
provenance — `given`, `computed`, `quoted`, `decided` (a model's choice among options, with probabilities) or `learned` —
and model-backed steps record the model's id and fingerprint, so a replay can tell when the model changed since a decision.
A model's quote that is not literally the text at its offsets, or a choice outside its options, is rejected and counted
(`system.stats`); a fallback producer runs or the question abstains. `print(res.audit())` shows what each answer rests on,
which safeguards fired, and how much of its support is deterministic. A decision without models and one with models are the
same system ([examples/12_grounded_audit.py](examples/12_grounded_audit.py)).

And it is fast. The strategist plans a flow over a 10 000-part catalog in about 6 ms and runs only the parts the questions
need (2.4% of that catalog). Hard checks run first, so a failing one skips the expensive rest; independent slow parts (API
calls, model inference) run in parallel. On an insurance-claim desk with slow services
([examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py)) a full decision takes 463 ms instead of 1 122 ms for a
script that computes everything, and 152 ms when an expired policy settles the claim first.

## Try it

- [solvi playground](https://huggingface.co/spaces/solvi-ai/playground): write a decision task in Python and run it, watch the
  strategist's plan, tamper with a trace and see the replay catch it, learn rules from examples.
- [solvi documents](https://huggingface.co/spaces/solvi-ai/documents): cited, typed answers from contracts, invoices, receipts,
  leases and more — the ModernBERT extractor (ONNX) and the decisions both run in your browser; add a field by describing it.
- [solvi arcade](https://huggingface.co/spaces/solvi-ai/arcade): game agents that explain every move — tic-tac-toe, maze,
  minesweeper, 20 questions, Mafia detective, a bot arena, and "hack the trace".
- [solvi realms](https://huggingface.co/spaces/solvi-ai/realms): an endless strategy game whose factions are solvi systems —
  tested for 100 000 turns: flat decision time (~0.3–0.6 ms), bounded memory and state, every sampled trace replay OK.
- All run **entirely in your browser** (Pyodide): no server, no GPU, nothing you type leaves the page.
- Models: [solvi-ai/solvi-large](https://huggingface.co/solvi-ai/solvi-large) (typed decisions, 396M, preview),
  [solvi-ai/solvi-base](https://huggingface.co/solvi-ai/solvi-base) (the same answers on a CPU / in ONNX, 150M, preview),
  [solvi-ai/extract-base](https://huggingface.co/solvi-ai/extract-base) (fields by description) and
  [solvi-ai/extract-receipts](https://huggingface.co/solvi-ai/extract-receipts). Each model card states what the model was
  measured on, how well it does, and its limits; all models are listed at [huggingface.co/solvi-ai](https://huggingface.co/solvi-ai).

## Gallery

[gallery/](gallery) — fifteen decision tasks across directions (support triage, email routing, content guard, security alerts,
AI-agent audit, release rollout, KYC/AML, clinical screening, credit with adverse-action reasons, procurement 3-way match,
double-charge refunds, predictive maintenance), each with scenarios, a runner and a side-by-side against an answer-only model.
Three are helpers for coding agents: a pre-edit rule check (allow / block / escalate a file write), review triage (quick
review only when seven risk questions are a confident "no", with a stated bound on risky changes that slip through) and a
skill picker with an honest "none".

## Install

```bash
pip install solvi              # core: rules, checks, learned answer heads (numpy, scipy, pydantic)
pip install "solvi[model]"     # + torch, transformers: ModernBERT field extractors for documents and the decider
pip install "solvi[onnx]"      # + onnxruntime, tokenizers: the decider (solvi.decide) on CPU without torch
pip install "solvi[serve]"     # + fastapi, uvicorn: `solvi serve app.py:system` — the questions over HTTP (also --mcp)
pip install "solvi[otel]"      # + opentelemetry: decisions as OpenTelemetry spans (solvi.otel)
pip install "solvi[duckdb]"    # + duckdb: stored decisions in a DuckDB file (solvi.DuckDBStorage); [postgres] for PostgreSQL
pip install "solvi[lora]"      # + torch, transformers, peft: part.adapt_lora, a LoRA adapter per question (experimental)
```

`solvi.agents` (guarding an agent's tool calls) needs only the core; its adapters use the PydanticAI, LangGraph or OpenAI
Agents SDK you already have.

Requires Python 3.10+.

## Quickstart (core only, no model)

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
                               checkpoints=["enough_balance"])])
res = system.ask({"start": date(2026, 10, 19), "end": date(2026, 10, 23),
                  "today": date(2026, 9, 25), "balance": 14})
print(res["approve"].answer, res["approve"].confidence, res["approve"].why)
print(res.computed_state)
print(res.trace.replay(cat))
```

Output:

```
approve 1.0 enough_notice = True
days_requested           = 5
remaining_after          = 9
enough_balance           = True
enough_notice            = True
{'ok': True, 'steps': 5, 'mismatches': [], 'models': [], 'catalog': 'same'}
```

With `"balance": 3` the hard check fails and the answer is `reject` with `status == "forced"`, whatever the rule says.
`solvi.show.show(res, cat)` prints answers, the planned flow, the computed state and the replay result in one go.

## Command line

```bash
solvi init triage --with-model && cd triage     # a typed catalog, passing cases.json, README, CI workflow
solvi test . && solvi check catalog.py:system   # regression cases and the catalog lint (what CI runs)
solvi ask catalog.py:system example.json --audit            # one decision and what it rests on (--json, --report html)
solvi models pull solvi-ai/solvi-base           # the only command that downloads; `solvi models` lists, `check` measures
solvi calibrate catalog.py:system route labels.csv --risk 0.1   # act_guard → route.calib.json, loaded by the catalog
solvi hook install                              # Claude Code's edits checked against .claude/solvi-rules.toml
```

Every command is in the [guide](docs/guide.md#command-line).

## How it works

- **Catalog.** `@cat.fn` (computation), `@cat.check` (bool), `@cat.extract` (value from text, returned as a `Quote` with
  offsets) and `@cat.rule(question)` (answer rule). A part's contract is its signature: argument names are the facts it
  reads, the function name is the fact it sets. Type hints are optional and become the facts' types
  (`def risk_score(risk_points: dict[str, float]) -> float`): producer and consumer types are checked when a part is
  registered, values are validated / coerced with pydantic at run time, and a value that fails is rejected like an
  ungrounded quote (safeguard `type_rejected`). Untyped parts cost nothing.
- **Questions.** `Question(name, text, Answer.yes_no() | Answer.choice([...]), checkpoints=[...])`. Questions without a
  rule get a small answer head trained from labeled examples (`system.fit`) or a readable learned rule list
  (`system.learn_rule`).
- **Strategist.** For each question it walks backwards from the rule's arguments (or the learned features) through the
  catalog signatures to the keys of `init_state`, adds the question's checkpoints and every check that touches a computed
  fact. Everything else in the catalog is not executed; the flow records why each part was taken or skipped.
- **Execution.** Each part runs once, even if several questions need it. Hard checks and their inputs run first; when one
  fails, the steps only the settled questions needed are skipped (`res.trace.skipped`). With `System(..., workers=8)`
  independent steps run in parallel threads as soon as their inputs are ready. Results go into `computed_state` with their
  provenance; extracted values keep their quote. Each step record is hashed and chained to the previous one in flow order,
  so the trace does not depend on scheduling.
- **Answers and trace.** `res[q].answer / .confidence / .why / .status` (`ok`, `forced`, `abstain`), plus
  `res.trace.replay(catalog)`, which recomputes every step from recorded inputs and reports mismatches, broken hash links
  and quotes outside the text.

## Typed decisions with a model

Types declare questions; the model proposes; checks decide. The fields of a pydantic model are the questions, their types
the kinds (one option, several, an ordered score, yes/no); a decider (`solvi.decide`, `solvi[onnx]` or `solvi[model]`)
answers them about a text or a JSON / pydantic state — several in one forward pass when the checkpoint can — with
probabilities, a calibrated confidence and act / escalate. Hard checks, constraints and rules still decide.

```python
from typing import Literal
from pydantic import BaseModel, Field
from solvi import Catalog, Scale, System
from solvi.decide import DecideModel

class Triage(BaseModel):
    team: Literal["billing", "technical", "shipping"] = Field(description="Which team should handle this ticket?")
    urgency: Scale[Literal["low", "medium", "high", "critical"]] = Field(description="How urgent is it?")
    angry: bool = Field(description="Is the customer angry?")
    topics: list[Literal["refund", "delay", "bug"]] = Field(description="What does the ticket mention?")

model = DecideModel.load("solvi-ai/solvi-base")         # or a local folder; solvi_decide.json says what it can do
cat = Catalog()
questions = model.questions(cat, Triage, text_fact="ticket", escalate_below=0.6)

@cat.check(hard=True, then={"urgency": "critical"})         # a legal threat is critical, whatever the model says
def no_legal_threat(ticket) -> bool:
    return "lawyer" not in str(ticket).lower()
questions[1].checkpoints.append("no_legal_threat")

res = System(cat, questions).ask({"ticket": {"subject": "Charged twice", "body": "Refund my double payment!",
                                             "customer": {"tier": "pro"}}})
print({q: (r.answer, r.status) for q, r in res.results.items()})
print(res.audit("team"))           # probabilities, the model's fingerprint, the shared pass, what escalated and why
```

A state is read as key paths (`customer.tier: pro`), the format the decider is trained on; an unsure or escalated answer
abstains with the reason (`system.stats["model_escalated"]`, `["low_confidence"]`). The checkpoint contract is in
[docs/decide_format.md](docs/decide_format.md); [examples/15_typed_decisions.py](examples/15_typed_decisions.py) runs the
whole story with a stand-in model. Published deciders are previews: read the model card before relying on one, and fit
it on 30–60 labelled examples of your task (`part.fit`, `part.calibrate_for`) — checks, constraints and escalation are what
make the answers safe to act on, not the model alone.

Every answer is a value and a confidence, and the types also declare answer primitives: `Maybe[T]` ("not stated" —
`solvi.Unknown` — is a real answer, unlike an abstention), `Span[float]` (an exact piece of the text, parsed), `Rank[...]`
(the top k, in order), `Estimate[0, 7, 14]` (a number with an interval), and evidence quotes on any answer
(`Claim(value, evidence=[...])`, `Question(require_evidence=True)`) — each checked in the text, from rules or a model
([guide](docs/guide.md#answer-primitives-not-stated-evidence-spans-rankings-estimates),
[examples/16_primitives.py](examples/16_primitives.py)).

## Escalation with a guarantee, several models, serving

- **A guaranteed risk.** `part.act_guard(examples, risk=0.10)` calibrates on a few hundred labelled examples of your stream
  so that P(answered alone and wrong) ≤ 10% for inputs like them (conformal risk control); the audit shows the promise
  behind every answer, or says there is none. `part.conformal(examples)` gives the person who takes an escalation a short
  list of candidates. Near ties escalate (`min_margin=`), and the answer does not depend on the order the options are listed
  in (sorted by default).
- **Any decision model.** `solvi.systemone.systemone(url, model)` puts any `POST /v1/systemone` service (Jev, Kev, Von,
  Laya-serve, …) behind your rules, and `solvi.llm.llm(base_url, model)` any OpenAI-compatible LLM server (OpenAI,
  OpenRouter, vLLM, llama.cpp, Ollama) — its JSON replies validated, an invalid one escalated, never guessed;
  `Cascade`, `Vote` and `Route` (`solvi.multi`) combine models — the next model only when one escalates, an answer
  only when models of different families agree, or a model picked by code — under one guarantee. On
  typed-decisions a vote of solvi-large and Julia 1 answered 50% alone against 31% / 40% for each alone, at the same
  10% risk (Julia in-distribution there; [examples/20_vote_across_families.py](examples/20_vote_across_families.py)).
- **Serving and operations.** `solvi serve module:system` exposes the questions over HTTP (OpenAPI from the same types),
  MCP and the System One API; `await system.aask(...)` runs async parts concurrently with timeouts; `costs="measured"`
  lets the planner pick the fastest equivalent source and switch when it slows down. `TraceStorage` keeps decisions with a
  hash chain across them; `solvi diff` shows which stored decisions a rule or model change would flip; `solvi test`,
  `solvi check` and the honesty suite (`solvi honesty`) belong in CI; `res.report(format="html")` and
  `solvi report decisions.db --html out.html` give an auditor one page per decision or per period, and `solvi.otel.export(res)` puts every step in
  your OpenTelemetry traces. `res.counterfactual("approve")` says what would have changed the answer ("approve if
  amount ≤ 1000 (now 1200)"), re-running only the code with the models' recorded proposals held.
- **Guarding an agent's tool calls (preview).** The agent proposes `{"name": tool, "arguments": {...}}`; `solvi.agents.Guard` checks
  it — the tool is in the catalog, the arguments validate against its types, the values that must come from the
  conversation are quoted there (and not only from a tool output that says "ignore previous instructions"), your policies
  (limits, roles, allow-lists) are ordinary hard checks, and an optional decider asks "did the user ask for this?" under
  `act_guard` and `perturb` — then allows it (solvi runs the function), denies it with the reasons, or escalates it to a
  person. Every decision is a stored, replayable trace. Adapters for PydanticAI, LangGraph and the OpenAI Agents SDK, and
  `solvi serve --guard catalog.py:guard --upstream CMD` in front of an MCP server
  ([guide](docs/guide.md#guarding-an-agents-tool-calls), [examples/19_agent_guard.py](examples/19_agent_guard.py)).
- **Behind a coding agent's hooks (preview).** `solvi hook install` puts solvi in front of Claude Code's edits and prompts:
  every Edit / Write is checked against a rules file (forbidden patterns, required functions, Python calls read from the
  code; fuzzy questions for a model, which block only with a calibration) and denied with the rule and the lines, sent
  to the user, or let through; a prompt gets the one project skill it needs, or nothing. Deterministic by default, about
  0.1 s a call; every decision stored and verifiable; Codex as a preview
  ([guide](docs/guide.md#solvi-behind-a-coding-agents-hooks), [examples/22_coding_agent_hooks.py](examples/22_coding_agent_hooks.py)).
- **Text in.** `system.ask_text("please refund order A-10457, 1.5 million rubles, paid 12 September", decider)`: the
  decider picks which question the message asks (or escalates when unsure), each input field is read with a quote and a
  deterministic parser (numbers, dates, enums, yes / no), missing required fields are listed for a clarifying question,
  and the trace says those values were read by a model, not given. `solvi serve` answers texts at `POST /ask_text`
  and as the MCP tool `ask_text`.
- **Long documents.** `decider.decision(..., long="retrieve")`: a contract longer than the decider reads is split into
  sections, BM25 picks the few that bear on the question, the decider reads only those, and quotes point into the whole
  document; the trace lists the sections read. `long="full"` reads a text whole up to the length a checkpoint trained on
  long inputs declares (`max_len_long`), and retrieves within that length beyond it.
- **Learning from corrections.** `part.memory()` escalates an answer when similar corrected cases say another one;
  `fit_fast` heads refit on all kept examples as corrections accumulate; `part.adapt_lora(examples, holdout=0.3)` trains
  a small LoRA adapter for one question on solvi-base once it has ~100 labelled answers (`solvi[lora]`, experimental);
  `System.learning(store)` proposes updates from trusted corrections only and promotes one when it passes held-out,
  honesty and calibration gates, with rollback (experimental, off unless called)
  ([guide](docs/guide.md#a-memory-of-corrections-partmemory)).
- **Records you can check later.** `store.signature()` — 64 bytes kept next to the chain's head — later names the one
  stored record that was edited and restores its hash (preview). `solvi.charts` draws a chart in which every number is
  quoted from the text and checked (unit, scale, a pie that adds up), as a deterministic SVG that replays to the same
  bytes (preview; [examples/21_verified_chart.py](examples/21_verified_chart.py)). The audit, `show` and the safeguard
  report render in Russian with `System(..., lang="ru")`.

## Planning around dead ends and costs (code strategist)

The default strategist needs the inputs of every producer of a fact. `solvi.strategy.ModelStrategist()` plans around
producers whose inputs are never given, and with `producers="equivalent"` picks the cheapest verified plan by declared
`cost=` (an exact 0/1 program; hard checks that govern a question always stay in the plan). No model is involved; the plan
is one hashed record in the trace and replay re-verifies it.

```python
from solvi.strategy import ModelStrategist
system = System(cat, questions, strategist=ModelStrategist(producers="equivalent"))
```

A segment model that proposes producers when costs are not declared, and `solvi.aliases` (wiring parameter names that match
no fact), ship as **experimental**; their weights are not published. See [docs/strategist.md](docs/strategist.md) and
[examples/17_model_strategist.py](examples/17_model_strategist.py).

## Extract from documents

With `solvi[model]`, fields are found by a fine-tuned ModernBERT extractor. The extracted value is always a span of the
document, so every answer built on it can be cited.

```python
from solvi.extract_multi import MultiSpanExtractor

ex = MultiSpanExtractor(["total", "date"])          # ModernBERT-large, one pass per document for all fields
ex.fit(train_docs, train_spans, epochs=4)           # train_spans: [{"total": (start, end), "date": (start, end) | None}]
cat.extract(ex.field("total"))                      # doc -> Quote(text, start, end, confidence)
cat.extract(ex.field("date"))

@cat.fn
def amount(total):                                  # extracted values are strings; parse them in ordinary functions
    return float(total.replace(",", ""))
```

For long documents (contracts), `solvi.extract_long.LongSpanExtractor` reads the whole text in overlapping 1024-token
windows, takes a field description instead of a fixed field list, and supports "no answer" with a per-field threshold.
See [docs/guide.md](docs/guide.md#extracting-fields-from-documents).

## Results

Same training documents for both sides. The baseline, Laya, is a ModernBERT-large model that answers the typed questions
directly, fine-tuned with its authors' recipe. Details and caveats: [docs/benchmarks.md](docs/benchmarks.md).

| Task (test set) | solvi | Baseline |
|---|---|---|
| SROIE receipts, 6 questions (361 receipts) | **97.8%** | 92.6% |
| SROIE, only 100 labeled training receipts | **95.9%** | 80.0% |
| CORD receipts, 4 questions (100 receipts) | **98.5%** | 96.0% |
| CORD, share of questions answered at >= 99% precision | **99.7%** | 14.2% |
| CUAD contracts, 5 questions (102 contracts, median 33k chars) | **94.8%** | 77.5% (sees first 1024 tokens only) |

- Calibrated confidence: ECE 0.008 (SROIE), 0.011 (CORD), 0.027 (CUAD).
- On the document benchmarks, 100% of answers are backed by a quote at stated offsets, or the system abstains.
- Speed: about 0.4 ms per decision when no model is involved; about 39 ms per receipt with the one-pass extractor on an
  A100 GPU.

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
solvi it broke no hard check either. Full tables, caveats and a reproducible
runner with every raw answer: [docs/vs_llm.md](docs/vs_llm.md), [benchmarks/vs_llm/](benchmarks/vs_llm/).

## Speed

Strategist on random layered catalogs ([benchmarks/strategist_scale.py](benchmarks/strategist_scale.py), one CPU core):

| catalog parts | plan | plan + run + trace | parts run | share of catalog |
|---:|---:|---:|---:|---:|
| 100 | 0.2 ms | 1.3 ms | 31 | 31% |
| 1 000 | 0.7 ms | 2.6 ms | 115 | 12% |
| 10 000 | 6 ms | 10 ms | 236 | 2.4% |

Insurance claim desk with six slow services of 100-300 ms ([examples/09_strategy_at_scale.py](examples/09_strategy_at_scale.py)):

| questions asked | script computing everything | solvi, one by one | solvi, `workers=8` |
|---|---:|---:|---:|
| fast track? | 1 122 ms | 503 ms | 313 ms |
| full decision + payout | 1 122 ms | 773 ms | 461 ms |
| all five questions | 1 122 ms | 1 025 ms | 463 ms |
| expired policy (hard check settles it) | 1 122 ms | 152 ms | 153 ms |

A rule-only decision on a small catalog takes well under a millisecond; on documents the extractor dominates (about 39 ms per
receipt with the one-pass extractor on an A100).

## When to use it

- The answer is a computation or a rule over a few values found in a document or a record: receipts, invoices,
  contracts, requests, orders.
- You need to show why: auditors, compliance, or a human reviewing low-confidence cases.
- Some rules are non-negotiable (hard checks), and the rest can be learned from about 100 labeled examples.

## When not to use it

- Open-ended free-text questions or generated answers. solvi answers typed questions only: yes/no, choices, scores,
  multi-label, "not stated", exact spans of the text, rankings and number ranges.
- New fields with no labeled examples. Extracting a field from its description alone does not work yet (14% and 66% on
  two held-out fields); a universal extractor is coming.
- No labels at all. Plan on roughly 100 labeled documents (field positions) per task.
- CPU-only deployment with a quantized model: the int8 ONNX extractor loses up to 12 points on amounts, company
  names and addresses. fp32 on CPU keeps accuracy but takes about 0.7 s per receipt on 2 cores.

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
| [examples/10_learn_in_milliseconds.py](examples/10_learn_in_milliseconds.py) | `fit_fast`: a new question learned in milliseconds, then corrected one example at a time (each correction ~0.2 ms, nothing retrained) |
| [examples/11_answer_types_and_constraints.py](examples/11_answer_types_and_constraints.py) | Multi-label and ordinal answers tied by constraints between answers; contradictions in learned answers are repaired by joint decoding |
| [examples/12_grounded_audit.py](examples/12_grounded_audit.py) | One catalog with and without models: provenance, `res.audit()`, a hallucinated quote caught by grounding, a decision outside its options, a model changed since the decision, lifetime safeguard stats |
| [examples/13_decide_model.py](examples/13_decide_model.py) | Support-email routing by a decider model as a catalog part: bias correction on unlabelled emails, 16 labelled examples, abstention, a constraint with a rule-based question, a hard check, the audit, `teach`, escalation for a target error rate, a JSON ticket (the real model with `SOLVI_DECIDE_MODEL`, a stand-in otherwise) |
| [examples/14_typed_catalog.py](examples/14_typed_catalog.py) | Typed facts: a pydantic request, type hints as fact types, answer types from the rules' return types (Enum, Literal, bool), a mismatch caught at registration, rejected values → fallback / abstention, a response as JSON that loads back and replays |
| [examples/15_typed_decisions.py](examples/15_typed_decisions.py) | Typed decisions: a pydantic ticket, the questions as a pydantic model's fields (choice, ordinal score, yes/no, multi-label), four answers from one forward pass, a hard check, a constraint and a rule over the model, an escalation in the audit and stats (the real model with `SOLVI_DECIDE_MODEL`, a stand-in otherwise) |
| [examples/16_primitives.py](examples/16_primitives.py) | Answer primitives: "not stated" vs abstain, spans parsed into numbers, evidence quotes checked in the text (`require_evidence`), a ranking with scores, an estimate with an interval — from rules and from a decider with the answer-primitives contract; confidence per kind, JSON round trip, replay |
| [examples/17_model_strategist.py](examples/17_model_strategist.py) | The code strategist: dead ends dropped, the cheapest verified plan by declared costs, a model's proposal checked and rejected; aliases for names that match no fact (experimental; stand-ins without weights) |
| [examples/18_several_models.py](examples/18_several_models.py) | Several models, one decision: a cascade small → large, a vote of two model families, a route by code — each under one `act_guard` guarantee, with cost per question; every stage in the audit and the trace |
| [examples/19_agent_guard.py](examples/19_agent_guard.py) | An accounts-payable agent's tool calls through a `Guard`: grounded arguments, an invented IBAN denied, a budget escalation approved by a person, an instruction hidden in an invoice, an authorizer with `act_guard` and `perturb`; every decision stored and replayed (a scripted agent, no API keys) |
| [examples/20_vote_across_families.py](examples/20_vote_across_families.py) | A vote of two model families behind the System One API (stand-in servers started in-process): each alone and the vote under one `act_guard` guarantee; a sure mistake of one family escalates; the audit and the replay |
| [examples/21_verified_chart.py](examples/21_verified_chart.py) | A verified chart (`solvi.charts`, preview): every number quoted from the text and checked; a careless model's swapped digit, invented share and unquoted value dropped with reasons; a deterministic SVG that replays to identical bytes |
| [examples/22_coding_agent_hooks.py](examples/22_coding_agent_hooks.py) | A coding agent's session behind `solvi hook`: the hooks installed in a temporary project, a clean edit allowed, an edit that takes an employee id from the browser denied with the rule and the line, a migration with an empty downgrade and a comment that tries to talk past the rules denied, a skill picked for one prompt and none for another; the store verified and one decision audited and replayed |
| [examples/07_receipts_model.py](examples/07_receipts_model.py) | Expense check on a scanned receipt: a receipts-tuned extractor cites each field, rules and a hard check decide (needs `solvi[model]`) |
| [examples/08_contracts_by_description.py](examples/08_contracts_by_description.py) | Contract review with fields defined only in words: the general extractor reads the whole contract, cites clauses or says "absent" (needs `solvi[model]`) |

Run them from a clone: `python examples/01_leave_request.py`.

## More

- [docs/guide.md](docs/guide.md): full API walkthrough.
- [docs/decide_format.md](docs/decide_format.md): the decider checkpoint contract (for training your own).
- [docs/strategist.md](docs/strategist.md): the code strategist and the experimental model strategist and name matching.
- [ROADMAP.md](ROADMAP.md), [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md).
- [docs/benchmarks.md](docs/benchmarks.md): setups, per-question numbers, caveats.
- [docs/vs_llm.md](docs/vs_llm.md): solvi vs asking an LLM (Grok 4.7, gpt-oss-120b, Qwen3, DeepSeek), with raw answers.
- [benchmarks/](benchmarks/): dataset loaders and benchmark scripts (SROIE, CORD, CUAD, Kleister-NDA).
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
